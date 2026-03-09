"""
Stage 2 Arm Policy Wrapper
===========================
Loads the Stage 2 arm actor (right arm, 39 obs -> 7 act)
and runs inference for the hierarchical control pipeline.

7 right arm joints controlled:
  right_shoulder_pitch_joint, right_shoulder_roll_joint,
  right_shoulder_yaw_joint, right_elbow_joint,
  right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint

Network: 39 -> [256, ELU, 256, ELU, 128, ELU] -> 7 (NO LayerNorm)
Action scale: 2.0  (target = default_arm + clamp(action, -1.5, 1.5) * 2.0)
Default arm: [0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0]

Checkpoint format (DualActorCritic):
  model_state_dict:
    arm_actor.net.0.weight  [256, 39]
    arm_actor.net.0.bias    [256]
    arm_actor.net.2.weight  [256, 256]
    arm_actor.net.2.bias    [256]
    arm_actor.net.4.weight  [128, 256]
    arm_actor.net.4.bias    [128]
    arm_actor.net.6.weight  [7, 128]
    arm_actor.net.6.bias    [7]
    arm_actor.log_std       [7]
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from typing import Optional


# Stage 2 arm obs/act dimensions
ARM_OBS_DIM = 39
ARM_ACT_DIM = 7

# Action scale and clamp (must match training)
ARM_ACTION_SCALE = 2.0
ARM_ACTION_CLAMP = 1.5  # raw action clamped before scaling

# Default arm pose (Stage 2, 29-DoF robot)
ARM_DEFAULT = [0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0]

# Palm forward offset for EE computation (matches Stage 2 training: 0.02)
PALM_FORWARD_OFFSET = 0.02

# Shoulder offset in body frame (right shoulder)
SHOULDER_OFFSET = [0.0, -0.174, 0.259]

# Reach steps
MAX_REACH_STEPS = 150

# The 7 arm joints controlled by Stage 2 policy, in 29-DoF names
ARM_POLICY_JOINT_NAMES_29DOF = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Finger joint names for right hand (29-DoF DEX3)
# Not used by Stage 2 obs, but still needed for finger controller
RIGHT_FINGER_JOINT_NAMES_29DOF = [
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]


def get_palm_forward(quat: torch.Tensor) -> torch.Tensor:
    """Extract forward direction (first column of rotation matrix) from wxyz quaternion."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y)
    ], dim=-1)


def compute_orientation_error(palm_quat: torch.Tensor, target_dir: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute angular error between palm forward and target direction."""
    forward = get_palm_forward(palm_quat)
    if target_dir is None:
        target_dir = torch.zeros_like(forward)
        target_dir[:, 2] = -1.0  # palm down
    dot = torch.clamp((forward * target_dir).sum(dim=-1), -1.0, 1.0)
    return torch.acos(dot)


class ArmPolicyWrapper:
    """
    Wrapper for the Stage 2 arm actor neural network.

    Loads weights from the DualActorCritic checkpoint and provides:
      - get_action(obs_39) -> action_7
      - build_obs(...) -> obs_39  (given robot state + target)

    Only controls RIGHT arm (7 joints). Left arm stays at default.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        # Build arm actor: 39 -> [256, ELU, 256, ELU, 128, ELU] -> 7
        layers = []
        prev = ARM_OBS_DIM
        for h in [256, 256, 128]:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, ARM_ACT_DIM))
        self.net = nn.Sequential(*layers).to(self.device)
        self.net.eval()

        # Load weights from checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))

        # Extract arm_actor weights
        arm_state = {}
        for key, val in state.items():
            if key.startswith("arm_actor.net."):
                new_key = key.replace("arm_actor.net.", "")
                arm_state[new_key] = val

        if not arm_state:
            raise RuntimeError(
                f"No arm_actor.net.* keys found in checkpoint! "
                f"Available keys: {[k for k in state.keys() if 'arm' in k]}"
            )

        self.net.load_state_dict(arm_state)

        # Default arm pose
        self._default_arm = torch.tensor(
            ARM_DEFAULT, dtype=torch.float32, device=self.device
        )

        # Internal state
        self._prev_targets = None
        self._prev_action = None  # raw network output for prev_arm_act obs

        # Print info
        ckpt_name = os.path.basename(checkpoint_path)
        curriculum = ckpt.get("curriculum_level", "?")
        best_reward = ckpt.get("best_reward", "?")
        iteration = ckpt.get("iteration", ckpt.get("iter", "?"))
        print(f"[ArmPolicy Stage2] Architecture: {ARM_OBS_DIM} -> [256,256,128](ELU) -> {ARM_ACT_DIM}")
        print(f"[ArmPolicy Stage2] Checkpoint: {ckpt_name}")
        print(f"[ArmPolicy Stage2] iter={iteration}, reward={best_reward}, curriculum={curriculum}")
        print(f"[ArmPolicy Stage2] Action scale: {ARM_ACTION_SCALE}, clamp: +/-{ARM_ACTION_CLAMP}")
        print(f"[ArmPolicy Stage2] Default arm: {ARM_DEFAULT}")

    @property
    def prev_action(self) -> Optional[torch.Tensor]:
        """Previous raw action output (for prev_arm_act observation)."""
        return self._prev_action

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Run arm actor inference.

        Args:
            obs: [N, 39] arm observation vector

        Returns:
            action: [N, 7] raw arm actions (before clamping/scaling)
        """
        return self.net(obs)

    def get_arm_targets(self, obs: torch.Tensor, smooth_alpha: float = 0.2) -> torch.Tensor:
        """
        Get absolute joint targets from arm policy.

        Args:
            obs: [N, 39] arm observation
            smooth_alpha: exponential smoothing factor (0=no smoothing, 1=full smoothing)
                Stage 2 uses 0.2 (light smoothing) since action_scale=2.0 already
                produces large targets; heavy smoothing delays convergence.

        Returns:
            targets: [N, 7] absolute joint positions for the 7 policy-controlled joints.
        """
        raw_action = self.get_action(obs)
        # Store raw action for prev_arm_act observation (before clamping)
        self._prev_action = raw_action.clone()
        # Clamp raw action (matches training)
        clamped_action = raw_action.clamp(-ARM_ACTION_CLAMP, ARM_ACTION_CLAMP)
        # target = default + clamped_action * scale
        targets = self._default_arm.unsqueeze(0) + clamped_action * ARM_ACTION_SCALE
        # Light EMA smoothing to reduce jitter
        if smooth_alpha > 0 and self._prev_targets is not None:
            targets = (1.0 - smooth_alpha) * targets + smooth_alpha * self._prev_targets
        self._prev_targets = targets.clone()
        return targets

    def reset_state(self, current_targets: "torch.Tensor | None" = None):
        """Reset internal state (call when activating the arm policy).

        Args:
            current_targets: [N, 7] current right arm joint positions.
                If provided, used as initial smoothing reference.
        """
        if current_targets is not None:
            self._prev_targets = current_targets.clone()
            n = current_targets.shape[0]
            self._prev_action = torch.zeros(n, ARM_ACT_DIM, device=self.device)
        else:
            self._prev_targets = None
            self._prev_action = None

    @staticmethod
    def build_obs(
        arm_pos: torch.Tensor,           # [N, 7] right arm joint positions
        arm_vel: torch.Tensor,           # [N, 7] right arm joint velocities
        ee_body: torch.Tensor,           # [N, 3] EE position in body frame
        palm_quat: torch.Tensor,         # [N, 4] palm quaternion (wxyz)
        target_body: torch.Tensor,       # [N, 3] target position in body frame
        prev_arm_act: torch.Tensor,      # [N, 7] previous raw arm action
        steps_since_spawn: torch.Tensor, # [N] steps since arm was activated
        target_orient: Optional[torch.Tensor] = None,  # [N, 3] target orient dir
    ) -> torch.Tensor:
        """
        Build 39-dim observation matching Stage 2 training format exactly.

        Order: arm_pos(7) + arm_vel*0.1(7) + ee_body(3) + palm_quat(4)
               + target_body(3) + target_orient(3) + pos_error(3)
               + orient_err_norm(1) + prev_arm_act(7) + steps_norm(1) = 39

        Returns:
            obs: [N, 39] clamped to [-10, 10]
        """
        n = arm_pos.shape[0]
        device = arm_pos.device

        if target_orient is None:
            target_orient = torch.zeros(n, 3, device=device)
            target_orient[:, 2] = -1.0  # palm down

        # Position error
        pos_error = target_body - ee_body

        # Orientation error normalized by pi
        orient_err = compute_orientation_error(palm_quat, target_orient).unsqueeze(-1) / 3.14159

        # Steps normalized
        steps_norm = (steps_since_spawn.float() / MAX_REACH_STEPS).unsqueeze(-1).clamp(0, 2)

        # Assemble 39-dim observation (exact Stage 2 order)
        obs = torch.cat([
            arm_pos,               # 7   [0:7]
            arm_vel * 0.1,         # 7   [7:14]
            ee_body,               # 3   [14:17]
            palm_quat,             # 4   [17:21]
            target_body,           # 3   [21:24]
            target_orient,         # 3   [24:27]
            pos_error,             # 3   [27:30]
            orient_err,            # 1   [30]
            prev_arm_act,          # 7   [31:38]
            steps_norm,            # 1   [38]
        ], dim=-1)  # = 39

        return obs.clamp(-10, 10).nan_to_num()
