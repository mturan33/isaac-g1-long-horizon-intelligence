"""
Walk-To Skill
==============
Navigate the robot to a target (x, y) position using an adaptive PID
controller that compensates for locomotion policy inconsistencies.

Architecture:
    target (x, y) -> AdaptivePIDWalkController -> [vx, vy, vyaw] -> LocoPolicy -> joint targets

The PID controller:
    - Turns in place when heading error > 40 degrees
    - Uses integral term to detect and overcome stalls
    - Scales forward velocity by heading alignment
    - Decelerates smoothly on final approach

Termination:
    - Success: distance to target < threshold
    - Timeout: exceeded max_steps
    - Failure: robot fell (base_height < min_height)
"""

from __future__ import annotations

import torch
from typing import Optional

from .base_skill import BaseSkill, SkillResult, SkillStatus
from ..low_level.velocity_command import AdaptivePIDWalkController, get_yaw_from_quat
from ..config.skill_config import WalkToConfig
from ..config.joint_config import MIN_BASE_HEIGHT


class WalkToSkill(BaseSkill):
    """Walk to a target XY position using adaptive PID control."""

    def __init__(
        self,
        config: Optional[WalkToConfig] = None,
        device: str = "cuda",
        num_envs: int = 1,
    ):
        super().__init__(name="walk_to", device=device)
        self.cfg = config or WalkToConfig()
        self._max_steps = self.cfg.max_steps
        self._num_envs = num_envs

        # Adaptive PID controller
        self._pid: Optional[AdaptivePIDWalkController] = None

        # Target
        self._target_pos: Optional[torch.Tensor] = None

    def _ensure_pid(self, num_envs: int):
        """Create/recreate PID controller if needed."""
        if self._pid is None or self._pid.num_envs != num_envs:
            self._pid = AdaptivePIDWalkController(
                max_lin_vel_x=self.cfg.max_forward_vel,
                max_lin_vel_y=self.cfg.max_lateral_vel,
                max_ang_vel_z=self.cfg.max_yaw_rate,
                num_envs=num_envs,
                device=str(self.device),
            )
        else:
            self._pid.reset()

    def reset(
        self,
        target_x: float = None,
        target_y: float = None,
        target_positions: torch.Tensor = None,
        **kwargs,
    ) -> None:
        """
        Initialize walk_to skill.

        Args:
            target_x: Target X position for all envs (world frame, meters)
            target_y: Target Y position for all envs (world frame, meters)
            target_positions: Per-env targets [num_envs, 2] (overrides target_x/y)
        """
        super().reset()
        if target_positions is not None:
            self._target_pos = target_positions.to(dtype=torch.float32, device=self.device)
            num_envs = self._target_pos.shape[0]
            print(f"[WalkTo] Per-env targets: {num_envs} envs")
        elif target_x is not None and target_y is not None:
            self._target_pos = torch.tensor(
                [[target_x, target_y]], dtype=torch.float32, device=self.device
            )
            num_envs = 1
            print(f"[WalkTo] Target: ({target_x:.2f}, {target_y:.2f})")
        else:
            raise ValueError("Must provide target_positions or (target_x, target_y)")

        self._ensure_pid(num_envs)

    def step(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, bool, SkillResult]:
        """
        Execute one walk_to step.

        Returns velocity command for the locomotion policy.
        """
        super().step(obs_dict)

        # Check timeout
        timeout = self._check_timeout()
        if timeout is not None:
            num_envs = obs_dict["root_pos"].shape[0]
            zero_cmd = torch.zeros(num_envs, 3, device=self.device)
            return zero_cmd, True, timeout

        # Extract robot state
        root_pos = obs_dict["root_pos"]       # [num_envs, 3]
        root_quat = obs_dict["root_quat"]     # [num_envs, 4]
        base_height = obs_dict.get("base_height", root_pos[:, 2])

        robot_pos_xy = root_pos[:, :2]        # [num_envs, 2]
        robot_yaw = get_yaw_from_quat(root_quat)  # [num_envs]

        # Check if robot fell — fail only if majority fell (tolerates 1 env falling)
        num_envs = robot_pos_xy.shape[0]
        fallen = (base_height < MIN_BASE_HEIGHT).sum().item()
        if fallen > num_envs // 2:
            zero_cmd = torch.zeros(num_envs, 3, device=self.device)
            return zero_cmd, True, self._make_failure(
                reason="Robot fell",
                base_height=base_height.min().item(),
            )

        # Expand target for num_envs (handles both single and per-env targets)
        target = self._target_pos
        if target.shape[0] == 1 and robot_pos_xy.shape[0] > 1:
            target = target.expand(robot_pos_xy.shape[0], -1)

        # Ensure PID controller exists with correct num_envs
        if self._pid is None:
            self._ensure_pid(robot_pos_xy.shape[0])

        # Compute velocity command with adaptive PID
        cmd_vel, distance = self._pid.compute(
            robot_pos_xy, robot_yaw, target
        )

        # Filter out fallen robots from distance calculation
        # Fallen robots have random/drifted positions that corrupt the mean
        standing_mask = base_height.squeeze() > MIN_BASE_HEIGHT
        if standing_mask.any():
            effective_dist = distance[standing_mask].mean().item()
        else:
            effective_dist = distance.mean().item()

        # Check arrival -- use stop_distance if set, otherwise position_threshold
        arrival_dist = self.cfg.stop_distance if self.cfg.stop_distance > 0 else self.cfg.position_threshold
        if effective_dist < arrival_dist:
            # Stop the robot
            zero_cmd = torch.zeros_like(cmd_vel)
            return zero_cmd, True, self._make_success(
                reason="Reached target",
                final_distance=effective_dist,
            )

        # Log progress periodically
        if self._step_count % 100 == 0:
            stall = f", boost={self._pid._stall_boost.mean():.2f}" if self._pid._stall_boost.mean() > 0.01 else ""
            n_standing = standing_mask.sum().item()
            print(
                f"[WalkTo] Step {self._step_count}: "
                f"dist={effective_dist:.2f}m ({n_standing}/{num_envs} standing), "
                f"cmd=[{cmd_vel[0,0]:.2f}, {cmd_vel[0,1]:.2f}, {cmd_vel[0,2]:.2f}]"
                f"{stall}"
            )

        return cmd_vel, False, self._make_running(
            distance=effective_dist,
        )

    def get_affordance(self, state: dict) -> float:
        """
        Estimate success probability based on current state.

        Higher when:
          - Robot is standing (not squatting)
          - Target is within reasonable distance (< 10m)
          - Robot is not holding an object (walk while holding is harder)
        """
        affordance = 1.0

        # Check robot stance
        robot = state.get("robot", {})
        if robot.get("stance") == "squatting":
            affordance *= 0.3  # Walking while squatting is hard

        # Check distance (if target known)
        if self._target_pos is not None and "position" in robot:
            robot_pos = robot["position"]
            dx = self._target_pos[0, 0].item() - robot_pos[0]
            dy = self._target_pos[0, 1].item() - robot_pos[1]
            dist = (dx**2 + dy**2) ** 0.5
            if dist > 10.0:
                affordance *= 0.5  # Long distance
            elif dist < 0.3:
                affordance *= 0.95  # Already close

        # Holding object penalty (walking is less stable)
        if robot.get("holding") is not None:
            affordance *= 0.7

        return affordance
