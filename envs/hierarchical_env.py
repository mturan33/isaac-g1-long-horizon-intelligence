"""
Hierarchical G1 Environment (V6.2 Loco + DEX3)
=================================================
Isaac Lab environment for hierarchical control of the G1 humanoid
with DEX3 3-finger hands (43 DoF = 15 loco + 14 arm + 14 finger).

Key change from previous version:
  - Uses V6.2 unified locomotion policy (66 obs → 15 act)
  - Loco policy controls ONLY legs + waist (15 joints)
  - Arms controlled separately by ArmController (14 joints)
  - Fingers controlled by FingerController (14 joints)
  - No more "overriding" — loco never touches arms!

Scene (matching PickPlace-Locomanipulation-G1 env):
  - G1 robot with DEX3 hands on flat terrain
  - PackingTable (kinematic, 3m ahead) with built-in basket
  - Steering wheel on the table (dynamic rigid body)
  - Dome light

Control pipeline (walking mode):
  velocity_command [vx, vy, vyaw]
      → V6.2 LocoPolicy (66 obs → 15 loco actions)
      → legs+waist targets (15) + arm defaults (14) + finger targets (14)
      → Robot PD actuators (43 total)

Control pipeline (manipulation mode):
  velocity_command [vx, vy, vyaw]
      → V6.2 LocoPolicy (66 obs → 15 loco actions)
      → legs+waist targets (15) + arm_controller targets (14) + finger targets (14)
      → Robot PD actuators (43 total)

V6.2 actuator parameters are matched EXACTLY to training config.
"""

from __future__ import annotations

import math
import torch
import numpy as np
from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


# =============================================================================
# V6.2 CONSTANTS — must match training exactly
# =============================================================================

PHYSICS_DT = 1.0 / 200.0   # 200 Hz physics
DECIMATION = 4               # 4 physics steps per control step
CONTROL_DT = PHYSICS_DT * DECIMATION  # 0.02s = 50 Hz

HEIGHT_DEFAULT = 0.80
GAIT_FREQUENCY = 1.5  # Hz

LEG_ACTION_SCALE = 0.4   # radians
WAIST_ACTION_SCALE = 0.2  # radians

# DEX3 USD path — SAME as V6.2 training (from unitree_sim_isaaclab)
DEX3_USD_PATH = "C:/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd"

# Joint names — ordered by control group
LOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]  # 15

ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # 14

HAND_JOINT_NAMES = [
    "left_hand_index_0_joint", "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint", "left_hand_index_1_joint",
    "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_1_joint", "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]  # 14

# Default poses (V6.2 training init_state)
DEFAULT_LOCO_POSES = {
    "left_hip_pitch_joint": -0.20, "right_hip_pitch_joint": -0.20,
    "left_hip_roll_joint": 0.0, "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.42, "right_knee_joint": 0.42,
    "left_ankle_pitch_joint": -0.23, "right_ankle_pitch_joint": -0.23,
    "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
}
DEFAULT_ARM_POSES = {
    "left_shoulder_pitch_joint": 0.35, "left_shoulder_roll_joint": 0.18,
    "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.87,
    "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.35, "right_shoulder_roll_joint": -0.18,
    "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.87,
    "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
DEFAULT_HAND_POSES = {name: 0.0 for name in HAND_JOINT_NAMES}
DEFAULT_ALL_POSES = {**DEFAULT_LOCO_POSES, **DEFAULT_ARM_POSES, **DEFAULT_HAND_POSES}

# Default poses as ordered lists
DEFAULT_LOCO_LIST = [DEFAULT_LOCO_POSES[j] for j in LOCO_JOINT_NAMES]
DEFAULT_ARM_LIST = [DEFAULT_ARM_POSES[j] for j in ARM_JOINT_NAMES]
DEFAULT_HAND_LIST = [DEFAULT_HAND_POSES[j] for j in HAND_JOINT_NAMES]


def quat_to_euler_xyz_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternion (Isaac Lab convention) to roll, pitch, yaw.
    Must match V6.2 training exactly."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1)


# =============================================================================
# Scene Configuration — V6.2 actuator params exactly matching training
# =============================================================================

@configclass
class HierarchicalSceneCfg(InteractiveSceneCfg):
    """Scene: flat ground + G1-DEX3 robot + PackingTable + steering wheel + light.
    Matches PickPlace-Locomanipulation-G1 env layout.
    Robot actuator parameters match V6.2 training config EXACTLY."""

    # -- Terrain --
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )

    # -- G1 Robot with DEX3 hands (43 DoF) --
    # Actuator params from ulc_g1_29dof_cfg.py ACTUATOR_PARAMS
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DEX3_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.80),
            joint_pos=DEFAULT_ALL_POSES,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.90,
        actuators={
            # Legs + waist: high stiffness for stability
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint", ".*_hip_roll_joint",
                    ".*_hip_pitch_joint", ".*_knee_joint", ".*waist.*",
                ],
                effort_limit_sim={
                    ".*_hip_yaw_joint": 88.0,
                    ".*_hip_roll_joint": 139.0,
                    ".*_hip_pitch_joint": 88.0,
                    ".*_knee_joint": 139.0,
                    ".*waist_yaw_joint": 88.0,
                    ".*waist_roll_joint": 88.0,
                    ".*waist_pitch_joint": 88.0,
                },
                velocity_limit_sim={
                    ".*_hip_yaw_joint": 32.0,
                    ".*_hip_roll_joint": 20.0,
                    ".*_hip_pitch_joint": 32.0,
                    ".*_knee_joint": 20.0,
                    ".*waist_yaw_joint": 32.0,
                    ".*waist_roll_joint": 30.0,
                    ".*waist_pitch_joint": 30.0,
                },
                stiffness={
                    ".*_hip_yaw_joint": 150.0,
                    ".*_hip_roll_joint": 150.0,
                    ".*_hip_pitch_joint": 200.0,
                    ".*_knee_joint": 200.0,
                    ".*waist.*": 200.0,
                },
                damping={
                    ".*_hip_yaw_joint": 5.0,
                    ".*_hip_roll_joint": 5.0,
                    ".*_hip_pitch_joint": 5.0,
                    ".*_knee_joint": 5.0,
                    ".*waist.*": 10.0,
                },
                armature=0.01,
            ),
            # Feet: lower stiffness for compliance
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit_sim={
                    ".*_ankle_pitch_joint": 35.0,
                    ".*_ankle_roll_joint": 35.0,
                },
                velocity_limit_sim={
                    ".*_ankle_pitch_joint": 30.0,
                    ".*_ankle_roll_joint": 30.0,
                },
                stiffness=20.0,
                damping=2.0,
                armature=0.01,
            ),
            # Shoulders
            "shoulders": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"],
                effort_limit_sim={
                    ".*_shoulder_pitch_joint": 25.0,
                    ".*_shoulder_roll_joint": 25.0,
                },
                velocity_limit_sim={
                    ".*_shoulder_pitch_joint": 37.0,
                    ".*_shoulder_roll_joint": 37.0,
                },
                stiffness=100.0,
                damping=2.0,
                armature=0.01,
            ),
            # Arms (shoulder_yaw + elbow)
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_yaw_joint", ".*_elbow_joint"],
                effort_limit_sim={
                    ".*_shoulder_yaw_joint": 25.0,
                    ".*_elbow_joint": 25.0,
                },
                velocity_limit_sim={
                    ".*_shoulder_yaw_joint": 37.0,
                    ".*_elbow_joint": 37.0,
                },
                stiffness=50.0,
                damping=2.0,
                armature=0.01,
            ),
            # Wrists
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=[".*_wrist_.*"],
                effort_limit_sim={
                    ".*_wrist_yaw_joint": 5.0,
                    ".*_wrist_roll_joint": 25.0,
                    ".*_wrist_pitch_joint": 5.0,
                },
                velocity_limit_sim={
                    ".*_wrist_yaw_joint": 22.0,
                    ".*_wrist_roll_joint": 37.0,
                    ".*_wrist_pitch_joint": 22.0,
                },
                stiffness=40.0,
                damping=2.0,
                armature=0.01,
            ),
            # DEX3 Hands
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hand_index_.*_joint",
                    ".*_hand_middle_.*_joint",
                    ".*_hand_thumb_.*_joint",
                ],
                effort_limit=300,
                velocity_limit=100.0,
                stiffness={".*": 100.0},
                damping={".*": 10.0},
                armature={".*": 0.1},
            ),
        },
    )

    # -- PackingTable: 3m ahead, rotated 90deg CW around Z --
    # PackingTable USD: surface ~z=0.70 when spawned at z=-0.3, has built-in basket
    # 90deg CW (-90deg) rotation: quat wxyz = (cos(-45), 0, 0, sin(-45)) = (0.7071, 0, 0, -0.7071)
    # Basket faces near side (toward robot), open end faces away
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="C:/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/unitree_sim_isaaclab/assets/objects/PackingTable/PackingTable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.0, 0.0, -0.3),
            rot=(0.7071, 0.0, 0.0, -0.7071),
        ),
    )

    # -- Steering wheel on table (near front edge, right-arm reachable) --
    # Table rotated 90deg CW: narrow end faces robot, long axis along Y
    # Place near the front edge (x=2.75), slightly right of center (y=-0.10)
    # Table surface extends roughly x=[2.6, 3.4], y=[-0.85, 0.85]
    pickup_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=10.0,  # High damping to resist being pushed
                max_linear_velocity=0.5,  # Limit velocity
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.75, -0.10, 0.72),  # On table surface, near front edge
        ),
    )

    # -- Dome light --
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# =============================================================================
# Environment Wrapper
# =============================================================================

class HierarchicalG1Env:
    """
    V6.2 loco + DEX3 hierarchical control environment.

    Loco policy controls legs + waist (15 joints).
    Arms controlled by ArmController (14 joints).
    Fingers controlled by FingerController (14 joints).
    Total: 43 joints.

    Walking mode:
        env.step(vel_cmd)
        → loco policy → legs+waist targets
        → arms held at default
        → fingers from controller

    Manipulation mode:
        env.step_manipulation(vel_cmd, arm_targets)
        → loco policy → legs+waist targets
        → arms from arm_targets
        → fingers from controller
    """

    def __init__(
        self,
        sim: sim_utils.SimulationContext,
        scene_cfg: HierarchicalSceneCfg,
        checkpoint_path: str,
        num_envs: int = 16,
        device: str = "cuda:0",
        arm_checkpoint_path: Optional[str] = None,
    ):
        self.sim = sim
        self.device = device
        self.num_envs = num_envs
        self.decimation = DECIMATION
        self.physics_dt = PHYSICS_DT
        self.control_dt = CONTROL_DT
        self.step_count = 0

        # Operating mode
        self._manipulation_mode = False

        # -- Create scene --
        scene_cfg.num_envs = num_envs
        scene_cfg.env_spacing = 8.0
        self.scene = InteractiveScene(scene_cfg)

        # -- Get entity handles --
        self.robot: Articulation = self.scene["robot"]
        self.table: RigidObject = self.scene["table"]
        self.pickup_obj: RigidObject = self.scene["pickup_object"]

        # -- Load V6.2 locomotion policy --
        from ..low_level.policy_wrapper import LocomotionPolicy
        self.loco_policy = LocomotionPolicy(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        # -- Create finger controller --
        from ..low_level.finger_controller import FingerController
        self.finger_controller = FingerController(
            num_envs=num_envs,
            device=device,
        )

        # -- Create arm controller (heuristic fallback) --
        from ..low_level.arm_controller import ArmController
        self.arm_controller = ArmController(
            num_envs=num_envs,
            device=device,
        )

        # -- Optionally load Stage 2 arm policy --
        self.arm_policy = None
        self._arm_policy_enabled = False
        if arm_checkpoint_path is not None:
            from ..low_level.arm_policy_wrapper import ArmPolicyWrapper
            self.arm_policy = ArmPolicyWrapper(
                checkpoint_path=arm_checkpoint_path,
                device=device,
            )

        # Joint indices (set after first reset)
        self._loco_idx: Optional[torch.Tensor] = None
        self._arm_idx: Optional[torch.Tensor] = None
        self._hand_idx: Optional[torch.Tensor] = None
        self._indices_resolved = False

        # V6.2 state tracking
        self._default_loco = torch.tensor(
            DEFAULT_LOCO_LIST, dtype=torch.float32, device=device
        )
        self._default_arm = torch.tensor(
            DEFAULT_ARM_LIST, dtype=torch.float32, device=device
        )
        self._default_hand = torch.tensor(
            DEFAULT_HAND_LIST, dtype=torch.float32, device=device
        )

        leg_scales = [LEG_ACTION_SCALE] * 12
        waist_scales = [WAIST_ACTION_SCALE] * 3
        self._action_scales = torch.tensor(
            leg_scales + waist_scales, dtype=torch.float32, device=device
        )

        # V6.2 internal state
        self._prev_act = torch.zeros(num_envs, 15, device=device)
        self._phase = torch.zeros(num_envs, device=device)
        self._height_cmd = torch.ones(num_envs, device=device) * HEIGHT_DEFAULT
        self._torso_cmd = torch.zeros(num_envs, 3, device=device)

        # Gravity vector (constant)
        self._gravity_vec = torch.tensor(
            [0.0, 0.0, -1.0], dtype=torch.float32, device=device
        )

        # Initial positions (set in reset)
        self._initial_pos: Optional[torch.Tensor] = None
        self._is_reset = False

        # -- Stage 2 arm policy state (only used when arm_policy is loaded) --
        # These are resolved in _resolve_joint_indices after first reset
        self._arm_policy_joint_idx: Optional[torch.Tensor] = None  # 7 joints in robot
        self._right_finger_idx: Optional[torch.Tensor] = None      # 7 right finger joints
        self._palm_body_idx: Optional[int] = None                   # palm body index
        # Arm policy state buffers
        self._arm_steps_since_spawn = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._arm_target_world = torch.zeros(num_envs, 3, device=device)  # stored in WORLD frame
        self._arm_target_body = torch.zeros(num_envs, 3, device=device)   # recomputed each step
        self._arm_target_orient = torch.zeros(num_envs, 3, device=device)
        self._arm_target_orient[:, 2] = -1.0  # palm down default
        # Finger limits (resolved after first reset)
        self._right_finger_lower: Optional[torch.Tensor] = None
        self._right_finger_upper: Optional[torch.Tensor] = None

        # -- Magnetic grasp: snap object to palm when close enough --
        self._object_attached = False
        self._attach_offset_body = torch.zeros(num_envs, 3, device=device)  # offset in palm frame

        # -- Debug visualization markers --
        self._debug_markers_enabled = False
        self._ee_marker: Optional[VisualizationMarkers] = None
        self._target_marker: Optional[VisualizationMarkers] = None

        print(f"[HierarchicalG1Env V6.2] {num_envs} envs, device={device}")
        print(f"[HierarchicalG1Env V6.2] Control: {1.0/self.control_dt:.0f} Hz "
              f"({self.decimation}x decimation)")
        arm_mode = "Stage2 policy" if self.arm_policy is not None else "heuristic"
        print(f"[HierarchicalG1Env V6.2] Joints: 15 loco + 14 arm + 14 finger = 43")
        print(f"[HierarchicalG1Env V6.2] Arm control: {arm_mode}")

    # --------------------------------------------------------------------- #
    # Joint index resolution
    # --------------------------------------------------------------------- #
    def _resolve_joint_indices(self):
        """Map LOCO/ARM/HAND joint names to robot's articulation indices."""
        joint_names = self.robot.joint_names
        total = len(joint_names)

        # Find loco joint indices (must all exist)
        loco_idx = []
        for name in LOCO_JOINT_NAMES:
            if name not in joint_names:
                raise RuntimeError(f"Loco joint '{name}' not found in robot! "
                                   f"Available: {joint_names}")
            loco_idx.append(joint_names.index(name))
        self._loco_idx = torch.tensor(loco_idx, device=self.device, dtype=torch.long)

        # Find arm joint indices
        arm_idx = []
        for name in ARM_JOINT_NAMES:
            if name not in joint_names:
                raise RuntimeError(f"Arm joint '{name}' not found in robot!")
            arm_idx.append(joint_names.index(name))
        self._arm_idx = torch.tensor(arm_idx, device=self.device, dtype=torch.long)

        # Find hand joint indices
        hand_idx = []
        for name in HAND_JOINT_NAMES:
            if name not in joint_names:
                raise RuntimeError(f"Hand joint '{name}' not found in robot!")
            hand_idx.append(joint_names.index(name))
        self._hand_idx = torch.tensor(hand_idx, device=self.device, dtype=torch.long)

        self._indices_resolved = True

        print(f"[HierarchicalG1Env V6.2] Joint mapping resolved ({total} total):")
        print(f"  Loco (15): {[joint_names[i] for i in loco_idx[:4]]}...")
        print(f"  Arm  (14): {[joint_names[i] for i in arm_idx[:4]]}...")
        print(f"  Hand (14): {[joint_names[i] for i in hand_idx[:4]]}...")

        # -- Arm policy indices (Stage 2: right arm 7 joints + right fingers 7) --
        if self.arm_policy is not None:
            from ..low_level.arm_policy_wrapper import (
                ARM_POLICY_JOINT_NAMES_29DOF,
                RIGHT_FINGER_JOINT_NAMES_29DOF,
            )
            # 7 arm policy joints
            ap_idx = []
            for name in ARM_POLICY_JOINT_NAMES_29DOF:
                if name not in joint_names:
                    raise RuntimeError(f"Arm policy joint '{name}' not found!")
                ap_idx.append(joint_names.index(name))
            self._arm_policy_joint_idx = torch.tensor(
                ap_idx, device=self.device, dtype=torch.long
            )

            # 7 right finger joints
            rf_idx = []
            for name in RIGHT_FINGER_JOINT_NAMES_29DOF:
                if name not in joint_names:
                    raise RuntimeError(f"Right finger joint '{name}' not found!")
                rf_idx.append(joint_names.index(name))
            self._right_finger_idx = torch.tensor(
                rf_idx, device=self.device, dtype=torch.long
            )

            # Finger joint limits
            joint_limits = self.robot.root_physx_view.get_dof_limits()
            self._right_finger_lower = torch.tensor(
                [joint_limits[0, i, 0].item() for i in rf_idx],
                device=self.device
            )
            self._right_finger_upper = torch.tensor(
                [joint_limits[0, i, 1].item() for i in rf_idx],
                device=self.device
            )

            # Palm/EE body index
            # Stage 2 controls all 7 right arm joints including wrist_pitch/yaw.
            # Search for palm link in DEX3 hand.
            # Search priority: right_palm > right_wrist_pitch > right_wrist_yaw > fallback
            body_names = self.robot.body_names
            self._palm_body_idx = None
            search_order = [
                "right_hand_palm",      # DEX3 actual palm link
                "right_palm",           # 23-DoF exact name
                "right_wrist_pitch",    # 29-DoF equivalent (child of wrist_roll)
                "right_wrist_yaw",      # fallback (end of wrist chain)
                "right_hand_base",      # alternate naming
            ]
            for search_term in search_order:
                for i, name in enumerate(body_names):
                    if search_term in name.lower():
                        self._palm_body_idx = i
                        break
                if self._palm_body_idx is not None:
                    break
            if self._palm_body_idx is None:
                # Fallback: use last body (often the right hand tip)
                self._palm_body_idx = len(body_names) - 1
                print(f"  [WARN] No palm/wrist body found! Using last body: {body_names[-1]}")
            # Always print all body names for debugging
            print(f"  All bodies ({len(body_names)}): "
                  f"{[n for n in body_names if 'right' in n.lower() and ('wrist' in n.lower() or 'palm' in n.lower() or 'hand' in n.lower())]}")
            if self._palm_body_idx is not None:
                print(f"  EE body: {body_names[self._palm_body_idx]} (idx={self._palm_body_idx})")

            print(f"  ArmPolicy joints ({len(ap_idx)}): {[joint_names[i] for i in ap_idx]}")
            print(f"  Right fingers (7): {[joint_names[i] for i in rf_idx[:3]]}...")

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def set_manipulation_mode(self, enabled: bool):
        """Toggle between walking mode and manipulation mode."""
        self._manipulation_mode = enabled
        mode_name = "MANIPULATION" if enabled else "WALKING"
        print(f"[HierarchicalG1Env V6.2] Mode: {mode_name}")

    def reset(self) -> dict:
        """Reset the environment and return initial observations."""
        if not self._is_reset:
            self.sim.reset()
            self._is_reset = True

        # Reset all scene entities
        indices = torch.arange(self.num_envs, device=self.device)
        self.robot.reset(indices)
        self.table.reset(indices)
        self.pickup_obj.reset(indices)
        self._object_attached = False  # release object on reset

        # Write resets to sim and step once
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.physics_dt)

        # Resolve joint indices (first time only)
        if not self._indices_resolved:
            self._resolve_joint_indices()

        # Reset V6.2 state
        self._prev_act.zero_()
        self._phase[:] = torch.rand(self.num_envs, device=self.device)
        self._height_cmd[:] = HEIGHT_DEFAULT
        self._torso_cmd.zero_()

        # Reset controllers
        self.finger_controller.reset()
        self.arm_controller.reset()
        self.step_count = 0
        self._manipulation_mode = False
        self._arm_policy_enabled = False

        # Reset arm policy state
        self._arm_steps_since_spawn.zero_()
        self._arm_target_world.zero_()
        self._arm_target_body.zero_()
        self._arm_target_orient.zero_()
        self._arm_target_orient[:, 2] = -1.0

        # Store initial XY positions
        self._initial_pos = self.robot.data.root_pos_w[:, :2].clone()

        return self.get_obs()

    def step(self, velocity_command: torch.Tensor) -> dict:
        """
        Step in WALKING mode.

        Loco policy controls legs+waist (15).
        Arms held at default (14).
        Fingers from controller (14).

        Args:
            velocity_command: [N, 3] = [vx, vy, vyaw] in body frame

        Returns:
            obs_dict
        """
        # Build V6.2 obs and get loco actions
        loco_targets = self._run_loco_policy(velocity_command)

        # Arm targets: default pose
        arm_targets = self._default_arm.unsqueeze(0).expand(self.num_envs, -1)

        # Finger targets
        finger_targets = self.finger_controller.get_targets()

        # Apply all targets
        self._apply_targets(loco_targets, arm_targets, finger_targets)

        # Physics sub-stepping
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self._update_attached_object()
            self.sim.step()

        self.scene.update(self.control_dt)
        self.step_count += 1
        return self.get_obs()

    def step_manipulation(
        self,
        velocity_command: torch.Tensor,
        arm_targets: torch.Tensor,
    ) -> dict:
        """
        Step in MANIPULATION mode.

        Loco policy controls legs+waist (15).
        Arm targets provided externally (14).
        Fingers from controller (14).

        Args:
            velocity_command: [N, 3] velocity for leg balance
            arm_targets: [N, 14] absolute arm joint positions

        Returns:
            obs_dict
        """
        # Build V6.2 obs and get loco actions
        loco_targets = self._run_loco_policy(velocity_command)

        # Finger targets
        finger_targets = self.finger_controller.get_targets()

        # Apply all targets
        self._apply_targets(loco_targets, arm_targets, finger_targets)

        # Physics sub-stepping
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self._update_attached_object()
            self.sim.step()

        self.scene.update(self.control_dt)
        self.step_count += 1
        # Update debug markers each step
        self.update_debug_markers()
        return self.get_obs()

    # --------------------------------------------------------------------- #
    # V6.2 observation building and policy inference
    # --------------------------------------------------------------------- #
    def _build_loco_obs(self, velocity_command: torch.Tensor) -> torch.Tensor:
        """
        Build 66-dim observation matching V6.2 training exactly.

        Order:
            lin_vel_b(3) + ang_vel_b(3) + proj_gravity(3)
            + jp_leg(12) + jv_leg*0.1(12) + jp_waist(3) + jv_waist*0.1(3)
            + height_cmd(1) + vel_cmd(3) + gait(2) + prev_act(15)
            + torso_euler(3) + torso_cmd(3)
            = 66
        """
        n = self.num_envs
        r = self.robot
        q = r.data.root_quat_w  # wxyz

        # Body-frame velocities (matching V6.2 computation)
        lin_vel_b = quat_apply_inverse(q, r.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(q, r.data.root_ang_vel_w)
        proj_gravity = quat_apply_inverse(
            q, self._gravity_vec.expand(n, -1)
        )

        # Loco joint positions and velocities (in LOCO_JOINT_NAMES order)
        jp_all = r.data.joint_pos
        jv_all = r.data.joint_vel

        jp_leg = jp_all[:, self._loco_idx[:12]]
        jv_leg = jv_all[:, self._loco_idx[:12]] * 0.1
        jp_waist = jp_all[:, self._loco_idx[12:15]]
        jv_waist = jv_all[:, self._loco_idx[12:15]] * 0.1

        # Gait phase
        gait = torch.stack([
            torch.sin(2 * math.pi * self._phase),
            torch.cos(2 * math.pi * self._phase),
        ], dim=-1)

        # Torso euler from wxyz quaternion
        torso_euler = quat_to_euler_xyz_wxyz(q)

        # Assemble 66-dim observation
        obs = torch.cat([
            lin_vel_b,                         # 3
            ang_vel_b,                         # 3
            proj_gravity,                      # 3
            jp_leg, jv_leg,                    # 24
            jp_waist, jv_waist,                # 6
            self._height_cmd[:, None],         # 1
            velocity_command,                   # 3
            gait,                               # 2
            self._prev_act,                    # 15
            torso_euler,                        # 3
            self._torso_cmd,                   # 3
        ], dim=-1)  # = 66

        return obs.clamp(-10, 10).nan_to_num()

    def _run_loco_policy(self, velocity_command: torch.Tensor) -> torch.Tensor:
        """
        Build obs, run V6.2 loco policy, convert to absolute joint targets.

        Returns:
            loco_targets: [N, 15] absolute joint position targets
                          in LOCO_JOINT_NAMES order
        """
        # Build 66-dim obs
        obs = self._build_loco_obs(velocity_command)

        # Run network
        with torch.inference_mode():
            raw_actions = self.loco_policy.get_raw_action(obs)

        # Convert to absolute targets: default + actions * scales
        targets = (
            self._default_loco.unsqueeze(0)
            + raw_actions * self._action_scales.unsqueeze(0)
        )

        # Waist clamp (V6.2 training match)
        targets[:, 12].clamp_(-0.15, 0.15)   # waist_yaw
        targets[:, 13].clamp_(-0.15, 0.15)   # waist_roll
        targets[:, 14].clamp_(-0.2, 0.2)     # waist_pitch

        # Hip yaw clamp (V6.1+: prevent scissor gait)
        targets[:, 4].clamp_(-0.3, 0.3)      # left_hip_yaw
        targets[:, 5].clamp_(-0.3, 0.3)      # right_hip_yaw

        # Update V6.2 state
        self._prev_act = raw_actions.clone()
        self._phase = (self._phase + GAIT_FREQUENCY * CONTROL_DT) % 1.0

        return targets

    # --------------------------------------------------------------------- #
    # Joint target application
    # --------------------------------------------------------------------- #
    def _apply_targets(
        self,
        loco_targets: torch.Tensor,
        arm_targets: torch.Tensor,
        finger_targets: torch.Tensor,
    ):
        """
        Set joint position targets for all 43 DoF.

        Args:
            loco_targets: [N, 15] legs+waist in LOCO_JOINT_NAMES order
            arm_targets:  [N, 14] arms in ARM_JOINT_NAMES order
            finger_targets: [N, 14] fingers in HAND_JOINT_NAMES order
        """
        # Start from robot defaults (handles any unmapped joints)
        tgt = self.robot.data.default_joint_pos.clone()

        # Place targets at correct global indices
        tgt[:, self._loco_idx] = loco_targets
        tgt[:, self._arm_idx] = arm_targets
        tgt[:, self._hand_idx] = finger_targets

        self.robot.set_joint_position_target(tgt)

    # --------------------------------------------------------------------- #
    # Stage 7 Arm Policy support
    # --------------------------------------------------------------------- #
    def enable_arm_policy(self, enabled: bool = True):
        """Enable/disable Stage 2 arm policy for right arm."""
        if enabled and self.arm_policy is None:
            print("[HierarchicalG1Env] WARNING: No arm policy loaded, staying in heuristic mode")
            return
        self._arm_policy_enabled = enabled
        mode = "Stage2 policy" if enabled else "heuristic"
        print(f"[HierarchicalG1Env] Arm control: {mode}")

    def set_arm_target_world(self, target_world: torch.Tensor):
        """
        Set arm reach target in world coordinates.
        The body-frame target is recomputed EACH STEP in _build_arm_obs()
        since the robot moves and the body frame changes continuously.

        Args:
            target_world: [N, 3] or [3] target position in world frame
        """
        if target_world.ndim == 1:
            target_world = target_world.unsqueeze(0).expand(self.num_envs, -1)
        self._arm_target_world = target_world.clone()
        # Also compute initial body frame for reset_arm_policy_state
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        self._arm_target_body = quat_apply_inverse(root_quat, target_world - root_pos)

    def set_arm_target_body(self, target_body: torch.Tensor):
        """
        Set arm reach target directly in body frame.

        Args:
            target_body: [N, 3] or [3] target in body frame
        """
        if target_body.ndim == 1:
            target_body = target_body.unsqueeze(0).expand(self.num_envs, -1)
        self._arm_target_body = target_body.clone()

    def _compute_palm_ee(self):
        """
        Compute end-effector position from palm body.
        Returns ee_world [N,3] and palm_quat [N,4] (wxyz).
        """
        from ..low_level.arm_policy_wrapper import get_palm_forward, PALM_FORWARD_OFFSET
        palm_pos = self.robot.data.body_pos_w[:, self._palm_body_idx]
        palm_quat = self.robot.data.body_quat_w[:, self._palm_body_idx]
        palm_fwd = get_palm_forward(palm_quat)
        ee_pos = palm_pos + PALM_FORWARD_OFFSET * palm_fwd
        return ee_pos, palm_quat

    # ================================================================== #
    #  Magnetic Grasp: attach/detach object to/from palm
    # ================================================================== #

    def attach_object_to_hand(self, max_dist: float = 0.20) -> bool:
        """Attach the pickup object to the palm if EE is within max_dist.

        Computes offset from palm to object center in palm local frame,
        then each subsequent sim step teleports the object to follow the palm.

        Args:
            max_dist: Max EE-object distance to trigger attach.

        Returns True if attached, False if too far.
        """
        ee_world, palm_quat = self._compute_palm_ee()
        obj_pos = self.pickup_obj.data.root_pos_w
        dist = (ee_world - obj_pos).norm(dim=-1)
        mean_dist = dist.mean().item()

        if mean_dist < max_dist:
            # Compute offset: cup_pos - ee_pos in palm frame (keep actual distance)
            diff_world = obj_pos - ee_world
            self._attach_offset_body = quat_apply_inverse(palm_quat, diff_world)
            self._object_attached = True
            print(f"  [MagneticGrasp] Object attached! dist={mean_dist:.3f}m")
            return True
        else:
            print(f"  [MagneticGrasp] Object too far: {mean_dist:.3f}m (max: {max_dist:.2f}m)")
            return False

    def detach_object(self):
        """Release the attached object (it will fall under gravity)."""
        if self._object_attached:
            self._object_attached = False
            print("  [MagneticGrasp] Object detached")

    # ================================================================== #
    #  Debug Visualization: draw EE and target spheres
    # ================================================================== #

    def enable_debug_markers(self, enabled: bool = True):
        """Enable/disable debug visualization markers for EE and arm target.

        Creates colored spheres:
          - GREEN sphere: End-Effector (palm EE) position
          - RED sphere: Arm target position (where arm is trying to reach)
        """
        self._debug_markers_enabled = enabled
        if enabled and self._ee_marker is None:
            # Create EE marker (green sphere)
            ee_cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/EE_Marker",
                markers={
                    "ee": sim_utils.SphereCfg(
                        radius=0.03,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0),  # Green
                        ),
                    ),
                },
            )
            self._ee_marker = VisualizationMarkers(ee_cfg)

            # Create arm target marker (red sphere)
            target_cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/Target_Marker",
                markers={
                    "target": sim_utils.SphereCfg(
                        radius=0.03,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.0, 0.0),  # Red
                        ),
                    ),
                },
            )
            self._target_marker = VisualizationMarkers(target_cfg)

            # Create object position marker (blue sphere)
            obj_cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/Obj_Marker",
                markers={
                    "obj": sim_utils.SphereCfg(
                        radius=0.025,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 0.5, 1.0),  # Blue
                        ),
                    ),
                },
            )
            self._obj_marker = VisualizationMarkers(obj_cfg)
            print("[Debug] Markers enabled: GREEN=EE, RED=target, BLUE=object")

        if not enabled and self._ee_marker is not None:
            self._ee_marker.set_visibility(False)
            self._target_marker.set_visibility(False)
            self._obj_marker.set_visibility(False)
            print("[Debug] Markers disabled")

    def update_debug_markers(self):
        """Update debug marker positions to current EE and target.
        Call this each step during reach to visualize arm tracking.
        """
        if not self._debug_markers_enabled or self._ee_marker is None:
            return

        ee_world, _ = self._compute_palm_ee()
        target_world = self._arm_target_world
        obj_world = self.pickup_obj.data.root_pos_w

        # Update marker positions (all envs)
        self._ee_marker.visualize(translations=ee_world.detach().cpu().numpy())
        self._target_marker.visualize(translations=target_world.detach().cpu().numpy())
        self._obj_marker.visualize(translations=obj_world.detach().cpu().numpy())

    def _update_attached_object(self):
        """Teleport attached object to follow the palm each sim step.
        Call this AFTER scene.write_data_to_sim() and BEFORE sim.step().
        """
        if not self._object_attached:
            return

        from isaaclab.utils.math import quat_apply

        ee_world, palm_quat = self._compute_palm_ee()
        # Reconstruct cup world position from saved offset
        obj_target = ee_world + quat_apply(palm_quat, self._attach_offset_body)

        # Build root state [N, 13]: pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        root_state = self.pickup_obj.data.default_root_state.clone()
        root_state[:, :3] = obj_target
        root_state[:, 3:7] = palm_quat  # cup follows palm orientation
        root_state[:, 7:] = 0.0  # zero velocity

        self.pickup_obj.write_root_state_to_sim(root_state)

    def _build_arm_obs(self) -> torch.Tensor:
        """
        Build 39-dim observation for Stage 2 arm actor.
        Matches the training format exactly.

        CRITICAL: Target body-frame position is recomputed each step from
        the stored world-frame target, since the robot moves and the body
        frame changes continuously.
        """
        from ..low_level.arm_policy_wrapper import ArmPolicyWrapper, ARM_ACT_DIM

        r = self.robot
        root_pos = r.data.root_pos_w
        root_quat = r.data.root_quat_w

        # --- Recompute target in body frame each step ---
        # This is essential: the robot walks/rotates, so body frame changes
        self._arm_target_body = quat_apply_inverse(
            root_quat, self._arm_target_world - root_pos
        )

        # Right arm joint pos/vel (7 policy joints)
        arm_pos = r.data.joint_pos[:, self._arm_policy_joint_idx]
        arm_vel = r.data.joint_vel[:, self._arm_policy_joint_idx]

        # EE computation
        ee_world, palm_quat = self._compute_palm_ee()
        ee_body = quat_apply_inverse(root_quat, ee_world - root_pos)

        # Previous arm action (for obs)
        prev_arm_act = self.arm_policy.prev_action
        if prev_arm_act is None:
            prev_arm_act = torch.zeros(self.num_envs, ARM_ACT_DIM, device=self.device)

        obs = ArmPolicyWrapper.build_obs(
            arm_pos=arm_pos,
            arm_vel=arm_vel,
            ee_body=ee_body,
            palm_quat=palm_quat,
            target_body=self._arm_target_body,
            prev_arm_act=prev_arm_act,
            steps_since_spawn=self._arm_steps_since_spawn,
            target_orient=self._arm_target_orient,
        )

        # Debug: print obs shape and key values on first step
        if self._arm_steps_since_spawn[0].item() == 0:
            pos_err = self._arm_target_body - ee_body
            print(f"  [ArmObs] shape={obs.shape}, arm_pos[0]={arm_pos[0].tolist()}")
            print(f"  [ArmObs] ee_body[0]={ee_body[0].tolist()}")
            print(f"  [ArmObs] target_body[0]={self._arm_target_body[0].tolist()}")
            print(f"  [ArmObs] pos_error[0]={pos_err[0].tolist()}")

        # Increment arm step counter
        self._arm_steps_since_spawn += 1

        return obs

    def _get_arm_policy_targets(self) -> torch.Tensor:
        """
        Run Stage 2 arm policy and return full 14-dim arm targets.

        The policy outputs 7 joints (all right arm joints).
        Left arm (7 joints) stays at heuristic default.

        Returns:
            arm_targets: [N, 14] in ARM_JOINT_NAMES order
        """
        # Build obs and run policy
        obs = self._build_arm_obs()
        right_7_targets = self.arm_policy.get_arm_targets(obs)  # [N, 7]

        # Start with default arm pose (14 joints: left 7 + right 7)
        arm_targets = self._default_arm.unsqueeze(0).expand(self.num_envs, -1).clone()

        # Map 7 policy outputs to right arm (indices 7-13 in ARM_JOINT_NAMES)
        # ARM_JOINT_NAMES order: L_sp, L_sr, L_sy, L_e, L_wr, L_wp, L_wy,
        #                        R_sp, R_sr, R_sy, R_e, R_wr, R_wp, R_wy
        arm_targets[:, 7:14] = right_7_targets

        return arm_targets

    def reset_arm_policy_state(self):
        """Reset arm policy state buffers (call when activating arm policy)."""
        if self.arm_policy is None:
            return
        # Reset arm policy internal state (smoothing + prev_action)
        # Pass current arm joint positions so first step blends smoothly
        current_arm_7 = self.robot.data.joint_pos[:, self._arm_policy_joint_idx]
        self.arm_policy.reset_state(current_targets=current_arm_7)
        self._arm_steps_since_spawn.zero_()
        print(f"[ArmPolicy] Reset state ({current_arm_7.shape[1]} joints)")

    def step_arm_policy(self, velocity_command: torch.Tensor) -> dict:
        """
        Step in MANIPULATION mode using Stage 2 arm policy for right arm.

        Loco policy controls legs+waist (15).
        Arm policy controls right arm (7 joints).
        Left arm at default (7 joints).
        Fingers from controller (14).

        Args:
            velocity_command: [N, 3] velocity for leg balance

        Returns:
            obs_dict
        """
        # Loco actions
        loco_targets = self._run_loco_policy(velocity_command)

        # Arm targets from Stage 2 policy
        arm_targets = self._get_arm_policy_targets()

        # Finger targets
        finger_targets = self.finger_controller.get_targets()

        # Apply all targets
        self._apply_targets(loco_targets, arm_targets, finger_targets)

        # Physics sub-stepping
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self._update_attached_object()
            self.sim.step()

        self.scene.update(self.control_dt)
        self.step_count += 1
        # Update debug markers each step
        self.update_debug_markers()
        return self.get_obs()

    # --------------------------------------------------------------------- #
    # Observations
    # --------------------------------------------------------------------- #
    def get_obs(self) -> dict:
        """
        Get robot observations for the skill layer.

        Returns dict with:
            root_pos          : [N, 3]  world position
            root_quat         : [N, 4]  wxyz quaternion
            base_ang_vel      : [N, 3]  angular velocity in body frame
            projected_gravity : [N, 3]  gravity in body frame
            base_height       : [N]     base height (z)
            joint_pos_loco    : [N, 15] loco joint positions
            joint_vel_loco    : [N, 15] loco joint velocities
            joint_pos_arm     : [N, 14] arm joint positions
            joint_vel_arm     : [N, 14] arm joint velocities
            joint_pos_finger  : [N, 14] finger joint positions
            joint_vel_finger  : [N, 14] finger joint velocities
            base_lin_vel      : [N, 3]  linear velocity in body frame
        """
        jp = self.robot.data.joint_pos
        jv = self.robot.data.joint_vel
        q = self.robot.data.root_quat_w

        return {
            "root_pos": self.robot.data.root_pos_w,
            "root_quat": q,
            "base_ang_vel": quat_apply_inverse(q, self.robot.data.root_ang_vel_w),
            "projected_gravity": quat_apply_inverse(
                q, self._gravity_vec.expand(self.num_envs, -1)),
            "base_height": self.robot.data.root_pos_w[:, 2],
            "base_lin_vel": quat_apply_inverse(q, self.robot.data.root_lin_vel_w),
            "joint_pos_loco": jp[:, self._loco_idx],
            "joint_vel_loco": jv[:, self._loco_idx],
            "joint_pos_arm": jp[:, self._arm_idx],
            "joint_vel_arm": jv[:, self._arm_idx],
            "joint_pos_finger": jp[:, self._hand_idx],
            "joint_vel_finger": jv[:, self._hand_idx],
            # Backward compatibility aliases
            "joint_pos": jp[:, self._loco_idx],
            "joint_vel": jv[:, self._loco_idx],
            "joint_pos_body": torch.cat([jp[:, self._loco_idx], jp[:, self._arm_idx]], dim=-1),
            "joint_vel_body": torch.cat([jv[:, self._loco_idx], jv[:, self._arm_idx]], dim=-1),
        }

    @property
    def initial_positions(self) -> torch.Tensor:
        """Initial XY positions [num_envs, 2]."""
        if self._initial_pos is None:
            raise RuntimeError("Call reset() first")
        return self._initial_pos

    def close(self):
        """Clean up resources."""
        pass
