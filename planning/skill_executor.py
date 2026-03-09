"""
Skill Executor
================
Executes a plan (list of skill steps) sequentially on the HierarchicalG1Env.

Skills:
    - walk_to: WalkToSkill with stop_distance, then 50-step stabilize
    - pre_reach: raise arm above table to avoid collision (intermediate target)
    - reach: manipulation mode -> Stage 2 arm policy -> magnetic attach
    - grasp: finger_controller.close("both"), hold
    - lift: raise arm above basket height using arm policy (intermediate target)
    - lateral_walk: sidestep while holding arm position (slow, stable)
    - lower: lower arm into basket using arm policy
    - place: detach object, open fingers, arm back to default
"""

from __future__ import annotations

import math
import time
import torch
from typing import Any, Optional

from .semantic_map import SemanticMap
from ..low_level.velocity_command import get_yaw_from_quat, normalize_angle


class SkillExecutor:
    """Execute a skill plan on the HierarchicalG1Env.

    Args:
        env: HierarchicalG1Env instance
        semantic_map: SemanticMap instance for real-time position queries
        simulation_app: IsaacSim app handle (for is_running() check)
    """

    # Right shoulder offset in body frame (from arm_policy_wrapper.py)
    SHOULDER_OFFSET = [0.0, -0.174, 0.259]
    MAX_REACH = 0.55  # Stage 2 arm policy has 55cm reach capability

    # Arm joint indices within the 14-joint arm group
    # ARM_JOINT_NAMES order: L_sh_pitch(0), L_sh_roll(1), L_sh_yaw(2), L_elbow(3),
    #   L_wr_roll(4), L_wr_pitch(5), L_wr_yaw(6),
    #   R_sh_pitch(7), R_sh_roll(8), R_sh_yaw(9), R_elbow(10),
    #   R_wr_roll(11), R_wr_pitch(12), R_wr_yaw(13)
    R_SHOULDER_PITCH = 7
    R_SHOULDER_ROLL = 8
    R_SHOULDER_YAW = 9
    R_ELBOW = 10

    def __init__(
        self,
        env: Any,
        semantic_map: SemanticMap,
        simulation_app: Any = None,
    ):
        self.env = env
        self.semantic_map = semantic_map
        self.sim_app = simulation_app
        self.device = env.device

        self._stand_cmd = torch.zeros(env.num_envs, 3, device=self.device)
        self._hold_arm_targets: Optional[torch.Tensor] = None
        # Persistent PID hold position (shared across skills)
        self._hold_pos_xy: Optional[torch.Tensor] = None
        self._hold_yaw: Optional[torch.Tensor] = None

        # Skill dispatch table
        self._skills = {
            "walk_to": self._execute_walk_to,
            "pre_reach": self._execute_pre_reach,
            "reach": self._execute_reach,
            "grasp": self._execute_grasp,
            "lift": self._execute_lift,
            "lateral_walk": self._execute_lateral_walk,
            "lower": self._execute_lower,
            "place": self._execute_place,
            "walk_to_position": self._execute_walk_to_position,
        }

    def _is_running(self) -> bool:
        """Check if simulation is still running."""
        if self.sim_app is not None:
            return self.sim_app.is_running()
        return True

    def _compute_hold_cmd(
        self,
        hold_pos_xy: torch.Tensor,
        hold_yaw: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Compute position-correction velocity to keep robot at hold position.

        Used during reach/lift/lower to counteract arm policy reaction forces
        that would otherwise push the robot backward.

        Args:
            hold_pos_xy: [num_envs, 2] target XY position to maintain
            hold_yaw: [num_envs] target heading to maintain

        Returns:
            hold_cmd: [num_envs, 3] velocity command [vx, vy, vyaw]
            drift: scalar, mean distance from hold position
        """
        env = self.env
        cur_root_pos = env.robot.data.root_pos_w
        cur_root_quat = env.robot.data.root_quat_w
        cur_pos_xy = cur_root_pos[:, :2]
        cur_yaw = get_yaw_from_quat(cur_root_quat)

        # Position error in world frame (toward hold position)
        delta_w = hold_pos_xy - cur_pos_xy
        drift = delta_w.norm(dim=-1).mean().item()

        # Convert to body frame
        cos_y = torch.cos(cur_yaw)
        sin_y = torch.sin(cur_yaw)
        dx_body = cos_y * delta_w[:, 0] + sin_y * delta_w[:, 1]
        dy_body = -sin_y * delta_w[:, 0] + cos_y * delta_w[:, 1]

        # P-controller: push robot back toward hold position
        vx = (dx_body * 1.5).clamp(-0.3, 0.3)
        vy = (dy_body * 1.0).clamp(-0.15, 0.15)

        # Heading hold — stronger yaw correction to fight arm torque
        heading_err = normalize_angle(hold_yaw - cur_yaw)
        vyaw = (heading_err * 2.5).clamp(-0.6, 0.6)

        hold_cmd = torch.stack([vx, vy, vyaw], dim=-1)
        return hold_cmd, drift

    def execute_plan(self, plan: list) -> dict:
        """Execute a sequence of skill steps.

        Args:
            plan: List of {"skill": str, "params": dict}

        Returns:
            {"plan_results": [...], "completed": bool}
        """
        results = []
        print(f"\n{'='*60}")
        print(f"  EXECUTING PLAN ({len(plan)} steps)")
        print(f"{'='*60}")

        for i, step in enumerate(plan):
            if not self._is_running():
                break

            skill_name = step["skill"]
            params = step.get("params", {})

            print(f"\n{'-'*50}")
            print(f"  Step {i+1}/{len(plan)}: {skill_name}({params})")
            print(f"{'-'*50}")

            # Update semantic map for latest positions
            self.semantic_map.update()

            # Dispatch skill
            handler = self._skills.get(skill_name)
            if handler is None:
                result = {"status": "failed", "reason": f"Unknown skill: {skill_name}"}
            else:
                result = handler(**params)

            results.append({"skill": skill_name, "params": params, "result": result})
            print(f"  -> {skill_name}: {result['status']} ({result.get('reason', '')})")

            if result["status"] == "failed":
                print(f"\n  [Executor] PLAN FAILED at step {i+1}")
                break

        completed = all(r["result"]["status"] == "success" for r in results)
        print(f"\n{'='*60}")
        print(f"  PLAN {'COMPLETED' if completed else 'INCOMPLETE'}")
        print(f"  Results: {sum(1 for r in results if r['result']['status'] == 'success')}/{len(results)} succeeded")
        print(f"{'='*60}")

        return {"plan_results": results, "completed": completed}

    # ------------------------------------------------------------------
    # walk_to: Navigate to object/surface using WalkToSkill
    # ------------------------------------------------------------------
    def _execute_walk_to(self, target: str, stop_distance: float = 0.25, hold_arm: bool = False) -> dict:
        """Walk to an object or surface, stopping at stop_distance.

        Uses WalkToSkill with the semantic map position.
        If hold_arm=True, keeps arm at current position (manipulation mode)
        instead of switching to walking mode (which resets arm to default).
        """
        from ..skills.walk_to import WalkToSkill
        from ..config.skill_config import WalkToConfig

        # Get per-env target positions (each env has objects at different world positions)
        per_env_pos = self.semantic_map.get_per_env_position(target)
        if per_env_pos is not None:
            target_xy = per_env_pos[:, :2]  # [num_envs, 2]
        else:
            # Fallback: single position expanded to all envs
            target_pos = self.semantic_map.get_position(target)
            if target_pos is None:
                return {"status": "failed", "reason": f"Target '{target}' not found in semantic map"}
            target_xy = torch.tensor(
                [[target_pos[0], target_pos[1]]],
                dtype=torch.float32,
                device=self.device,
            ).expand(self.env.num_envs, -1)

        env = self.env

        if hold_arm and self._hold_arm_targets is not None:
            # Keep manipulation mode -- arm stays at current (grasp) position
            env.set_manipulation_mode(True)
            env.enable_arm_policy(False)
            arm_targets = self._hold_arm_targets

            # Detailed arm state logging for diagnostics
            import math as _math_wt
            from isaaclab.utils.math import quat_apply_inverse as _qai_wt
            ee_w, _ = env._compute_palm_ee()
            r_pos = env.robot.data.root_pos_w
            r_quat = env.robot.data.root_quat_w
            ee_b = _qai_wt(r_quat, ee_w - r_pos)
            r_yaw = get_yaw_from_quat(r_quat)
            print(f"  [WalkTo] Hold-arm walk | EE world: [{ee_w[0,0]:.3f},{ee_w[0,1]:.3f},{ee_w[0,2]:.3f}] | "
                  f"EE body: [{ee_b[0,0]:.3f},{ee_b[0,1]:.3f},{ee_b[0,2]:.3f}] | "
                  f"yaw={_math_wt.degrees(r_yaw[0].item()):.1f}deg")
            if ee_b[0, 0].item() < 0:
                print(f"  [WalkTo] WARNING: Arm BEHIND robot (body X={ee_b[0,0]:.3f}). Turning will be difficult!")
        else:
            # Normal walking mode -- arm returns to default
            env.set_manipulation_mode(False)
            arm_targets = None

        # For hold_arm walks, use omnidirectional controller
        # (much faster than turn-then-walk for nearby targets after lift)
        if hold_arm and arm_targets is not None:
            return self._omni_walk_to(target, target_xy, stop_distance, arm_targets)

        # Configure WalkTo skill with stop_distance
        walk_cfg = WalkToConfig()
        walk_cfg.stop_distance = stop_distance
        walk_cfg.max_steps = 4000  # 80s at 50Hz
        if hold_arm:
            walk_cfg.max_forward_vel = 0.3   # Slow walk when carrying object (CoM shifted)
            walk_cfg.max_yaw_rate = 0.8
            walk_cfg.max_lateral_vel = 0.15
        else:
            walk_cfg.max_yaw_rate = 0.8
            walk_cfg.max_lateral_vel = 0.4

        skill = WalkToSkill(config=walk_cfg, device=str(self.device))
        skill.reset(target_positions=target_xy)

        # Execute walk loop
        obs = env.get_obs()
        walk_done = False
        start_time = time.time()

        while self._is_running() and not walk_done:
            vel_cmd, walk_done, result = skill.step(obs)

            if arm_targets is not None:
                obs = env.step_manipulation(vel_cmd, arm_targets)
            else:
                obs = env.step(vel_cmd)

            if (obs["base_height"] < 0.2).all():
                return {"status": "failed", "reason": "All robots fell during walk"}

        walk_time = time.time() - start_time
        print(f"  [WalkTo] {result.status.name} in {walk_time:.1f}s, {result.steps_taken} steps")

        # Stabilize after walk
        print("  [WalkTo] Stabilizing...")
        for _ in range(50):
            if not self._is_running():
                break
            if arm_targets is not None:
                obs = env.step_manipulation(self._stand_cmd, arm_targets)
            else:
                obs = env.step(self._stand_cmd)

        if result.succeeded:
            return {"status": "success", "reason": f"Reached within {stop_distance}m of {target}"}
        else:
            return {"status": "failed", "reason": f"Walk failed: {result.reason}"}

    # ------------------------------------------------------------------
    # _omni_walk_to: Omnidirectional walk for hold_arm (nearby targets)
    # ------------------------------------------------------------------
    def _omni_walk_to(
        self,
        target_name: str,
        target_xy: torch.Tensor,
        stop_distance: float,
        arm_targets: torch.Tensor,
        max_steps: int = 2000,
    ) -> dict:
        """Walk omnidirectionally to target while holding arm position.

        Instead of turn-first-then-walk (which causes orbiting for nearby
        targets when the robot has rotated during lift), this moves directly
        toward the target using simultaneous forward, lateral, and yaw
        velocity commands proportional to body-frame error.

        Used after lift when the robot needs to sidestep to the basket.
        """
        env = self.env
        MIN_H = 0.5
        start_time = time.time()
        effective_dist = float('inf')
        step = 0

        for step in range(max_steps):
            if not self._is_running():
                break

            root_pos = env.robot.data.root_pos_w
            root_quat = env.robot.data.root_quat_w
            base_h = root_pos[:, 2]

            # Per-env distance, filtered for standing robots
            delta = target_xy - root_pos[:, :2]
            dist_per_env = delta.norm(dim=-1)
            standing_mask = base_h > MIN_H
            if standing_mask.any():
                effective_dist = dist_per_env[standing_mask].mean().item()
            else:
                effective_dist = dist_per_env.mean().item()

            if effective_dist < stop_distance:
                break

            # Body-frame direction to target
            yaw = get_yaw_from_quat(root_quat)
            cos_y = torch.cos(yaw)
            sin_y = torch.sin(yaw)
            dx_body = cos_y * delta[:, 0] + sin_y * delta[:, 1]
            dy_body = -sin_y * delta[:, 0] + cos_y * delta[:, 1]

            # Proportional velocity commands (forward + lateral simultaneously)
            # High gains + wide clamps for fast walking in manipulation mode
            vx = (dx_body * 1.2).clamp(-0.50, 0.50)
            vy = (dy_body * 0.8).clamp(-0.30, 0.30)

            # Gradually face the target
            target_heading = torch.atan2(delta[:, 1], delta[:, 0])
            heading_err = normalize_angle(target_heading - yaw)
            vyaw = (heading_err * 0.8).clamp(-0.6, 0.6)

            vel_cmd = torch.stack([vx, vy, vyaw], dim=-1)
            obs = env.step_manipulation(vel_cmd, arm_targets)

            if step % 50 == 0:
                h = obs["base_height"].mean().item()
                stand_count = (obs["base_height"] > MIN_H).sum().item()
                per_env = ", ".join(
                    f"e{i}={dist_per_env[i]:.2f}" for i in range(env.num_envs)
                )
                print(
                    f"  [OmniWalk] Step {step} | dist={effective_dist:.2f}m "
                    f"[{per_env}] | h={h:.2f} | "
                    f"stand={stand_count}/{env.num_envs} | "
                    f"cmd=[{vx[0]:.2f},{vy[0]:.2f},{vyaw[0]:.2f}]"
                )

            if (obs["base_height"] < 0.2).all():
                return {"status": "failed", "reason": "All robots fell during omni walk"}

        walk_time = time.time() - start_time
        print(
            f"  [OmniWalk] Finished in {walk_time:.1f}s, {step + 1} steps, "
            f"dist={effective_dist:.2f}m"
        )

        # Stabilize
        print("  [OmniWalk] Stabilizing...")
        for _ in range(50):
            if not self._is_running():
                break
            obs = env.step_manipulation(self._stand_cmd, arm_targets)

        if effective_dist < stop_distance:
            return {"status": "success", "reason": f"Reached within {stop_distance}m of {target_name}"}
        else:
            return {"status": "failed", "reason": f"OmniWalk timeout at {effective_dist:.2f}m"}

    # ------------------------------------------------------------------
    # walk_to_position: Navigate to specific world coordinates
    # ------------------------------------------------------------------
    def _execute_walk_to_position(self, x: float, y: float, stop_distance: float = 0.3) -> dict:
        """Walk to specific XY world coordinates."""
        from ..skills.walk_to import WalkToSkill
        from ..config.skill_config import WalkToConfig

        target_xy = torch.tensor(
            [[x, y]], dtype=torch.float32, device=self.device,
        ).expand(self.env.num_envs, -1)

        self.env.set_manipulation_mode(False)

        walk_cfg = WalkToConfig()
        walk_cfg.stop_distance = stop_distance
        walk_cfg.max_steps = 4000
        walk_cfg.max_yaw_rate = 0.8
        walk_cfg.max_lateral_vel = 0.4

        skill = WalkToSkill(config=walk_cfg, device=str(self.device))
        skill.reset(target_positions=target_xy)

        obs = self.env.get_obs()
        walk_done = False

        while self._is_running() and not walk_done:
            vel_cmd, walk_done, result = skill.step(obs)
            obs = self.env.step(vel_cmd)
            if (obs["base_height"] < 0.2).all():
                return {"status": "failed", "reason": "All robots fell"}

        # Stabilize
        for _ in range(50):
            if not self._is_running():
                break
            obs = self.env.step(self._stand_cmd)

        if result.succeeded:
            return {"status": "success", "reason": f"Reached ({x:.1f}, {y:.1f})"}
        else:
            return {"status": "failed", "reason": f"Walk failed: {result.reason}"}

    # ------------------------------------------------------------------
    # pre_reach: Raise arm HIGH before walking to table (avoid collision)
    # ------------------------------------------------------------------
    def _execute_pre_reach(self, target: str = "") -> dict:
        """Raise arm to a HIGH position BEFORE walking to the table.

        This must happen while the robot is still FAR from the table.
        The arm is raised to a fixed high body-frame position (not object-relative)
        so there is no table collision risk.

        After raising, the arm is FROZEN so walk_to can hold the position
        while approaching the table with hold_arm=True.

        Body-frame target: [0.25, -0.10, 0.20]
          - X=0.25: forward (toward table direction)
          - Y=-0.10: slightly right (toward right arm)
          - Z=0.20: above shoulder height (~0.97m world, well above table ~0.70m)
        """
        from isaaclab.utils.math import quat_apply

        env = self.env
        if env.arm_policy is None:
            return {"status": "failed", "reason": "No arm policy loaded"}

        # Enable debug visualization markers
        env.enable_debug_markers(True)

        # Switch to manipulation mode + arm policy
        env.set_manipulation_mode(True)
        env.enable_arm_policy(True)

        root_pos = env.robot.data.root_pos_w
        root_quat = env.robot.data.root_quat_w

        # Fixed HIGH position in body frame — well above table height
        pre_target_body = torch.tensor(
            [[0.25, -0.10, 0.20]], dtype=torch.float32, device=self.device,
        ).expand(env.num_envs, -1)

        # Convert to world frame
        pre_target_world = quat_apply(root_quat, pre_target_body) + root_pos

        print(f"  [PreReach] Raising arm to HIGH position (body [0.25, -0.10, 0.20])")
        print(f"  [PreReach] Target world: [{pre_target_world[0,0]:.3f}, {pre_target_world[0,1]:.3f}, {pre_target_world[0,2]:.3f}]")

        # Set target and run arm policy
        env.set_arm_target_world(pre_target_world)
        env.reset_arm_policy_state()

        # Run arm policy for 100 steps (raise arm above table height)
        for step in range(100):
            if not self._is_running():
                break
            obs = env.step_arm_policy(self._stand_cmd)

            if step % 20 == 0:
                ee_now, _ = env._compute_palm_ee()
                h = obs["base_height"].mean().item()
                standing = (obs["base_height"] > 0.5).sum().item()
                print(f"  [PreReach] Step {step:3d} | h={h:.2f} | stand={standing}/{env.num_envs} | "
                      f"EE=[{ee_now[0,0]:.2f},{ee_now[0,1]:.2f},{ee_now[0,2]:.2f}]")

        # FREEZE arm at raised position — save targets for hold_arm walk
        env.enable_arm_policy(False)
        self._hold_arm_targets = env.robot.data.joint_pos[:, env._arm_idx].clone()

        ee_final, _ = env._compute_palm_ee()
        print(f"  [PreReach] Final EE: [{ee_final[0,0]:.3f}, {ee_final[0,1]:.3f}, {ee_final[0,2]:.3f}]")

        return {"status": "success", "reason": f"Arm raised high (EE z={ee_final[0,2].item():.3f}m)"}

    # ------------------------------------------------------------------
    # reach: Extend arm to target using Stage 2 arm policy + magnetic attach
    # ------------------------------------------------------------------
    def _execute_reach(self, target: str) -> dict:
        """Reach toward a target with the Stage 2 arm policy.

        Strategy:
        1. Compute reachable target clamped to MAX_REACH from shoulder
        2. Set arm target and reach (from pre_reach elevated position)
        3. Magnetic attach when EE within 0.10m of object (10cm threshold)
        4. Hold phase: freeze arm, continue loco for stability
        """
        from isaaclab.utils.math import quat_apply_inverse, quat_apply

        env = self.env

        if env.arm_policy is None:
            return {"status": "failed", "reason": "No arm policy loaded"}

        # Enable debug visualization markers
        env.enable_debug_markers(True)

        # Switch to manipulation mode + arm policy
        # (may already be active from pre_reach — that's fine)
        env.set_manipulation_mode(True)
        env.enable_arm_policy(True)

        # Get per-env target position (refresh from semantic map)
        self.semantic_map.update()
        per_env_pos = self.semantic_map.get_per_env_position(target)
        if per_env_pos is not None:
            obj_pos_all = per_env_pos  # [num_envs, 3]
        else:
            target_pos = self.semantic_map.get_object_position(target)
            if target_pos is None:
                return {"status": "failed", "reason": f"Target '{target}' not found"}
            obj_pos_all = torch.tensor(
                [target_pos], dtype=torch.float32, device=self.device,
            ).expand(env.num_envs, -1)

        # Compute reachable target within arm workspace
        shoulder_offset = torch.tensor(self.SHOULDER_OFFSET, device=self.device)

        root_pos = env.robot.data.root_pos_w
        root_quat = env.robot.data.root_quat_w

        # Object in body frame
        obj_body = quat_apply_inverse(root_quat, obj_pos_all - root_pos)

        # Direction from shoulder to object
        obj_from_shoulder = obj_body - shoulder_offset.unsqueeze(0)
        dist_from_shoulder = obj_from_shoulder.norm(dim=-1, keepdim=True)

        # Debug: print world coordinates
        print(f"  [Reach] === DEBUG COORDINATES (env 0) ===")
        print(f"  [Reach]   Robot pos:    [{root_pos[0,0]:.3f}, {root_pos[0,1]:.3f}, {root_pos[0,2]:.3f}]")
        print(f"  [Reach]   Obj world:    [{obj_pos_all[0,0]:.3f}, {obj_pos_all[0,1]:.3f}, {obj_pos_all[0,2]:.3f}]")
        print(f"  [Reach]   Obj body:     [{obj_body[0,0]:.3f}, {obj_body[0,1]:.3f}, {obj_body[0,2]:.3f}]")
        print(f"  [Reach]   Shoulder:     [{shoulder_offset[0]:.3f}, {shoulder_offset[1]:.3f}, {shoulder_offset[2]:.3f}]")
        print(f"  [Reach]   Dist from shoulder: {dist_from_shoulder.mean():.3f}m (max: {self.MAX_REACH}m)")

        # Clamp 3D distance from shoulder to MAX_REACH
        dist_3d = obj_from_shoulder.norm(dim=-1, keepdim=True)
        scale_3d = torch.clamp(self.MAX_REACH / (dist_3d + 1e-6), max=1.0)

        reachable_target_body = shoulder_offset.unsqueeze(0) + obj_from_shoulder * scale_3d

        clamped = (dist_3d.mean() > self.MAX_REACH)
        if clamped:
            print(f"  [Reach] Target CLAMPED: {dist_3d.mean():.3f}m -> {self.MAX_REACH}m")
        else:
            print(f"  [Reach] Target within reach, using actual object position")

        print(f"  [Reach]   Reachable body: [{reachable_target_body[0,0]:.3f}, {reachable_target_body[0,1]:.3f}, {reachable_target_body[0,2]:.3f}]")

        # Convert to world frame and set as arm target
        reachable_target_world = quat_apply(root_quat, reachable_target_body) + root_pos
        print(f"  [Reach]   Reachable world: [{reachable_target_world[0,0]:.3f}, {reachable_target_world[0,1]:.3f}, {reachable_target_world[0,2]:.3f}]")

        env.set_arm_target_world(reachable_target_world)
        env.reset_arm_policy_state()

        # Save hold position -- PID will maintain robot here during reach + grasp + lift
        # This counteracts arm policy reaction forces that push robot backward
        # Persists across skills so grasp/lift can reuse it
        hold_pos_xy = root_pos[:, :2].clone()
        hold_yaw = get_yaw_from_quat(root_quat).clone()
        self._hold_pos_xy = hold_pos_xy
        self._hold_yaw = hold_yaw
        print(f"  [Reach] Hold position: [{hold_pos_xy[0,0]:.3f}, {hold_pos_xy[0,1]:.3f}], yaw={hold_yaw[0]:.3f}")

        # PID position holding during reach
        reach_steps = 160
        best_obj_dist = float('inf')
        best_ee_dist = float('inf')
        attached_during_reach = False

        print(f"  [Reach] Starting active reach ({reach_steps} steps) with PID hold...")
        for step in range(reach_steps):
            if not self._is_running():
                break

            # PID hold: counteract arm reaction forces
            hold_cmd, drift = self._compute_hold_cmd(hold_pos_xy, hold_yaw)
            obs = env.step_arm_policy(hold_cmd)

            # Periodically refresh hold position to prevent stale accumulation.
            # Without this, 160+ steps of drift make PID corrections enormous
            # at grasp start (0.5m * kp=1.5 = 0.75m/s clamped to 0.3) -> robot falls
            if step > 0 and step % 40 == 0:
                _rp = env.robot.data.root_pos_w
                _rq = env.robot.data.root_quat_w
                hold_pos_xy = _rp[:, :2].clone()
                hold_yaw = get_yaw_from_quat(_rq).clone()
                self._hold_pos_xy = hold_pos_xy
                self._hold_yaw = hold_yaw

            # Track distances using LIVE positions
            ee_world, _ = env._compute_palm_ee()
            live_obj_pos = env.pickup_obj.data.root_pos_w
            ee_dist = (ee_world - env._arm_target_world).norm(dim=-1).mean().item()
            obj_dist = (ee_world - live_obj_pos).norm(dim=-1).mean().item()
            best_ee_dist = min(best_ee_dist, ee_dist)
            best_obj_dist = min(best_obj_dist, obj_dist)

            if step % 10 == 0:
                h = obs["base_height"].mean().item()
                standing = (obs["base_height"] > 0.5).sum().item()
                print(f"  [Reach] Step {step:3d} | h={h:.2f} | "
                      f"stand={standing}/{env.num_envs} | "
                      f"EE->obj={obj_dist:.3f} | drift={drift:.3f} | "
                      f"cmd=[{hold_cmd[0,0]:.2f},{hold_cmd[0,1]:.2f},{hold_cmd[0,2]:.2f}]")

            # Magnetic attach: 0.20m trigger (20cm threshold)
            if not attached_during_reach and obj_dist < 0.20:
                attached_during_reach = env.attach_object_to_hand(max_dist=0.25)
                if attached_during_reach:
                    print(f"  [Reach] ** Magnetic attach at step {step}! dist={obj_dist:.3f}m **")
                    break

        print(f"  [Reach] Best EE->target: {best_ee_dist:.3f}m, Best EE->obj: {best_obj_dist:.3f}m")

        # Hold phase: freeze arm, PID hold position for stability
        # Shortened to 25 steps -- less time with destabilizing asymmetric arm load
        print("  [Reach] Holding arm position (25 steps)...")
        env.enable_arm_policy(False)
        self._hold_arm_targets = env.robot.data.joint_pos[:, env._arm_idx].clone()

        # Refresh hold position to CURRENT -- prevents fighting accumulated drift
        hold_pos_xy = env.robot.data.root_pos_w[:, :2].clone()
        hold_yaw = get_yaw_from_quat(env.robot.data.root_quat_w).clone()
        self._hold_pos_xy = hold_pos_xy
        self._hold_yaw = hold_yaw

        for step in range(25):
            if not self._is_running():
                break
            hold_cmd, drift = self._compute_hold_cmd(hold_pos_xy, hold_yaw)
            obs = env.step_manipulation(hold_cmd, self._hold_arm_targets)
            if step % 12 == 0:
                h = obs["base_height"].mean().item()
                standing = (obs["base_height"] > 0.5).sum().item()
                print(f"  [Reach] Hold {step}/25 | h={h:.2f} | stand={standing}/{env.num_envs} | drift={drift:.3f}")

        # Final distance check
        ee_world, _ = env._compute_palm_ee()
        live_obj_pos = env.pickup_obj.data.root_pos_w
        final_obj_dist = (ee_world - live_obj_pos).norm(dim=-1).mean().item()
        print(f"  [Reach] Final EE->obj: {final_obj_dist:.3f}m, attached={attached_during_reach}")

        return {
            "status": "success",
            "reason": f"Reached (best obj dist: {best_obj_dist:.3f}m, attached={attached_during_reach})",
            "attached": attached_during_reach,
        }

    # ------------------------------------------------------------------
    # grasp: Close fingers + magnetic attach
    # ------------------------------------------------------------------
    def _execute_grasp(self) -> dict:
        """Close fingers and magnetically attach object to palm.

        1. Close fingers (visual)
        2. Try magnetic attach (snap object to palm if close enough)
           - Skipped if already attached during reach phase
        3. Brief hold for stability
        Uses GENTLE PID hold (0.5x) -- robot is vulnerable with extended arm + object.
        """
        env = self.env
        env.finger_controller.close(hand="both")

        arm_targets = self._hold_arm_targets
        if arm_targets is None:
            arm_targets = env.robot.data.joint_pos[:, env._arm_idx].clone()

        # ALWAYS refresh hold position to current -- prevents accumulated drift from
        # reach phase causing large PID corrections that destabilize the robot
        root_pos = env.robot.data.root_pos_w
        root_quat = env.robot.data.root_quat_w
        hold_pos_xy = root_pos[:, :2].clone()
        hold_yaw = get_yaw_from_quat(root_quat).clone()
        self._hold_pos_xy = hold_pos_xy
        self._hold_yaw = hold_yaw

        # Check if already attached during reach
        already_attached = getattr(env, '_object_attached', False)
        if already_attached:
            print("  [Grasp] Object already attached from reach phase")

        h = root_pos[:, 2].mean().item()
        standing = (root_pos[:, 2] > 0.5).sum().item()
        print(f"  [Grasp] Start | h={h:.2f} | stand={standing}/{env.num_envs} | "
              f"hold=[{hold_pos_xy[0,0]:.3f},{hold_pos_xy[0,1]:.3f}]")

        # Close fingers for 20 steps with GENTLE PID hold
        for step in range(20):
            if not self._is_running():
                break
            hold_cmd, drift = self._compute_hold_cmd(hold_pos_xy, hold_yaw)
            hold_cmd *= 0.5  # Gentle -- robot is vulnerable with extended arm
            obs = env.step_manipulation(hold_cmd, arm_targets)

        # Magnetic attach (skip if already attached)
        if not already_attached:
            attached = env.attach_object_to_hand(max_dist=0.25)
        else:
            attached = True

        # Brief hold (25 steps) with gentle PID -- minimize time in unstable config
        for step in range(25):
            if not self._is_running():
                break
            hold_cmd, drift = self._compute_hold_cmd(hold_pos_xy, hold_yaw)
            hold_cmd *= 0.5  # Gentle corrections
            obs = env.step_manipulation(hold_cmd, arm_targets)

            if step % 12 == 0:
                h = obs["base_height"].mean().item()
                standing = (obs["base_height"] > 0.5).sum().item()
                print(f"  [Grasp] Hold {step}/25 | h={h:.2f} | stand={standing}/{env.num_envs} | "
                      f"drift={drift:.3f} | Attached: {attached}")

            # Early stop if majority fell -- no point holding further
            fallen = (obs["base_height"] < 0.5).sum().item()
            if fallen > env.num_envs // 2:
                print(f"  [Grasp] Majority fell ({fallen}/{env.num_envs}), ending grasp early")
                break

        if attached:
            return {"status": "success", "reason": "Object attached to hand"}
        else:
            return {"status": "failed", "reason": "Could not attach object (too far)"}

    # ------------------------------------------------------------------
    # lift: Raise arm above basket height using arm policy (intermediate target)
    # ------------------------------------------------------------------
    def _execute_lift(self) -> dict:
        """Lift the held object STRAIGHT UP above basket height using the arm policy.

        Sets a target position directly ABOVE current EE (vertical lift),
        then lets the Stage 2 arm policy figure out the joint angles.
        This is the intermediate target before lateral walk to basket.
        """
        from isaaclab.utils.math import quat_apply_inverse, quat_apply

        env = self.env
        if env.arm_policy is None:
            return {"status": "failed", "reason": "No arm policy loaded"}

        # Current EE and robot state
        ee_world, _ = env._compute_palm_ee()
        root_pos = env.robot.data.root_pos_w
        root_quat = env.robot.data.root_quat_w

        # Compute lift target in body frame: UP + ensure arm stays in FRONT
        ee_body = quat_apply_inverse(root_quat, ee_world - root_pos)
        lift_body = ee_body.clone()
        lift_body[:, 2] += 0.15  # 15cm straight up (body frame +Z)
        # CRITICAL: Clamp body X >= 0.15 to keep arm in front of robot
        # Without this, arm policy may swing arm behind body during lift
        lift_body[:, 0] = torch.clamp(lift_body[:, 0], min=0.15)

        # Clamp to arm workspace
        shoulder_offset = torch.tensor(self.SHOULDER_OFFSET, device=self.device)
        from_shoulder = lift_body - shoulder_offset.unsqueeze(0)
        dist = from_shoulder.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.MAX_REACH / (dist + 1e-6), max=1.0)
        lift_body_clamped = shoulder_offset.unsqueeze(0) + from_shoulder * scale

        # Convert to world frame
        lift_target_world = quat_apply(root_quat, lift_body_clamped) + root_pos

        print(f"  [Lift] Current EE body: [{ee_body[0,0]:.3f}, {ee_body[0,1]:.3f}, {ee_body[0,2]:.3f}]")
        print(f"  [Lift] Lift target body: [{lift_body_clamped[0,0]:.3f}, {lift_body_clamped[0,1]:.3f}, {lift_body_clamped[0,2]:.3f}]")
        print(f"  [Lift] Current EE world: [{ee_world[0,0]:.3f}, {ee_world[0,1]:.3f}, {ee_world[0,2]:.3f}]")
        print(f"  [Lift] Target world:     [{lift_target_world[0,0]:.3f}, {lift_target_world[0,1]:.3f}, {lift_target_world[0,2]:.3f}]")

        # Enable arm policy and set target
        env.enable_arm_policy(True)
        env.set_arm_target_world(lift_target_world)
        env.reset_arm_policy_state()

        # Save hold position — PID prevents drift during lift
        hold_pos_xy = root_pos[:, :2].clone()
        hold_yaw = get_yaw_from_quat(root_quat).clone()
        print(f"  [Lift] PID hold at [{hold_pos_xy[0,0]:.3f}, {hold_pos_xy[0,1]:.3f}]")

        # Run arm policy for 80 steps with PID hold (reduced from 120 — arm oscillates if too long)
        behind_count = 0
        for step in range(80):
            if not self._is_running():
                break
            hold_cmd, drift = self._compute_hold_cmd(hold_pos_xy, hold_yaw)
            obs = env.step_arm_policy(hold_cmd)

            if step % 10 == 0:
                ee_now, _ = env._compute_palm_ee()
                h = obs["base_height"].mean().item()
                standing = (obs["base_height"] > 0.5).sum().item()
                # Check body-frame X to detect arm going behind
                root_pos_c = env.robot.data.root_pos_w
                root_quat_c = env.robot.data.root_quat_w
                ee_body_c = quat_apply_inverse(root_quat_c, ee_now - root_pos_c)
                body_x = ee_body_c[0, 0].item()

                if step % 20 == 0:
                    print(f"  [Lift] Step {step:3d} | h={h:.2f} | stand={standing}/{env.num_envs} | "
                          f"drift={drift:.3f} | EE=[{ee_now[0,0]:.2f},{ee_now[0,1]:.2f},{ee_now[0,2]:.2f}] | "
                          f"bodyX={body_x:.2f}")

                # Early stop if arm goes behind robot for multiple checks
                if step > 15 and body_x < -0.05:
                    behind_count += 1
                    if behind_count >= 3:
                        print(f"  [Lift] EARLY STOP at step {step}: arm behind robot (bodyX={body_x:.3f}, count={behind_count})")
                        break
                else:
                    behind_count = max(0, behind_count - 1)

        # Freeze arm at current position
        env.enable_arm_policy(False)
        self._hold_arm_targets = env.robot.data.joint_pos[:, env._arm_idx].clone()

        # Log arm body-frame position at freeze time
        from isaaclab.utils.math import quat_apply_inverse as _qai_lift
        ee_freeze, _ = env._compute_palm_ee()
        root_pos_f = env.robot.data.root_pos_w
        root_quat_f = env.robot.data.root_quat_w
        ee_body_freeze = _qai_lift(root_quat_f, ee_freeze - root_pos_f)
        cur_yaw_freeze = get_yaw_from_quat(root_quat_f)
        import math as _math
        print(f"  [Lift] ARM FROZEN | EE world: [{ee_freeze[0,0]:.3f},{ee_freeze[0,1]:.3f},{ee_freeze[0,2]:.3f}] | "
              f"EE body: [{ee_body_freeze[0,0]:.3f},{ee_body_freeze[0,1]:.3f},{ee_body_freeze[0,2]:.3f}] | "
              f"yaw={_math.degrees(cur_yaw_freeze[0].item()):.1f}deg")

        # Stabilize after lift — PID hold continues during stabilization
        # Reduced to 30 steps to limit yaw drift with asymmetric load
        print("  [Lift] Stabilizing (30 steps)...")
        for step in range(30):
            if not self._is_running():
                break
            hold_cmd, drift = self._compute_hold_cmd(hold_pos_xy, hold_yaw)
            obs = env.step_manipulation(hold_cmd, self._hold_arm_targets)
            if step % 15 == 0:
                h = obs["base_height"].mean().item()
                standing = (obs["base_height"] > 0.5).sum().item()
                print(f"  [Lift] Stabilize {step}/30 | h={h:.2f} | stand={standing}/{env.num_envs} | drift={drift:.3f}")

        # Log arm body-frame position AFTER stabilization to detect drift
        ee_final, _ = env._compute_palm_ee()
        root_pos_s = env.robot.data.root_pos_w
        root_quat_s = env.robot.data.root_quat_w
        ee_body_final = _qai_lift(root_quat_s, ee_final - root_pos_s)
        cur_yaw_final = get_yaw_from_quat(root_quat_s)
        yaw_drift_deg = _math.degrees(cur_yaw_final[0].item() - cur_yaw_freeze[0].item())
        print(f"  [Lift] AFTER STAB | EE world: [{ee_final[0,0]:.3f},{ee_final[0,1]:.3f},{ee_final[0,2]:.3f}] | "
              f"EE body: [{ee_body_final[0,0]:.3f},{ee_body_final[0,1]:.3f},{ee_body_final[0,2]:.3f}] | "
              f"yaw={_math.degrees(cur_yaw_final[0].item()):.1f}deg (drift={yaw_drift_deg:.1f}deg)")

        # Check if arm ended up behind robot (body X < 0 means behind)
        if ee_body_final[0, 0].item() < 0:
            print(f"  [Lift] WARNING: Arm is BEHIND robot! body X={ee_body_final[0,0]:.3f}")

        return {"status": "success", "reason": f"Lifted to z={ee_final[0,2].item():.3f}m"}

    # ------------------------------------------------------------------
    # lateral_walk: Sidestep while holding arm position (slow, stable)
    # ------------------------------------------------------------------
    def _execute_lateral_walk(self, direction: str = "right", distance: float = 0.4, speed: float = 0.10) -> dict:
        """Walk laterally while holding the arm in lifted position.

        Args:
            direction: "right" or "left" (robot's perspective)
            distance: meters to walk sideways
            speed: lateral velocity (m/s) — default 0.10 for stability
        """
        env = self.env

        if self._hold_arm_targets is None:
            return {"status": "failed", "reason": "No arm targets held"}

        # Lateral velocity command (negative Y = right in body frame)
        vy = -speed if direction == "right" else speed
        lateral_cmd = torch.zeros(env.num_envs, 3, device=self.device)
        lateral_cmd[:, 1] = vy

        # Steps: distance / speed / control_dt
        steps = int(distance / speed / 0.02)  # 0.02s per step at 50Hz

        print(f"  [Lateral] Walking {direction} {distance}m at {speed}m/s ({steps} steps)")

        for step in range(steps):
            if not self._is_running():
                break
            obs = env.step_manipulation(lateral_cmd, self._hold_arm_targets)

            if step % 50 == 0:
                h = obs["base_height"].mean().item()
                standing = (obs["base_height"] > 0.5).sum().item()
                ee_world, _ = env._compute_palm_ee()
                print(f"  [Lateral] Step {step}/{steps} | h={h:.2f} | "
                      f"stand={standing}/{env.num_envs} | "
                      f"EE=[{ee_world[0,0]:.2f},{ee_world[0,1]:.2f},{ee_world[0,2]:.2f}]")

            # Safety check: stop if robot falls
            if (obs["base_height"] < 0.5).sum().item() > env.num_envs // 2:
                print(f"  [Lateral] WARNING: too many robots falling, stopping")
                break

        # Brief stabilize
        for _ in range(30):
            if not self._is_running():
                break
            obs = env.step_manipulation(self._stand_cmd, self._hold_arm_targets)

        return {"status": "success", "reason": f"Walked {direction} ~{distance}m"}

    # ------------------------------------------------------------------
    # lower: Lower arm into basket using arm policy
    # ------------------------------------------------------------------
    def _execute_lower(self) -> dict:
        """Lower the held object into the basket using the arm policy.

        Uses GRADUAL TARGET INTERPOLATION to prevent arm oscillation.
        Instead of jumping to the final target (which causes overshoot + oscillation),
        the target moves smoothly from the current EE body position to the final
        lower position over 80 steps, then holds for 20 steps.

        Body-frame target: [0.25, -0.20, -0.05]
          - X=0.25: forward (toward basket/table)
          - Y=-0.20: slightly right (right arm workspace)
          - Z=-0.05: slightly below root (basket height ~0.70m, root ~0.77m)
        """
        from isaaclab.utils.math import quat_apply_inverse, quat_apply

        env = self.env
        if env.arm_policy is None:
            return {"status": "failed", "reason": "No arm policy loaded"}

        # Current EE and robot state
        ee_world, _ = env._compute_palm_ee()
        root_pos = env.robot.data.root_pos_w
        root_quat = env.robot.data.root_quat_w
        ee_body = quat_apply_inverse(root_quat, ee_world - root_pos)

        # Start position (current EE in body frame)
        lower_body_start = ee_body.clone()

        # Final forward-down body-frame target (toward basket)
        lower_body_final = torch.tensor(
            [[0.25, -0.20, -0.05]], dtype=torch.float32, device=self.device,
        ).expand(env.num_envs, -1)

        print(f"  [Lower] Start EE body:  [{ee_body[0,0]:.3f}, {ee_body[0,1]:.3f}, {ee_body[0,2]:.3f}]")
        print(f"  [Lower] Target body:    [{lower_body_final[0,0]:.3f}, {lower_body_final[0,1]:.3f}, {lower_body_final[0,2]:.3f}]")

        # Clamp final target to arm workspace
        shoulder_offset = torch.tensor(self.SHOULDER_OFFSET, device=self.device)
        from_shoulder = lower_body_final - shoulder_offset.unsqueeze(0)
        dist = from_shoulder.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.MAX_REACH / (dist + 1e-6), max=1.0)
        lower_body_clamped = shoulder_offset.unsqueeze(0) + from_shoulder * scale

        print(f"  [Lower] Current EE world: [{ee_world[0,0]:.3f}, {ee_world[0,1]:.3f}, {ee_world[0,2]:.3f}]")

        # Enable arm policy
        env.enable_arm_policy(True)
        env.reset_arm_policy_state()

        # Save hold position -- PID prevents drift during lower
        hold_pos_xy = root_pos[:, :2].clone()
        hold_yaw = get_yaw_from_quat(root_quat).clone()

        # Gradually interpolate target from current to final over ramp_steps
        # This prevents the arm policy from overshooting and oscillating
        ramp_steps = 80
        total_steps = 100

        for step in range(total_steps):
            if not self._is_running():
                break

            # Smooth progress: 0.0 -> 1.0 over ramp_steps, then stays at 1.0
            t = min(step / ramp_steps, 1.0)

            # Interpolated body-frame target
            interp_body = lower_body_start + t * (lower_body_clamped - lower_body_start)

            # Convert to world frame using CURRENT root pose (robot sways)
            cur_root_pos = env.robot.data.root_pos_w
            cur_root_quat = env.robot.data.root_quat_w
            interp_world = quat_apply(cur_root_quat, interp_body) + cur_root_pos

            # Update arm target each step (smooth tracking)
            env.set_arm_target_world(interp_world)

            hold_cmd, drift = self._compute_hold_cmd(hold_pos_xy, hold_yaw)
            obs = env.step_arm_policy(hold_cmd)

            if step % 20 == 0:
                ee_now, _ = env._compute_palm_ee()
                ee_body_now = quat_apply_inverse(cur_root_quat,
                    ee_now - cur_root_pos)
                h = obs["base_height"].mean().item()
                standing = (obs["base_height"] > 0.5).sum().item()
                print(f"  [Lower] Step {step:3d} | t={t:.2f} | h={h:.2f} | stand={standing}/{env.num_envs} | "
                      f"drift={drift:.3f} | EE z={ee_now[0,2]:.3f} | "
                      f"bodyXZ=[{ee_body_now[0,0]:.2f},{ee_body_now[0,2]:.2f}]")

        # Freeze arm at current position
        env.enable_arm_policy(False)
        self._hold_arm_targets = env.robot.data.joint_pos[:, env._arm_idx].clone()

        ee_final, _ = env._compute_palm_ee()
        print(f"  [Lower] Final EE: [{ee_final[0,0]:.3f}, {ee_final[0,1]:.3f}, {ee_final[0,2]:.3f}]")

        return {"status": "success", "reason": f"Lowered to z={ee_final[0,2].item():.3f}m"}

    # ------------------------------------------------------------------
    # place: Release object, open fingers, return arm to default
    # ------------------------------------------------------------------
    def _execute_place(self) -> dict:
        """Detach object, open fingers, return arm to default.

        1. Detach object (drops under gravity)
        2. Open fingers
        3. Return arm to default pose (heuristic)
        4. 200-step transition
        5. Switch back to walking mode
        """
        from ..low_level.arm_controller import ArmPose

        env = self.env

        # Detach object (magnetic grasp release)
        env.detach_object()

        # Switch to heuristic arm (default pose)
        env.enable_arm_policy(False)
        env.arm_controller.set_pose(ArmPose.DEFAULT)
        env.finger_controller.open(hand="both")

        for step in range(200):
            if not self._is_running():
                break
            arm_targets = env.arm_controller.get_targets()
            obs = env.step_manipulation(self._stand_cmd, arm_targets)

            if step % 50 == 0:
                h = obs["base_height"].mean().item()
                standing = (obs["base_height"] > 0.5).sum().item()
                print(f"  [Place] Step {step:4d} | Height: {h:.2f}m | "
                      f"Standing: {standing}/{env.num_envs}")

        # Switch back to walking mode
        env.set_manipulation_mode(False)
        self._hold_arm_targets = None

        return {"status": "success", "reason": "Object released, arm returned to default"}
