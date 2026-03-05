"""
Velocity Command Generator
============================
Converts high-level targets (position, heading) into velocity commands
[vx, vy, vyaw] that the locomotion policy understands.

Uses P-controllers with configurable gains and velocity limits.
"""

from __future__ import annotations

import torch
import math
from typing import Optional

from ..config.joint_config import CMD_RANGE_LIMIT


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """Normalize angle to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def get_yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """
    Extract yaw angle from quaternion [w, x, y, z] or [x, y, z, w].

    Args:
        quat: Quaternion tensor [num_envs, 4]

    Returns:
        yaw: Yaw angle [num_envs]
    """
    # Isaac Lab uses [w, x, y, z] convention
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return yaw


class VelocityCommandGenerator:
    """
    Generates velocity commands to navigate to a target position.

    Takes robot state (position, orientation) and target position,
    outputs [vx, vy, vyaw] velocity commands for the locomotion policy.
    """

    def __init__(
        self,
        kp_linear: float = 1.0,
        kp_angular: float = 0.8,
        max_lin_vel_x: float = 0.8,
        max_lin_vel_y: float = 0.25,
        max_ang_vel_z: float = 0.2,
        min_lin_vel: float = 0.05,
        heading_deadzone: float = 0.05,
        device: str = "cuda",
    ):
        self.kp_linear = kp_linear
        self.kp_angular = kp_angular
        self.max_lin_vel_x = max_lin_vel_x
        self.max_lin_vel_y = max_lin_vel_y
        self.max_ang_vel_z = max_ang_vel_z
        self.min_lin_vel = min_lin_vel
        self.heading_deadzone = heading_deadzone
        self.device = torch.device(device)

    def compute_walk_command(
        self,
        robot_pos: torch.Tensor,
        robot_yaw: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute velocity command to walk toward target position.

        Args:
            robot_pos: Robot XY position [num_envs, 2]
            robot_yaw: Robot heading angle [num_envs]
            target_pos: Target XY position [num_envs, 2]

        Returns:
            cmd_vel: Velocity command [num_envs, 3] = [vx, vy, vyaw]
            distance: Distance to target [num_envs]
        """
        # Vector from robot to target in world frame
        delta_world = target_pos - robot_pos  # [num_envs, 2]
        distance = torch.norm(delta_world, dim=-1)  # [num_envs]

        # Desired heading (world frame)
        target_heading = torch.atan2(delta_world[:, 1], delta_world[:, 0])

        # Heading error
        heading_error = normalize_angle(target_heading - robot_yaw)

        # Transform delta to body frame for forward/lateral velocity
        cos_yaw = torch.cos(robot_yaw)
        sin_yaw = torch.sin(robot_yaw)
        dx_body = cos_yaw * delta_world[:, 0] + sin_yaw * delta_world[:, 1]
        dy_body = -sin_yaw * delta_world[:, 0] + cos_yaw * delta_world[:, 1]

        # P-controller for linear velocity (in body frame)
        vx = torch.clamp(
            self.kp_linear * dx_body,
            -self.max_lin_vel_x,
            self.max_lin_vel_x,
        )
        vy = torch.clamp(
            self.kp_linear * dy_body,
            -self.max_lin_vel_y,
            self.max_lin_vel_y,
        )

        # P-controller for angular velocity
        vyaw = torch.clamp(
            self.kp_angular * heading_error,
            -self.max_ang_vel_z,
            self.max_ang_vel_z,
        )

        # Apply heading deadzone
        vyaw = torch.where(
            torch.abs(heading_error) < self.heading_deadzone,
            torch.zeros_like(vyaw),
            vyaw,
        )

        cmd_vel = torch.stack([vx, vy, vyaw], dim=-1)  # [num_envs, 3]
        return cmd_vel, distance

    def compute_turn_command(
        self,
        robot_yaw: torch.Tensor,
        target_heading: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute velocity command to turn in place to target heading.

        Args:
            robot_yaw: Current heading [num_envs]
            target_heading: Desired heading [num_envs]

        Returns:
            cmd_vel: Velocity command [num_envs, 3] = [0, 0, vyaw]
            heading_error: Remaining error [num_envs]
        """
        heading_error = normalize_angle(target_heading - robot_yaw)

        vyaw = torch.clamp(
            self.kp_angular * heading_error,
            -self.max_ang_vel_z,
            self.max_ang_vel_z,
        )

        cmd_vel = torch.zeros(robot_yaw.shape[0], 3, device=self.device)
        cmd_vel[:, 2] = vyaw

        return cmd_vel, heading_error

    def compute_stand_command(self, num_envs: int = 1) -> torch.Tensor:
        """Return zero velocity command (stand still)."""
        return torch.zeros(num_envs, 3, device=self.device)


class AdaptivePIDWalkController:
    """PID-based adaptive walk controller that handles loco policy inconsistencies.

    Improvements over simple P-controller:
    1. Turn-then-walk: Turn in place when heading error > 45 degrees
    2. PID for distance: Integral detects stall, derivative damps oscillation
    3. Heading-scaled forward vel: Slow down when not aligned
    4. Stall detection: If no progress for N steps, boost velocity
    5. Smooth deceleration: Slow down on final approach

    Args:
        max_lin_vel_x: Max forward velocity (m/s)
        max_lin_vel_y: Max lateral velocity (m/s)
        max_ang_vel_z: Max yaw rate (rad/s)
        num_envs: Number of environments
        device: torch device
    """

    def __init__(
        self,
        max_lin_vel_x: float = 0.8,
        max_lin_vel_y: float = 0.4,
        max_ang_vel_z: float = 0.8,
        num_envs: int = 1,
        device: str = "cuda",
    ):
        self.max_lin_vel_x = max_lin_vel_x
        self.max_lin_vel_y = max_lin_vel_y
        self.max_ang_vel_z = max_ang_vel_z
        self.num_envs = num_envs
        self.device = torch.device(device)

        # PID gains for distance
        self.kp_dist = 1.2       # Proportional
        self.ki_dist = 0.02      # Integral (accumulates when stuck)
        self.kd_dist = 0.3       # Derivative (damps oscillation)

        # PID gains for heading
        self.kp_yaw = 1.5
        self.ki_yaw = 0.01
        self.kd_yaw = 0.2

        # Turn-first threshold (radians)
        self.turn_first_threshold = 0.7  # ~40 degrees

        # Stall detection
        self.stall_window = 50       # steps to check for stall
        self.stall_threshold = 0.05  # minimum progress in stall_window steps

        # State variables (per-env)
        self._integral_dist = torch.zeros(num_envs, device=self.device)
        self._integral_yaw = torch.zeros(num_envs, device=self.device)
        self._prev_dist = torch.zeros(num_envs, device=self.device)
        self._prev_heading_err = torch.zeros(num_envs, device=self.device)
        self._step_count = 0
        self._dist_history = []  # for stall detection
        self._stall_boost = torch.zeros(num_envs, device=self.device)

    def reset(self):
        """Reset controller state."""
        self._integral_dist.zero_()
        self._integral_yaw.zero_()
        self._prev_dist.zero_()
        self._prev_heading_err.zero_()
        self._step_count = 0
        self._dist_history = []
        self._stall_boost.zero_()

    def compute(
        self,
        robot_pos: torch.Tensor,
        robot_yaw: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute adaptive velocity command.

        Args:
            robot_pos: Robot XY position [num_envs, 2]
            robot_yaw: Robot heading angle [num_envs]
            target_pos: Target XY position [num_envs, 2]

        Returns:
            cmd_vel: [num_envs, 3] = [vx, vy, vyaw]
            distance: [num_envs]
        """
        # -- Compute errors --
        delta_world = target_pos - robot_pos
        distance = torch.norm(delta_world, dim=-1)
        target_heading = torch.atan2(delta_world[:, 1], delta_world[:, 0])
        heading_error = normalize_angle(target_heading - robot_yaw)

        # Body frame deltas
        cos_yaw = torch.cos(robot_yaw)
        sin_yaw = torch.sin(robot_yaw)
        dx_body = cos_yaw * delta_world[:, 0] + sin_yaw * delta_world[:, 1]
        dy_body = -sin_yaw * delta_world[:, 0] + cos_yaw * delta_world[:, 1]

        # -- PID for distance --
        dist_error = distance
        dist_deriv = distance - self._prev_dist if self._step_count > 0 else torch.zeros_like(distance)
        self._integral_dist += dist_error * 0.02  # dt ~ 0.02s at 50Hz
        # Anti-windup: clamp integral
        self._integral_dist.clamp_(-2.0, 2.0)

        pid_dist = (
            self.kp_dist * dist_error
            + self.ki_dist * self._integral_dist
            - self.kd_dist * dist_deriv  # negative: damp if dist increasing
        )

        # -- PID for yaw --
        yaw_deriv = (heading_error - self._prev_heading_err) if self._step_count > 0 else torch.zeros_like(heading_error)
        self._integral_yaw += heading_error * 0.02
        self._integral_yaw.clamp_(-1.0, 1.0)

        pid_yaw = (
            self.kp_yaw * heading_error
            + self.ki_yaw * self._integral_yaw
            + self.kd_yaw * yaw_deriv
        )

        # -- Heading-scaled forward velocity --
        # Reduce forward speed when heading error is large
        heading_alignment = torch.cos(heading_error).clamp(0, 1)  # 1.0=aligned, 0=perpendicular
        # Below 0 (>90 degrees off) means target is behind, use smaller value
        heading_alignment = torch.where(
            heading_alignment < 0.1, torch.full_like(heading_alignment, 0.1), heading_alignment
        )

        # -- Phase detection --
        abs_heading = heading_error.abs()
        turning_phase = abs_heading > self.turn_first_threshold  # Need to turn first

        # -- Compute velocities --
        # Forward velocity: PID-scaled, reduced by heading error
        vx_raw = pid_dist * heading_alignment
        vx = torch.where(
            turning_phase,
            # Small forward velocity during turn to prevent backward drift
            # (loco policy drifts backward during pure turning, causing outward spiral)
            torch.full_like(vx_raw, 0.1),
            torch.clamp(vx_raw, -self.max_lin_vel_x, self.max_lin_vel_x),
        )

        # Lateral velocity
        vy = torch.clamp(
            0.8 * dy_body,
            -self.max_lin_vel_y,
            self.max_lin_vel_y,
        )
        # Reduce lateral during turn phase
        vy = torch.where(turning_phase, vy * 0.3, vy)

        # Yaw velocity: stronger during turn phase
        vyaw = torch.clamp(pid_yaw, -self.max_ang_vel_z, self.max_ang_vel_z)

        # -- Stall detection --
        self._dist_history.append(distance.mean().item())
        if len(self._dist_history) > self.stall_window:
            old_dist = self._dist_history[-self.stall_window]
            new_dist = self._dist_history[-1]
            progress = old_dist - new_dist  # positive = getting closer
            mean_abs_heading = abs_heading.mean().item()
            if progress < self.stall_threshold:
                if mean_abs_heading > 0.5:
                    # Stalled because heading is wrong — don't boost forward (causes orbit)
                    # Just let the yaw controller align, then forward will work naturally
                    self._stall_boost = (self._stall_boost * 0.7).clamp(min=0)
                elif progress >= -0.05:
                    # Stalled but roughly aligned — boost to overcome
                    self._stall_boost = torch.clamp(
                        self._stall_boost + 0.02, 0, 0.4
                    )
                else:
                    # Going WRONG WAY (distance increasing fast) — decay boost fast
                    self._stall_boost = (self._stall_boost * 0.5).clamp(min=0)
            else:
                # Making progress, decay boost
                self._stall_boost = (self._stall_boost * 0.95).clamp(min=0)

        # -- Final approach deceleration --
        # Smooth slowdown within 0.8m (but not below 0.4x)
        approach_scale = torch.clamp(distance / 0.8, 0.4, 1.0)
        vx = vx * approach_scale
        vy = vy * approach_scale

        # Add stall boost scaled by heading alignment (prevents orbit when misaligned)
        aligned_boost = self._stall_boost * heading_alignment
        vx = (vx + aligned_boost).clamp(-self.max_lin_vel_x, self.max_lin_vel_x)

        # Ensure minimum forward vel when target is ahead and not turning
        vx = torch.where(
            (~turning_phase) & (dx_body > 0.1),
            vx.clamp(min=0.2),  # Minimum to overcome policy bias
            vx,
        )

        # -- Save state --
        self._prev_dist = distance.clone()
        self._prev_heading_err = heading_error.clone()
        self._step_count += 1

        cmd_vel = torch.stack([vx, vy, vyaw], dim=-1)
        return cmd_vel, distance
