#!/usr/bin/env python3
"""
VLM Planning Demo - Pick-and-Place (Locomanipulation G1 Style)
================================================================
Demonstrates the full VLM planning pipeline:
    1. Create scene (robot + PackingTable + steering wheel)
    2. SemanticMap reads object positions (ground truth)
    3. Planner generates skill sequence from natural language task
    4. SkillExecutor runs the plan on the environment

Scene matches PickPlace-Locomanipulation-G1-Abs-v0:
    PackingTable with built-in basket, steering wheel on surface.

Planners:
    --planner simple : Rule-based (no VLM needed, for testing)
    --planner vlm    : Qwen2.5-VL via Ollama (requires running Ollama)

Usage (from C:\\IsaacLab):
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\high_low_hierarchical_g1\\scripts\\demo_vlm_planning.py ^
        --num_envs 4 ^
        --checkpoint C:\\IsaacLab\\logs\\ulc\\g1_unified_stage1_2026-02-27_00-05-20\\model_best.pt ^
        --arm_checkpoint C:\\IsaacLab\\logs\\ulc\\ulc_g1_stage7_antigaming_2026-02-06_17-41-47\\model_best.pt ^
        --task "Pick up the steering wheel from the table" ^
        --planner simple
"""

# ============================================================================
# AppLauncher MUST be created before any Isaac Lab imports
# ============================================================================
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VLM Planning Demo - Pick-and-Place")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument(
    "--checkpoint", type=str, required=True,
    help="Path to V6.2 loco policy checkpoint (.pt file)",
)
parser.add_argument(
    "--arm_checkpoint", type=str, default=None,
    help="Path to Stage 2 arm policy checkpoint (.pt file)",
)
parser.add_argument(
    "--task", type=str,
    default="Pick up the steering wheel from the table",
    help="Natural language task description",
)
parser.add_argument(
    "--planner", type=str, default="simple", choices=["simple", "vlm"],
    help="Planner type: 'simple' (rule-based) or 'vlm' (Ollama Qwen3-VL)",
)
parser.add_argument(
    "--vlm_model", type=str, default="qwen3-vl:4b",
    help="Ollama model name for VLM planner (e.g. qwen3-vl:2b, qwen3-vl:4b, qwen3-vl:8b)",
)
parser.add_argument(
    "--no_stream", action="store_true",
    help="Disable live VLM reasoning display",
)
parser.add_argument(
    "--record", action="store_true",
    help="Record video frames (viewport capture → ffmpeg merge)",
)
parser.add_argument(
    "--record_dir", type=str, default="videos/demo_frames",
    help="Directory to save recorded frames",
)
parser.add_argument(
    "--record_fps", type=int, default=25,
    help="Target FPS for recording (captures every N-th sim step)",
)
parser.add_argument(
    "--closed_loop", action="store_true",
    help="Enable VLM closed-loop replanning (background thread, ~10s cycle)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# Isaac Lab imports (AFTER AppLauncher)
# ============================================================================
import os
import sys
import time
import json
import torch

# Force unbuffered stdout for real-time output in headless mode
sys.stdout.reconfigure(line_buffering=True)

import isaaclab.sim as sim_utils


# ============================================================================
# Video Recorder + Camera Tracker (offscreen rendering via replicator)
# ============================================================================

def _find_ffmpeg() -> str:
    """Find ffmpeg binary — prefers WinGet full build, then PATH."""
    import glob as _gl
    # Prefer WinGet full build (has libx264)
    pattern = os.path.expanduser(
        "~/AppData/Local/Microsoft/WinGet/Packages/*/ffmpeg*/bin/ffmpeg.exe"
    )
    candidates = _gl.glob(pattern)
    if candidates:
        return candidates[0]
    # Fallback to PATH
    import shutil as _sh
    return _sh.which("ffmpeg") or "ffmpeg"


class VideoRecorder:
    """Captures viewport frames at target FPS, then merges with ffmpeg.

    Uses omni.kit.viewport.utility for frame capture — requires GUI mode
    (do NOT use --headless when recording video).
    """

    def __init__(self, output_dir: str, fps: int = 25, control_dt: float = 0.02):
        self.output_dir = output_dir
        self.fps = fps
        self.frame_dir = os.path.join(output_dir, "frames")
        # Clean old frames from previous runs
        import shutil as _shutil_init
        if os.path.exists(self.frame_dir):
            _shutil_init.rmtree(self.frame_dir)
        os.makedirs(self.frame_dir, exist_ok=True)
        self.frame_count = 0
        self._sim_step = 0
        self._capture_interval = max(1, int(1.0 / (control_dt * fps)))

        from omni.kit.viewport.utility import get_active_viewport
        self.viewport = get_active_viewport()
        print(f"[VIDEO] Recorder initialized: {fps} FPS, "
              f"capture every {self._capture_interval} sim steps, "
              f"frames -> {self.frame_dir}")

    def set_camera_pose(self, eye: tuple, target: tuple):
        """Update the viewport camera position."""
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=list(eye), target=list(target))
        except Exception:
            pass

    def on_step(self):
        """Call after every sim step. Captures frame at correct interval."""
        self._sim_step += 1
        if self._sim_step % self._capture_interval == 0:
            from omni.kit.viewport.utility import capture_viewport_to_file
            frame_path = os.path.join(
                self.frame_dir, f"frame_{self.frame_count:06d}.png"
            )
            capture_viewport_to_file(self.viewport, frame_path)
            self.frame_count += 1

    def finalize(self, output_name: str = "demo_pickup.mp4"):
        """Merge frames into MP4 with ffmpeg."""
        import subprocess

        if self.frame_count == 0:
            print("[VIDEO] No frames captured!")
            return None

        output_path = os.path.join(self.output_dir, output_name)
        frame_pattern = os.path.join(self.frame_dir, "frame_%06d.png")
        ffmpeg_bin = _find_ffmpeg()

        # Wait for all async frame captures to flush to disk
        # capture_viewport_to_file is async — frames may not be written yet
        import time as _time_flush
        last_frame = os.path.join(self.frame_dir, f"frame_{self.frame_count - 1:06d}.png")
        print(f"[VIDEO] Waiting for {self.frame_count} frames to flush to disk...")
        for _wait in range(60):  # up to 30 seconds
            if os.path.exists(last_frame) and os.path.getsize(last_frame) > 0:
                break
            _time_flush.sleep(0.5)
        else:
            print(f"[VIDEO] WARNING: Last frame not found after 30s, proceeding anyway")
        # Extra settle to ensure all frames are fully written
        _time_flush.sleep(2.0)
        # Count actual frames on disk
        actual_frames = len([f for f in os.listdir(self.frame_dir) if f.endswith('.png')])
        print(f"[VIDEO] Frames on disk: {actual_frames}/{self.frame_count}")

        cmd = [
            ffmpeg_bin, "-y",
            "-framerate", str(self.fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_path,
        ]

        print(f"[VIDEO] Converting {actual_frames} frames to MP4...")
        print(f"[VIDEO] ffmpeg: {ffmpeg_bin}")
        print(f"[VIDEO] Command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0:
                print(f"[VIDEO] Saved: {output_path}")
                import shutil
                shutil.rmtree(self.frame_dir)
                print(f"[VIDEO] Cleaned up frames")
                return output_path
            else:
                stderr = result.stderr.decode(errors='replace')[:500]
                stdout = result.stdout.decode(errors='replace')[:200]
                print(f"[VIDEO] ffmpeg failed (rc={result.returncode})")
                print(f"[VIDEO] stderr: {stderr}")
                print(f"[VIDEO] stdout: {stdout}")
                print(f"[VIDEO] Frames saved in: {self.frame_dir}")
                return None
        except FileNotFoundError:
            print(f"[VIDEO] ffmpeg not found at: {ffmpeg_bin}")
            print(f"[VIDEO] Frames saved in: {self.frame_dir}")
            return None
        except Exception as e:
            print(f"[VIDEO] ffmpeg error: {e}")
            print(f"[VIDEO] Frames saved in: {self.frame_dir}")
            return None


class CameraTracker:
    """Tracks robot with body-frame camera offset (rotates with robot yaw).

    Camera stays at 45 degrees front-right regardless of robot orientation.
    Uses EMA smoothing on both position and yaw.
    Updates the VideoRecorder's offscreen camera (no viewport dependency).
    """

    EYE_RADIUS = 3.0
    EYE_ANGLE = -0.785      # -45deg front-right
    EYE_Z = 1.3
    TARGET_FWD = 0.2
    TARGET_Z = 0.65

    def __init__(self, recorder: "VideoRecorder" = None):
        self._recorder = recorder
        self._smooth_x = 0.0
        self._smooth_y = 0.0
        self._smooth_yaw = 0.0
        self._alpha_pos = 0.12
        self._alpha_yaw = 0.06
        self._initialized = False

    def update(self, robot_pos_w: torch.Tensor, robot_quat_w: torch.Tensor = None):
        """Call every sim step. Smooth EMA tracking with body-frame offset."""
        import math
        rx = robot_pos_w[0, 0].item()
        ry = robot_pos_w[0, 1].item()

        if robot_quat_w is not None:
            from high_low_hierarchical_g1.low_level.velocity_command import get_yaw_from_quat
            yaw = get_yaw_from_quat(robot_quat_w)[0].item()
        else:
            yaw = 0.0

        if not self._initialized:
            self._smooth_x = rx
            self._smooth_y = ry
            self._smooth_yaw = yaw
            self._initialized = True
            self._update_count = 0
        else:
            self._smooth_x += self._alpha_pos * (rx - self._smooth_x)
            self._smooth_y += self._alpha_pos * (ry - self._smooth_y)
            dyaw = math.atan2(math.sin(yaw - self._smooth_yaw),
                              math.cos(yaw - self._smooth_yaw))
            self._smooth_yaw += self._alpha_yaw * dyaw

        cam_angle_world = self._smooth_yaw + self.EYE_ANGLE
        eye = (
            self._smooth_x + self.EYE_RADIUS * math.cos(cam_angle_world),
            self._smooth_y + self.EYE_RADIUS * math.sin(cam_angle_world),
            self.EYE_Z,
        )
        target = (
            self._smooth_x + self.TARGET_FWD * math.cos(self._smooth_yaw),
            self._smooth_y + self.TARGET_FWD * math.sin(self._smooth_yaw),
            self.TARGET_Z,
        )

        # Update camera transform (works for both offscreen and viewport)
        if self._recorder is not None:
            self._recorder.set_camera_pose(eye, target)
            self._update_count = getattr(self, '_update_count', 0) + 1
            if self._update_count <= 2:
                print(f"[CAM] Update #{self._update_count}: eye={eye}, target={target}")
        else:
            # No recorder — still update viewport camera for GUI mode
            try:
                from isaacsim.core.utils.viewports import set_camera_view
                set_camera_view(eye=list(eye), target=list(target))
            except Exception:
                pass


def _wrap_env_for_recording(env, recorder: VideoRecorder = None,
                            camera_tracker: CameraTracker = None):
    """Monkey-patch env step methods for camera tracking and/or frame capture.

    This avoids modifying skill_executor.py — all ~22 step calls automatically
    get camera updates and frame capture without any changes.
    """
    original_step = env.step
    original_step_manipulation = env.step_manipulation
    original_step_arm_policy = env.step_arm_policy

    def _post_step():
        if camera_tracker is not None:
            camera_tracker.update(env.robot.data.root_pos_w, env.robot.data.root_quat_w)
        if recorder is not None:
            recorder.on_step()

    def _step_with_record(*args, **kwargs):
        result = original_step(*args, **kwargs)
        _post_step()
        return result

    def _step_manipulation_with_record(*args, **kwargs):
        result = original_step_manipulation(*args, **kwargs)
        _post_step()
        return result

    def _step_arm_policy_with_record(*args, **kwargs):
        result = original_step_arm_policy(*args, **kwargs)
        _post_step()
        return result

    env.step = _step_with_record
    env.step_manipulation = _step_manipulation_with_record
    env.step_arm_policy = _step_arm_policy_with_record
    features = []
    if camera_tracker is not None:
        features.append("camera tracking")
    if recorder is not None:
        features.append("frame capture")
    print(f"[VIDEO] Env step methods wrapped: {', '.join(features)}")

# Add parent of high_low_hierarchical_g1 to path so package imports work
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG_PARENT = os.path.dirname(_PKG_DIR)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from high_low_hierarchical_g1.envs.hierarchical_env import (
    HierarchicalG1Env, HierarchicalSceneCfg, PHYSICS_DT, CONTROL_DT,
)
from high_low_hierarchical_g1.planning.semantic_map import SemanticMap
from high_low_hierarchical_g1.planning.vlm_planner import OllamaVLMPlanner, SimplePlanner
from high_low_hierarchical_g1.planning.skill_executor import SkillExecutor


def _save_vlm_result(args, result, total_time, final_height, standing, num_envs, plan):
    """Save VLM run result with auto-scoring to results/vlm_runs/."""
    from datetime import datetime
    import json as _json

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "vlm_runs",
    )
    os.makedirs(results_dir, exist_ok=True)

    # --- Scoring (0-10) ---
    plan_results = result.get("plan_results", [])
    success_count = sum(1 for r in plan_results if r["result"]["status"] == "success")
    total_steps = len(plan_results)

    score = 0.0
    # Step completion: 5 points max (proportional)
    score += 5.0 * (success_count / max(total_steps, 1))
    # Standing bonus: 2 points if robot still standing at end
    if standing >= num_envs:
        score += 2.0
    elif standing > 0:
        score += 1.0
    # Speed bonus: 2 points max (under 60s = full, under 120s = half)
    if total_time < 60:
        score += 2.0
    elif total_time < 120:
        score += 1.0
    # Plan quality: 1 point if VLM plan was used (not fallback)
    vlm_used = not result.get("fallback", False)
    if vlm_used:
        score += 1.0
    score = round(min(10.0, score), 1)

    # --- Save ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{score}of10_{ts}.json"
    filepath = os.path.join(results_dir, filename)

    data = {
        "score": score,
        "timestamp": ts,
        "task": args.task,
        "planner": args.planner,
        "vlm_model": getattr(args, "vlm_model", None),
        "vlm_plan_used": vlm_used,
        "total_time_s": round(total_time, 1),
        "steps_succeeded": success_count,
        "steps_total": total_steps,
        "final_height_m": round(final_height, 3),
        "standing": f"{standing}/{num_envs}",
        "plan": [
            {"skill": s["skill"], "params": s.get("params", {})}
            for s in plan
        ],
        "step_results": [
            {
                "skill": r["skill"],
                "status": r["result"]["status"],
                "reason": r["result"].get("reason", ""),
            }
            for r in plan_results
        ],
        "scoring_breakdown": {
            "completion": f"{success_count}/{total_steps} = {5.0 * success_count / max(total_steps, 1):.1f}/5",
            "standing": f"{standing}/{num_envs} = {'2.0' if standing >= num_envs else '1.0' if standing > 0 else '0.0'}/2",
            "speed": f"{total_time:.1f}s = {'2.0' if total_time < 60 else '1.0' if total_time < 120 else '0.0'}/2",
            "vlm_plan": f"{'yes' if vlm_used else 'no'} = {'1.0' if vlm_used else '0.0'}/1",
        },
    }

    with open(filepath, "w") as f:
        _json.dump(data, f, indent=2)
    print(f"\n[Results] Saved: {filename} (score: {score}/10)")


def main():
    """Main VLM planning demo."""
    num_envs = args_cli.num_envs
    device = "cuda:0"

    print("=" * 60)
    print("  VLM Planning Demo - Pick-and-Place")
    print("=" * 60)
    print(f"  Task       : {args_cli.task}")
    print(f"  Planner    : {args_cli.planner}")
    print(f"  Environments: {num_envs}")
    print(f"  Loco ckpt  : {args_cli.checkpoint}")
    print(f"  Arm ckpt   : {args_cli.arm_checkpoint or 'None (heuristic)'}")
    if args_cli.planner == "vlm":
        print(f"  VLM model  : {args_cli.vlm_model}")
        print(f"  Streaming  : {not args_cli.no_stream}")
    if args_cli.record:
        print(f"  Recording  : {args_cli.record_dir} @ {args_cli.record_fps} FPS")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 0. Pre-load VLM model (parallel with Isaac Sim startup)
    # ------------------------------------------------------------------
    vlm = None
    if args_cli.planner == "vlm":
        vlm = OllamaVLMPlanner(
            model=args_cli.vlm_model,
            stream_reasoning=not args_cli.no_stream,
        )
        vlm.preload_model()

    # ------------------------------------------------------------------
    # 1. Create simulation context
    # ------------------------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYSICS_DT,
        device=device,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.5,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    if not args_cli.headless:
        # Initial camera view — will be overridden by CameraTracker after first step
        import math as _m
        _cam_ang = 0.0 + CameraTracker.EYE_ANGLE
        sim.set_camera_view(
            eye=[CameraTracker.EYE_RADIUS * _m.cos(_cam_ang),
                 CameraTracker.EYE_RADIUS * _m.sin(_cam_ang), CameraTracker.EYE_Z],
            target=[CameraTracker.TARGET_FWD, 0.0, CameraTracker.TARGET_Z],
        )

    # ------------------------------------------------------------------
    # 2. Create hierarchical environment
    # ------------------------------------------------------------------
    scene_cfg = HierarchicalSceneCfg()
    env = HierarchicalG1Env(
        sim=sim,
        scene_cfg=scene_cfg,
        checkpoint_path=args_cli.checkpoint,
        num_envs=num_envs,
        device=device,
        arm_checkpoint_path=args_cli.arm_checkpoint,
    )

    # ------------------------------------------------------------------
    # 2b. Camera tracking + video recording
    # ------------------------------------------------------------------
    recorder = None
    if args_cli.record:
        print("\n[Demo] Setting up video recording...")
        recorder = VideoRecorder(
            output_dir=args_cli.record_dir,
            fps=args_cli.record_fps,
            control_dt=CONTROL_DT,
        )
    # Camera tracker only in GUI mode (headless has no viewport)
    camera_tracker = None if args_cli.headless else CameraTracker(recorder=recorder)
    _wrap_env_for_recording(env, recorder=recorder, camera_tracker=camera_tracker)

    # ------------------------------------------------------------------
    # 3. Reset environment
    # ------------------------------------------------------------------
    print("\n[Demo] Resetting environment...")
    obs = env.reset()

    # Let the robot stabilize for a moment
    stand_cmd = torch.zeros(num_envs, 3, device=device)
    for _ in range(50):
        if not simulation_app.is_running():
            return
        obs = env.step(stand_cmd)

    # Verify object didn't fall off table during stabilization
    obj_z = env.pickup_obj.data.root_pos_w[0, 2].item()
    print(f"[Demo] Object z after stabilize: {obj_z:.3f}m "
          f"({'ON TABLE' if obj_z > 0.4 else 'FELL OFF TABLE!'})")

    # ------------------------------------------------------------------
    # 4. Create semantic map (ground truth mode)
    # ------------------------------------------------------------------
    print("\n[Demo] Creating semantic map (ground truth)...")
    semantic_map = SemanticMap(mode="ground_truth", env=env)
    semantic_map.update()

    # Print world state
    world_json = semantic_map.get_json()
    print(f"\n[Demo] World state:")
    print(json.dumps(world_json, indent=2, default=str))

    # ------------------------------------------------------------------
    # 5. Generate plan
    # ------------------------------------------------------------------
    print(f"\n[Demo] Planning: \"{args_cli.task}\"")
    plan = None
    vlm_fallback = False

    if args_cli.planner == "vlm" and vlm is not None:
        print(f"[Demo] Using VLM planner ({args_cli.vlm_model}, model pre-loaded)...")
        plan = vlm.plan(args_cli.task, world_json)

        if plan is None:
            print("[Demo] VLM planner failed, falling back to SimplePlanner")
            vlm_fallback = True

    if plan is None:
        print("[Demo] Using SimplePlanner (rule-based)...")
        simple = SimplePlanner()
        plan = simple.plan(args_cli.task, world_json)

    if not plan:
        print("[Demo] ERROR: No plan generated!")
        simulation_app.close()
        return

    print(f"\n[Demo] Generated plan ({len(plan)} steps):")
    for i, step in enumerate(plan):
        print(f"  {i+1}. {step['skill']}({step.get('params', {})})")

    # ------------------------------------------------------------------
    # 6. Execute plan
    # ------------------------------------------------------------------
    executor = SkillExecutor(
        env=env,
        semantic_map=semantic_map,
        simulation_app=simulation_app,
    )

    start_time = time.time()
    result = executor.execute_plan(plan)
    result["fallback"] = vlm_fallback
    total_time = time.time() - start_time

    # ------------------------------------------------------------------
    # 7. Print results
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  DEMO RESULTS")
    print(f"{'='*60}")
    print(f"  Task      : {args_cli.task}")
    print(f"  Planner   : {args_cli.planner}")
    print(f"  Completed : {result['completed']}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Steps:")
    for r in result["plan_results"]:
        status = r["result"]["status"]
        reason = r["result"].get("reason", "")
        symbol = "+" if status == "success" else "x"
        print(f"    [{symbol}] {r['skill']}: {status} - {reason}")

    # Final robot state
    final_height = obs["base_height"].mean().item()
    standing = (obs["base_height"] > 0.5).sum().item()
    print(f"\n  Final height: {final_height:.2f}m")
    print(f"  Standing: {standing}/{num_envs}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 7b. Save VLM run results
    # ------------------------------------------------------------------
    if args_cli.planner == "vlm":
        _save_vlm_result(args_cli, result, total_time, final_height, standing, num_envs, plan)
        if vlm is not None:
            vlm.unload_model()

    # ------------------------------------------------------------------
    # 8. Keep sim running briefly for visual inspection
    # ------------------------------------------------------------------
    print("\n[Demo] Holding for 1 second...")
    for i in range(50):  # ~1 second at 50Hz
        if not simulation_app.is_running():
            break
        obs = env.step(stand_cmd)

    # ------------------------------------------------------------------
    # 9. Finalize video recording
    # ------------------------------------------------------------------
    if recorder is not None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = f"demo_pickup_{timestamp}.mp4"
        print(f"\n[Demo] Finalizing video ({recorder.frame_count} frames)...")
        recorder.finalize(output_name=video_name)

    print("[Demo] Done.")


if __name__ == "__main__":
    import threading

    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        pass

    # Force exit with watchdog -- simulation_app.close() hangs on Windows
    def _force_exit():
        print("[Demo] Watchdog: forcing exit after timeout")
        os._exit(0)

    timer = threading.Timer(5.0, _force_exit)
    timer.daemon = True
    timer.start()

    try:
        simulation_app.close()
    except Exception:
        pass

    os._exit(0)
