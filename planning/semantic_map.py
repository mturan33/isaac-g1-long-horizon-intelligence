"""
Semantic Map
=============
Dual-mode world state representation for the VLM planning pipeline.

Modes:
    - "ground_truth": Reads object positions directly from Isaac Lab sim
    - "perception": Stub for YOLO + depth camera pipeline (future)

Both modes produce identical JSON output via get_json(), so the VLM planner
and skill executor work without any changes regardless of mode.
"""

from __future__ import annotations

import math
import torch
import numpy as np
from typing import Optional, Any


class SemanticMap:
    """Dual-mode semantic map for robot world understanding.

    Args:
        mode: "ground_truth" (sim) or "perception" (camera-based)
        env: HierarchicalG1Env instance (required for ground_truth mode)
        perception_module: G1PerceptionModule instance (for perception mode)
    """

    # Arm workspace radius -- objects within this distance are reachable
    ARM_REACH = 0.35  # meters

    def __init__(
        self,
        mode: str = "ground_truth",
        env: Any = None,
        perception_module: Any = None,
    ):
        self.mode = mode
        self.env = env
        self.perception = perception_module

        # World state
        self.objects: dict[str, dict] = {}
        self.surfaces: dict[str, dict] = {}
        self.interactables: dict[str, dict] = {}
        self.robot_state: dict = {}

        # Camera state (for VLM closed-loop)
        self.last_camera_rgb: Optional[np.ndarray] = None
        self.last_camera_b64: Optional[str] = None

        # Validate
        if mode == "ground_truth" and env is None:
            raise ValueError("ground_truth mode requires env parameter")
        if mode == "perception" and perception_module is None:
            print("[SemanticMap] WARNING: perception mode but no perception module provided")

        print(f"[SemanticMap] Initialized in '{mode}' mode")

    def update(self, rgb=None, depth=None, camera_intrinsics=None, camera_data=None):
        """Update world state. Call each frame before planning or execution.

        Args:
            camera_data: Optional [H,W,3] or [H,W,4] numpy array from head camera.
                Stored for VLM closed-loop replanning.
        """
        if self.mode == "ground_truth":
            self._update_from_sim()
        else:
            self._update_from_perception(rgb, depth, camera_intrinsics)
        self._update_robot_state()

        # Store camera image for VLM
        if camera_data is not None:
            self.last_camera_rgb = camera_data
            self.last_camera_b64 = self._rgb_to_base64(camera_data)

    # ------------------------------------------------------------------
    # Ground truth mode (Isaac Lab sim)
    # ------------------------------------------------------------------
    def _update_from_sim(self):
        """Read object/surface positions directly from simulation."""
        env = self.env

        # Pickup object (steering wheel)
        obj_pos = env.pickup_obj.data.root_pos_w[0].cpu().tolist()
        robot_pos = env.robot.data.root_pos_w[0].cpu().tolist()
        obj_dist = math.sqrt(
            (obj_pos[0] - robot_pos[0]) ** 2
            + (obj_pos[1] - robot_pos[1]) ** 2
        )
        self.objects["object_01"] = {
            "id": "object_01",
            "class": "steering_wheel",
            "position_3d": obj_pos,
            "graspable": True,
            "distance_to_robot": round(obj_dist, 3),
            "reachable": obj_dist < self.ARM_REACH,
            "confidence": 1.0,
        }

        # Table (PackingTable USD with built-in basket)
        table_pos = env.table.data.root_pos_w[0].cpu().tolist()
        self.surfaces["table_01"] = {
            "id": "table_01",
            "class": "table",
            "position_3d": table_pos,
            "placeable": True,
            "size": [1.5, 0.8, 0.70],  # PackingTable USD approximate dims
            "has_basket": True,  # grey basket on table surface
        }

        # Cabinet with drawer
        if hasattr(env, 'cabinet'):
            cab = env.cabinet
            cab_pos = cab.data.root_pos_w[0].cpu().tolist()
            # Read drawer joint state
            joint_pos = cab.data.joint_pos[0].cpu()  # [num_joints]
            # Find drawer_top_joint index
            joint_names = cab.joint_names
            drawer_idx = None
            for i, name in enumerate(joint_names):
                if "drawer_top" in name:
                    drawer_idx = i
                    break
            if drawer_idx is not None:
                drawer_pos_val = joint_pos[drawer_idx].item()
                max_travel = 0.30  # 30cm max drawer travel
                open_ratio = min(1.0, max(0.0, drawer_pos_val / max_travel))
            else:
                drawer_pos_val = 0.0
                open_ratio = 0.0

            # Read actual handle body position from articulation
            # This is "sim perception" — same output as real 3D YOLO would provide
            handle_pos = self._get_handle_position(cab, cab_pos)

            cab_dist = math.sqrt(
                (handle_pos[0] - robot_pos[0]) ** 2
                + (handle_pos[1] - robot_pos[1]) ** 2
            )

            self.interactables["drawer_01"] = {
                "id": "drawer_01",
                "class": "drawer",
                "position_3d": handle_pos,
                "handle_position": handle_pos,
                "open_ratio": round(open_ratio, 2),
                "state": "open" if open_ratio > 0.3 else "closed",
                "interaction": "pull",
                "distance_to_robot": round(cab_dist, 3),
            }

    # ------------------------------------------------------------------
    # Perception mode (camera-based -- stub)
    # ------------------------------------------------------------------
    def _update_from_perception(self, rgb, depth, intrinsics):
        """Update from YOLO + depth camera. Stub for future implementation."""
        if self.perception is None:
            print("[SemanticMap] Perception module not available, skipping update")
            return

        try:
            # Future: self.perception.detect(rgb, depth, intrinsics)
            # Parse Detection3D list into self.objects
            pass
        except Exception as e:
            print(f"[SemanticMap] Perception update failed: {e}")

    # ------------------------------------------------------------------
    # Robot state
    # ------------------------------------------------------------------
    def _update_robot_state(self):
        """Update robot position, heading, and holding status."""
        env = self.env
        if env is None:
            return

        root_pos = env.robot.data.root_pos_w[0].cpu().tolist()
        root_quat = env.robot.data.root_quat_w[0]

        # Heading from quaternion (yaw in degrees)
        w, x, y, z = root_quat[0].item(), root_quat[1].item(), root_quat[2].item(), root_quat[3].item()
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        heading_deg = math.degrees(yaw_rad)

        # Holding detection: fingers closed + nearest object close
        holding = None
        if hasattr(env, 'finger_controller') and env.finger_controller.is_closed():
            for obj_id, obj_data in self.objects.items():
                if obj_data.get("graspable", False):
                    obj_pos_3d = obj_data["position_3d"]
                    dist = math.sqrt(
                        (obj_pos_3d[0] - root_pos[0]) ** 2
                        + (obj_pos_3d[1] - root_pos[1]) ** 2
                    )
                    if dist < 0.5:
                        holding = obj_id
                        break

        # Determine stance
        base_height = root_pos[2]
        stance = "squatting" if base_height < 0.5 else "standing"

        self.robot_state = {
            "position": root_pos,
            "heading_deg": round(heading_deg, 1),
            "holding": holding,
            "stance": stance,
        }

    # ------------------------------------------------------------------
    # JSON output (identical format in both modes)
    # ------------------------------------------------------------------
    def get_json(self) -> dict:
        """Return standardized world state for VLM planner."""
        state = {
            "robot": self.robot_state,
            "objects": list(self.objects.values()),
            "surfaces": list(self.surfaces.values()),
        }
        if self.interactables:
            state["interactables"] = list(self.interactables.values())
        return state

    # ------------------------------------------------------------------
    # Position queries (for skill executor)
    # ------------------------------------------------------------------
    def get_object_position(self, object_id: str) -> Optional[list]:
        """Get real-time 3D position of an object."""
        obj = self.objects.get(object_id)
        if obj is None:
            for oid, odata in self.objects.items():
                if odata["class"] in object_id or object_id in odata["class"]:
                    return odata["position_3d"]
            return None
        return obj["position_3d"]

    def get_surface_position(self, surface_id: str) -> Optional[list]:
        """Get real-time 3D position of a surface."""
        surf = self.surfaces.get(surface_id)
        if surf is None:
            for sid, sdata in self.surfaces.items():
                if sdata["class"] in surface_id or surface_id in sid:
                    return sdata["position_3d"]
            return None
        return surf["position_3d"]

    def get_interactable_position(self, interactable_id: str) -> Optional[list]:
        """Get real-time 3D position of an interactable (e.g. drawer handle)."""
        ia = self.interactables.get(interactable_id)
        if ia is None:
            for iid, idata in self.interactables.items():
                if idata["class"] in interactable_id or interactable_id in idata["class"]:
                    return idata.get("handle_position", idata["position_3d"])
            return None
        return ia.get("handle_position", ia["position_3d"])

    def get_position(self, target_id: str) -> Optional[list]:
        """Get position of any object, surface, or interactable by id."""
        pos = self.get_object_position(target_id)
        if pos is not None:
            return pos
        pos = self.get_surface_position(target_id)
        if pos is not None:
            return pos
        return self.get_interactable_position(target_id)

    def get_per_env_position(self, target_id: str) -> Optional[torch.Tensor]:
        """Get per-env world position tensor [num_envs, 3] from sim."""
        if self.mode != "ground_truth" or self.env is None:
            return None

        # For interactables (drawer), return a walk target IN FRONT of the handle
        # (not the handle itself, which is flush with the cabinet body)
        if target_id in self.interactables:
            env = self.env
            if hasattr(env, 'cabinet'):
                cab = env.cabinet
                cab_pos = cab.data.root_pos_w[0].cpu().tolist()
                handle_pos = self._get_handle_position(cab, cab_pos)

                # Offset walk target 0.5m from handle toward robot
                # so robot stops in front of the handle, not at the cabinet edge
                robot_pos = env.robot.data.root_pos_w[0].cpu().tolist()
                dx = robot_pos[0] - handle_pos[0]
                dy = robot_pos[1] - handle_pos[1]
                dist_xy = math.sqrt(dx * dx + dy * dy) + 1e-6
                approach_offset = 0.5  # Stand 0.5m in front of handle
                walk_target = [
                    handle_pos[0] + approach_offset * dx / dist_xy,
                    handle_pos[1] + approach_offset * dy / dist_xy,
                    handle_pos[2],
                ]
                return torch.tensor(walk_target, device=env.device).unsqueeze(0).expand(
                    env.num_envs, -1
                ).clone()
            # Fallback to stored position
            ia = self.interactables[target_id]
            pos = ia.get("handle_position", ia["position_3d"])
            return torch.tensor(pos, device=self.env.device).unsqueeze(0).expand(
                self.env.num_envs, -1
            ).clone()

        entity = self._resolve_entity(target_id)
        if entity is not None:
            return entity.data.root_pos_w.clone()
        return None

    def _resolve_entity(self, target_id: str):
        """Resolve target_id to an Isaac Lab entity (RigidObject)."""
        env = self.env

        # Direct ID match
        id_map = {
            "object_01": env.pickup_obj,
            "table_01": env.table,
            "drawer_01": env.cabinet if hasattr(env, 'cabinet') else None,
        }
        if target_id in id_map:
            return id_map[target_id]

        # Class-based matching
        class_map = {
            "steering_wheel": env.pickup_obj,
            "object": env.pickup_obj,
            "table": env.table,
            "drawer": env.cabinet if hasattr(env, 'cabinet') else None,
            "cabinet": env.cabinet if hasattr(env, 'cabinet') else None,
        }
        for class_name, entity in class_map.items():
            if class_name in target_id or target_id in class_name:
                return entity
        return None

    # ------------------------------------------------------------------
    # Handle position from articulation bodies
    # ------------------------------------------------------------------
    def _get_handle_position(self, cabinet, cab_pos_list: list) -> list:
        """Get drawer handle world position from cabinet body data.

        Reads the actual body positions from the articulation to find
        the drawer handle. This is 'sim perception' — same output format
        as a real 3D object detector (YOLO + depth) would provide.

        Falls back to cabinet root + offset if body lookup fails.
        """
        try:
            # Try to find handle body by name
            body_names = cabinet.body_names
            handle_idx = None
            for i, name in enumerate(body_names):
                if "drawer_handle" in name or "handle_bottom" in name or "handle_top" in name:
                    handle_idx = i
                    break

            if handle_idx is not None:
                # Read actual handle body world position
                body_pos = cabinet.data.body_pos_w[0, handle_idx].cpu().tolist()
                return body_pos

            # Try: find any "drawer" body (the drawer itself, not just handle)
            for i, name in enumerate(body_names):
                if "drawer_top" in name and "joint" not in name:
                    body_pos = cabinet.data.body_pos_w[0, i].cpu().tolist()
                    return body_pos

        except Exception as e:
            pass  # Graceful fallback

        # Fallback: cabinet root + estimated offset
        # Use cabinet orientation to compute front-facing offset
        try:
            cab_quat = cabinet.data.root_quat_w[0].cpu()
            # Cabinet front direction from quaternion
            from isaaclab.utils.math import quat_apply
            forward = torch.tensor([0.3, 0.0, 0.25], dtype=torch.float32)
            offset = quat_apply(cab_quat.unsqueeze(0), forward.unsqueeze(0))[0].tolist()
            return [
                cab_pos_list[0] + offset[0],
                cab_pos_list[1] + offset[1],
                cab_pos_list[2] + offset[2],
            ]
        except Exception:
            return [cab_pos_list[0], cab_pos_list[1], cab_pos_list[2] + 0.25]

    # ------------------------------------------------------------------
    # Camera helpers (for VLM closed-loop)
    # ------------------------------------------------------------------
    def _rgb_to_base64(self, rgb_array: np.ndarray) -> str:
        """Convert RGB numpy array to base64 JPEG string for Ollama VLM."""
        import base64
        import io
        from PIL import Image

        img_np = rgb_array.astype(np.uint8)
        if img_np.shape[-1] == 4:  # RGBA → RGB
            img_np = img_np[:, :, :3]
        img = Image.fromarray(img_np)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def get_camera_base64(self) -> Optional[str]:
        """Get latest camera image as base64 JPEG for VLM."""
        return self.last_camera_b64

    def capture_camera(self):
        """Capture current camera frame from simulation and store it."""
        if self.env is None:
            return
        try:
            cam = self.env.scene["head_camera"]
            rgb = cam.data.output["rgb"][0].cpu().numpy()
            self.last_camera_rgb = rgb
            self.last_camera_b64 = self._rgb_to_base64(rgb)
        except Exception:
            pass  # Camera not available — graceful degrade
