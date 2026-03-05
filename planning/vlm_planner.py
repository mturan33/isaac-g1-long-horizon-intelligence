"""
VLM Task Planner
=================
Local VLM-based task planner using Qwen2.5-VL via Ollama,
plus a rule-based SimplePlanner fallback.

VLMPlanner:
    - Connects to local Ollama instance
    - Sends semantic map JSON + optional RGB image
    - Receives structured skill plan as JSON

SimplePlanner:
    - Rule-based fallback (no VLM needed)
    - Keyword matching for pick-and-place tasks
    - Uses semantic map to find objects and surfaces
"""

from __future__ import annotations

import json
import re
from typing import Optional

# Ollama HTTP client (requests is optional)
try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# Available skills for validation
AVAILABLE_SKILLS = {"walk_to", "pre_reach", "reach", "grasp", "lift", "lateral_walk", "lower", "place", "walk_to_position"}


class VLMPlanner:
    """Task planner using local VLM (Qwen2.5-VL) via Ollama.

    Args:
        model: Ollama model name (default: "qwen2.5vl:7b")
        ollama_url: Ollama API base URL
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        model: str = "qwen2.5vl:7b",
        ollama_url: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        self.model = model
        self.url = ollama_url
        self.timeout = timeout

        if not _HAS_REQUESTS:
            print("[VLMPlanner] WARNING: 'requests' not installed. VLM planning unavailable.")

    def plan(
        self,
        task: str,
        semantic_map_json: dict,
        rgb_image: Optional[str] = None,
    ) -> Optional[list]:
        """Generate a skill plan from a natural language task.

        Args:
            task: Natural language task description
            semantic_map_json: World state from SemanticMap.get_json()
            rgb_image: Optional base64-encoded RGB image

        Returns:
            List of skill steps [{skill, params}, ...] or None if failed
        """
        if not _HAS_REQUESTS:
            return None

        prompt = self._build_prompt(task, semantic_map_json)

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
            }
            if rgb_image is not None:
                payload["images"] = [rgb_image]

            response = requests.post(
                f"{self.url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            plan = self._parse_response(result)
            return plan

        except requests.ConnectionError:
            print(f"[VLMPlanner] Cannot connect to Ollama at {self.url}")
            return None
        except requests.Timeout:
            print(f"[VLMPlanner] Ollama request timed out ({self.timeout}s)")
            return None
        except Exception as e:
            print(f"[VLMPlanner] Error: {e}")
            return None

    def _build_prompt(self, task: str, semantic_map: dict) -> str:
        """Build few-shot prompt with skill library and world state."""
        return f"""You are a robot task planner for a Unitree G1 humanoid robot.

AVAILABLE SKILLS:
- pre_reach(target): Raise arm HIGH before approaching table. MUST be called BEFORE walk_to to avoid table collision.
- walk_to(target, stop_distance, hold_arm): Walk to object/surface. Use hold_arm=true after pre_reach.
- reach(target): Extend right arm DOWN toward target object using RL policy + magnetic attach at 10cm.
- grasp(): Close fingers to grasp object.
- lift(): Raise arm straight up above basket height after grasping.
- lateral_walk(direction, distance, speed): Walk sideways while holding object. direction="right"/"left". speed=0.10 for stability.
- lower(): Lower arm into basket/container after positioning above it.
- place(): Open fingers to release held object, return arm to default.
- walk_to_position(x, y): Walk to specific world coordinates.

ROBOT STATE:
{json.dumps(semantic_map, indent=2)}

TASK: {task}

IMPORTANT RULES:
1. ALWAYS pre_reach BEFORE walk_to (raise arm before approaching table)
2. Walk with hold_arm=true after pre_reach, stop_distance=0.25 (table blocks closer)
3. Arm workspace is 0.50m - robot must be close (stop_distance=0.25)
4. Always reach before grasping
5. Output ONLY valid JSON

OUTPUT FORMAT (JSON object with "plan" key containing array):
{{"plan": [{{"skill": "skill_name", "params": {{...}}}}]}}"""

    def _parse_response(self, response: dict) -> Optional[list]:
        """Parse and validate Ollama response."""
        try:
            text = response.get("response", "")
            data = json.loads(text)

            # Handle both {plan: [...]} and bare [...]
            if isinstance(data, dict) and "plan" in data:
                plan = data["plan"]
            elif isinstance(data, list):
                plan = data
            else:
                print(f"[VLMPlanner] Unexpected response format: {type(data)}")
                return None

            # Validate each step
            validated = []
            for step in plan:
                skill = step.get("skill", "")
                if skill not in AVAILABLE_SKILLS:
                    print(f"[VLMPlanner] Unknown skill: {skill}, skipping")
                    continue
                validated.append({
                    "skill": skill,
                    "params": step.get("params", {}),
                })

            if not validated:
                print("[VLMPlanner] No valid skills in response")
                return None

            return validated

        except json.JSONDecodeError as e:
            print(f"[VLMPlanner] JSON parse error: {e}")
            # Try to extract JSON from text
            text = response.get("response", "")
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    plan = json.loads(match.group())
                    return [
                        {"skill": s["skill"], "params": s.get("params", {})}
                        for s in plan
                        if s.get("skill") in AVAILABLE_SKILLS
                    ] or None
                except json.JSONDecodeError:
                    pass
            return None


class SimplePlanner:
    """Rule-based fallback planner. No VLM required.

    Generates standard skill sequences from keyword matching
    on the task string and the semantic map contents.
    """

    def plan(self, task: str, semantic_map_json: dict) -> list:
        """Generate a skill plan from task keywords and semantic map.

        Args:
            task: Natural language task (e.g., "Pick up the steering wheel from the table")
            semantic_map_json: World state from SemanticMap.get_json()

        Returns:
            List of skill steps [{skill, params}, ...]
        """
        task_lower = task.lower()
        objects = semantic_map_json.get("objects", [])
        surfaces = semantic_map_json.get("surfaces", [])

        # Detect task type
        is_pick = any(w in task_lower for w in ["pick", "grab", "grasp", "get", "take"])
        is_place = any(w in task_lower for w in ["place", "put", "set", "drop"])

        if is_pick:
            return self._plan_pick(task_lower, objects)
        elif is_place:
            return self._plan_place(task_lower, surfaces)
        else:
            # Default: walk to first object
            if objects:
                return [
                    {"skill": "walk_to", "params": {"target": objects[0]["id"], "stop_distance": 0.30}},
                ]
            return []

    def _plan_pick(self, task: str, objects: list) -> list:
        """Pick from table and place in basket:
        pre_reach -> walk(hold_arm) -> reach -> grasp -> lift -> lateral_walk -> lower -> place.

        Pre_reach raises arm HIGH while still far from table (no collision).
        Walk approaches with arm held up.  Reach descends from above.
        """
        target_obj = self._find_target_object(task, objects)
        if target_obj is None:
            print("[SimplePlanner] No graspable object found in scene")
            return []
        return [
            # 1. Raise arm HIGH before approaching table (avoids collision)
            {"skill": "pre_reach", "params": {"target": target_obj["id"]}},
            # 2. Walk to object with arm held up (0.25m — table blocks closer)
            {"skill": "walk_to", "params": {"target": target_obj["id"], "stop_distance": 0.25, "hold_arm": True}},
            # 3. Reach down to object and magnetically attach (10cm threshold)
            {"skill": "reach", "params": {"target": target_obj["id"]}},
            # 4. Close fingers around object
            {"skill": "grasp", "params": {}},
            # 5. Lift arm straight up above basket height
            {"skill": "lift", "params": {}},
            # 6. Walk to basket with arm held (0.45m — robot at table edge is close enough)
            {"skill": "walk_to", "params": {"target": "table_01", "stop_distance": 0.45, "hold_arm": True}},
            # 7. Lower arm into basket
            {"skill": "lower", "params": {}},
            # 8. Release into basket
            {"skill": "place", "params": {}},
        ]

    def _plan_place(self, task: str, surfaces: list) -> list:
        """Place only (assumes already holding): walk -> place."""
        target_surface = self._find_target_surface(task, surfaces)
        if target_surface is None:
            return []
        return [
            {"skill": "walk_to", "params": {"target": target_surface["id"], "stop_distance": 0.55, "hold_arm": True}},
            {"skill": "place", "params": {}},
        ]

    def _find_target_object(self, task: str, objects: list) -> Optional[dict]:
        """Find the most relevant graspable object from the task string."""
        # Try matching object class names to task keywords
        for obj in objects:
            if not obj.get("graspable", False):
                continue
            obj_class = obj["class"].lower()
            # Match any word in the class name
            for word in obj_class.split("_"):
                if word in task:
                    return obj

        # Fallback: first graspable object
        for obj in objects:
            if obj.get("graspable", False):
                return obj
        return None

    def _find_target_surface(self, task: str, surfaces: list) -> Optional[dict]:
        """Find the target surface from the task string."""
        # Check for surface class match
        for surf in surfaces:
            surf_class = surf["class"].lower()
            if surf_class in task:
                return surf

        # Fallback: first surface
        if surfaces:
            return surfaces[0]
        return None
