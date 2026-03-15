"""
Locomotion Policy Wrapper (V6.2 + Stage 2 Loco)
=================================================
Loads a trained locomotion policy checkpoint and provides
a clean inference interface.

Architecture: 66 -> 512(LN+ELU) -> 256(LN+ELU) -> 128(LN+ELU) -> 15
  - 66-dim observation (lin_vel, ang_vel, gravity, leg/waist pos/vel, cmds, gait, prev_act, euler)
  - 15-dim action (12 legs + 3 waist)
  - Does NOT control arms or fingers — those are handled externally

Supports two checkpoint formats:
  V6.2:        {"model": {"actor.0.weight": ..., "log_std": ...}}
  Stage 2 Loco: {"model": {"loco_actor.0.weight": ..., "loco_log_std": ..., "arm_actor.*": ...}}

V6.2 / Stage 2 Loco trained on 29-DoF + DEX3 G1 robot:
  - 43 total joints (15 loco + 14 arm + 14 fingers)
  - Loco policy controls legs + waist only (15 joints)
  - Arms set to default pose, fingers open
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from typing import Optional


class LocomotionPolicy:
    """
    V6.2 unified locomotion policy: 66 obs -> 15 loco actions.

    The env builds the 66-dim observation and converts actions to joint targets.
    This class is a thin wrapper around the actor network.

    Usage:
        policy = LocomotionPolicy("path/to/model_best.pt", device="cuda:0")
        raw_actions = policy.get_raw_action(obs_66)  # [N, 15]
    """

    OBS_DIM = 66
    ACT_DIM = 15

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Build actor: 66 -> 512(LN+ELU) -> 256(LN+ELU) -> 128(LN+ELU) -> 15
        layers = []
        prev = self.OBS_DIM
        for h in [512, 256, 128]:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, self.ACT_DIM))
        self.actor = nn.Sequential(*layers)

        # Load actor weights from checkpoint — supports V6.2 and Stage 2 Loco formats
        state_dict = checkpoint.get("model", checkpoint)

        # Detect checkpoint format
        has_loco_actor = any(k.startswith("loco_actor.") for k in state_dict)
        has_actor = any(k.startswith("actor.") for k in state_dict)

        actor_dict = {}
        if has_loco_actor:
            # Stage 2 Loco format: loco_actor.X -> X
            ckpt_type = "Stage2Loco"
            for k, v in state_dict.items():
                if k.startswith("loco_actor."):
                    actor_dict[k[len("loco_actor."):]] = v
        elif has_actor:
            # V6.2 format: actor.X -> X
            ckpt_type = "V6.2"
            for k, v in state_dict.items():
                if k.startswith("actor."):
                    actor_dict[k[len("actor."):]] = v
        else:
            raise ValueError(f"Unknown checkpoint format. Keys: {list(state_dict.keys())[:10]}")

        self.actor.load_state_dict(actor_dict)
        self.actor.to(self.device)
        self.actor.eval()

        # Metadata
        self.iteration = checkpoint.get("iteration", "?")
        self.best_reward = checkpoint.get("best_reward", "?")
        self.curriculum_level = checkpoint.get("curriculum_level", "?")

        print(f"[LocoPolicy {ckpt_type}] Architecture: {self.OBS_DIM} -> [512,256,128](LN+ELU) -> {self.ACT_DIM}")
        print(f"[LocoPolicy {ckpt_type}] Checkpoint: {os.path.basename(checkpoint_path)}")
        print(f"[LocoPolicy {ckpt_type}] iter={self.iteration}, best_reward={self.best_reward}, "
              f"curriculum_level={self.curriculum_level}")

    @torch.no_grad()
    def get_raw_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Run actor network inference.

        Args:
            obs: [N, 66] observation tensor (pre-built by env)

        Returns:
            raw_actions: [N, 15] network output (not yet scaled/offset)
        """
        return self.actor(obs)
