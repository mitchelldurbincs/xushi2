"""Reference-policy anchor for stabilizing PPO from a warm-start checkpoint.

Every warm-started Phase 4 PPO run in the journal collapses the warm-start
policy within ~25-50 updates. One driver is that early PPO steps, taken against
an initially miscalibrated value baseline, move the policy far from the only
known-good behavior before the dense reward gradient has a chance to steer it.

This module adds an OpenAI-Five-style annealed KL/imitation anchor: a frozen
copy of a reference policy (typically the exact warm-start checkpoint) is
evaluated on the *live rollout states*, and the student is penalized for
drifting from it. The coefficient anneals linearly to zero, so the anchor only
constrains the fragile early updates and then releases the policy to the reward.

It differs from ``cap_duel_distill`` (which collects fresh mini-game episodes as
the anchor distribution): here the anchor distribution is the policy's own
rollout, so the penalty is measured exactly where the policy is drifting.

Off by default; enabled only when ``reference_anchor_coef > 0`` and a checkpoint
is supplied.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from train.composition_rehearsal import load_frozen_mappo_teacher
from train.mappo_model import MappoActorCritic, MappoConfig

# Continuous-action layout shared with the rest of the Phase 4 stack:
# index 0/1 are move_x/move_y, index 2 is the aim delta; binary index 0 is
# primary_fire.
_MOVE_ACTION_INDICES = (0, 1)
_AIM_ACTION_INDEX = 2
_PRIMARY_FIRE_BINARY_INDEX = 0


def _assert_reference_compatible(teacher: MappoActorCritic, student_cfg: MappoConfig) -> None:
    fields = {
        "obs_dim": (teacher.cfg.obs_dim, student_cfg.obs_dim),
        "action_dim": (teacher.cfg.action_dim, student_cfg.action_dim),
        "continuous_action_dim": (
            teacher.cfg.continuous_action_dim,
            student_cfg.continuous_action_dim,
        ),
        "binary_action_dim": (teacher.cfg.binary_action_dim, student_cfg.binary_action_dim),
        "gru_hidden": (teacher.cfg.gru_hidden, student_cfg.gru_hidden),
    }
    mismatches = {key: value for key, value in fields.items() if value[0] != value[1]}
    if mismatches:
        raise ValueError(f"reference_anchor checkpoint is incompatible: {mismatches}")
    if student_cfg.continuous_action_dim <= _AIM_ACTION_INDEX:
        raise ValueError("reference_anchor requires an aim row in continuous actions")
    if student_cfg.binary_action_dim <= _PRIMARY_FIRE_BINARY_INDEX:
        raise ValueError("reference_anchor requires a primary_fire binary head")


class ReferencePolicyAnchor:
    """Frozen reference policy plus an annealed drift penalty on rollout states."""

    def __init__(
        self,
        *,
        teacher: MappoActorCritic,
        coef: float,
        anneal_updates: int,
        aim_coef: float,
        fire_coef: float,
        move_coef: float,
    ) -> None:
        if coef < 0.0:
            raise ValueError("reference_anchor.coef must be >= 0")
        if anneal_updates < 0:
            raise ValueError("reference_anchor.anneal_updates must be >= 0")
        for name, value in {
            "aim_coef": aim_coef,
            "fire_coef": fire_coef,
            "move_coef": move_coef,
        }.items():
            if value < 0.0:
                raise ValueError(f"reference_anchor.{name} must be >= 0")
        self.teacher = teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.coef = float(coef)
        self.anneal_updates = int(anneal_updates)
        self.aim_coef = float(aim_coef)
        self.fire_coef = float(fire_coef)
        self.move_coef = float(move_coef)

    def to(self, device: torch.device | str) -> ReferencePolicyAnchor:
        self.teacher.to(device)
        return self

    def coef_for_update(self, update_idx: int) -> float:
        """Linear anneal from ``coef`` at update 1 to 0 at ``anneal_updates``.

        With ``anneal_updates <= 0`` the coefficient is held constant (no
        anneal), which is only useful for diagnostics; production configs set a
        finite anneal so the anchor releases the policy to the reward gradient.
        """
        if self.coef <= 0.0:
            return 0.0
        if self.anneal_updates <= 0:
            return self.coef
        if update_idx >= self.anneal_updates:
            return 0.0
        remaining = 1.0 - (float(update_idx) / float(self.anneal_updates))
        return self.coef * max(0.0, remaining)

    def init_hidden(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return self.teacher.init_hidden(batch_size).to(device)

    @torch.no_grad()
    def targets(
        self, obs_t: torch.Tensor, h_ref: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference (tanh-mean continuous action, fire probability, next hidden)."""
        features, h_next = self.teacher.actor_head_features(obs_t, h_ref)
        mean, logits, _target_logits = self.teacher.policy_heads_from_features(obs_t, features)
        logits = self.teacher.masked_binary_logits(obs_t, logits)
        cont = torch.tanh(mean)
        fire_prob = torch.sigmoid(logits[:, _PRIMARY_FIRE_BINARY_INDEX])
        return cont, fire_prob, h_next.detach()


def reference_anchor_step_losses(
    student_cont: torch.Tensor,
    student_fire_logits: torch.Tensor,
    reference_cont: torch.Tensor,
    reference_fire_prob: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample (aim MSE, move MSE, fire BCE) between student and reference.

    ``student_cont`` is ``tanh(mean)`` so it is on the same scale as the
    reference's tanh-mean target. Fire uses BCE-with-logits against the frozen
    reference's fire probability (a soft label), matching ``cap_duel_distill``.
    """
    aim_err = (student_cont[:, _AIM_ACTION_INDEX] - reference_cont[:, _AIM_ACTION_INDEX]) ** 2
    move_err = (
        (
            student_cont[:, list(_MOVE_ACTION_INDICES)]
            - reference_cont[:, list(_MOVE_ACTION_INDICES)]
        )
        ** 2
    ).mean(dim=-1)
    fire_bce = F.binary_cross_entropy_with_logits(
        student_fire_logits[:, _PRIMARY_FIRE_BINARY_INDEX],
        reference_fire_prob,
        reduction="none",
    )
    return aim_err, move_err, fire_bce


def build_reference_policy_anchor(
    cfg: MappoConfig,
) -> ReferencePolicyAnchor | None:
    """Construct the anchor from a resolved :class:`MappoConfig`, or ``None``."""
    if cfg.reference_anchor_coef <= 0.0 or not cfg.reference_anchor_checkpoint:
        return None
    teacher = load_frozen_mappo_teacher(Path(str(cfg.reference_anchor_checkpoint)))
    _assert_reference_compatible(teacher, cfg)
    return ReferencePolicyAnchor(
        teacher=teacher,
        coef=cfg.reference_anchor_coef,
        anneal_updates=cfg.reference_anchor_anneal_updates,
        aim_coef=cfg.reference_anchor_aim_coef,
        fire_coef=cfg.reference_anchor_fire_coef,
        move_coef=cfg.reference_anchor_move_coef,
    )
