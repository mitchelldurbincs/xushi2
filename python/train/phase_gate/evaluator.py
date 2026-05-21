from __future__ import annotations

from statistics import mean, median, pstdev

from .models import (
    AggregationType,
    Comparator,
    GateDecision,
    GateStatus,
    HumanReview,
    ObjectiveCheck,
    ObjectiveResult,
    PhaseGateConfig,
    RunEvidence,
)


def _compare(value: float, comparator: Comparator, threshold: float) -> bool:
    if comparator == Comparator.GTE:
        return value >= threshold
    if comparator == Comparator.LTE:
        return value <= threshold
    if comparator == Comparator.GT:
        return value > threshold
    if comparator == Comparator.LT:
        return value < threshold
    if comparator == Comparator.EQ:
        return value == threshold
    if comparator == Comparator.NE:
        return value != threshold
    raise ValueError(f"Unsupported comparator: {comparator}")


def _aggregate(series: list[float], check: ObjectiveCheck) -> float:
    data = series[:] if check.aggregation.window == "all" else series[-check.aggregation.n :]
    if check.aggregation.type == AggregationType.MEAN:
        return mean(data)
    if check.aggregation.type == AggregationType.STDDEV:
        return pstdev(data) if len(data) > 1 else 0.0
    if check.aggregation.type == AggregationType.MIN:
        return min(data)
    if check.aggregation.type == AggregationType.MAX:
        return max(data)
    if check.aggregation.type == AggregationType.MEDIAN:
        return median(data)
    raise ValueError(f"Unsupported aggregation: {check.aggregation.type}")


def evaluate_phase_gate(cfg: PhaseGateConfig, run: RunEvidence, review: HumanReview | None = None) -> GateDecision:
    review = review or HumanReview()
    blockers: list[str] = []
    if cfg.blockers.fail_on_crash and run.crashed:
        blockers.append("run_crashed")
    if cfg.blockers.fail_on_nan and run.saw_nan:
        blockers.append("nan_detected")
    if cfg.blockers.fail_on_import_error and run.import_error:
        blockers.append("import_error")
    if cfg.blockers.fail_on_timeout_before_evidence and run.timed_out_before_evidence:
        blockers.append("timeout_before_evidence")
    if blockers:
        return GateDecision(
            phase=cfg.phase,
            status=GateStatus.BLOCKED,
            final_reason="One or more blocking conditions were detected.",
            run_id=run.run_id,
            identity={"git_commit": run.git_commit, "config_path": run.config_path, "seeds": run.seeds},
            artifacts={"wandb_run_url": run.wandb_run_url, "replay_artifacts": run.replay_artifacts, "viewer_command": run.viewer_command},
            blockers=blockers,
        )

    missing: list[str] = []
    if cfg.identity_requirements.require_git_commit and not run.git_commit:
        missing.append("git_commit")
    if cfg.identity_requirements.require_config_path and not run.config_path:
        missing.append("config_path")
    if cfg.identity_requirements.require_seeds:
        if not run.seeds:
            missing.append("seeds")
        elif len(set(run.seeds)) < cfg.identity_requirements.min_unique_seeds:
            missing.append(f"min_unique_seeds<{cfg.identity_requirements.min_unique_seeds}")
    if cfg.artifact_requirements.require_wandb_run_url and not run.wandb_run_url:
        missing.append("wandb_run_url")
    if cfg.artifact_requirements.require_replay_artifacts and len(run.replay_artifacts) < cfg.artifact_requirements.min_replay_count:
        missing.append(f"replay_artifacts<{cfg.artifact_requirements.min_replay_count}")

    if missing:
        return GateDecision(
            phase=cfg.phase,
            status=GateStatus.EVIDENCE_INSUFFICIENT,
            final_reason="Required identity or artifacts are missing.",
            run_id=run.run_id,
            identity={"git_commit": run.git_commit, "config_path": run.config_path, "seeds": run.seeds},
            artifacts={"wandb_run_url": run.wandb_run_url, "replay_artifacts": run.replay_artifacts, "viewer_command": run.viewer_command},
            missing_evidence=missing,
        )

    objective_results: list[ObjectiveResult] = []
    for check in cfg.objective_checks:
        series = run.metrics.get(check.metric, [])
        if len(series) < check.min_samples:
            if check.on_missing == "EVIDENCE_INSUFFICIENT":
                return GateDecision(
                    phase=cfg.phase,
                    status=GateStatus.EVIDENCE_INSUFFICIENT,
                    final_reason=f"Insufficient samples for metric: {check.metric}",
                    run_id=run.run_id,
                    identity={"git_commit": run.git_commit, "config_path": run.config_path, "seeds": run.seeds},
                    artifacts={"wandb_run_url": run.wandb_run_url, "replay_artifacts": run.replay_artifacts, "viewer_command": run.viewer_command},
                    missing_evidence=[f"metric:{check.metric}:samples<{check.min_samples}"],
                )
            objective_results.append(
                ObjectiveResult(
                    id=check.id,
                    metric=check.metric,
                    comparator=check.comparator,
                    threshold=check.threshold,
                    value=None,
                    passed=False,
                    reason="insufficient_samples",
                )
            )
            continue
        value = _aggregate(series, check)
        objective_results.append(
            ObjectiveResult(
                id=check.id,
                metric=check.metric,
                comparator=check.comparator,
                threshold=check.threshold,
                value=value,
                passed=_compare(value, check.comparator, check.threshold),
            )
        )

    if any(not result.passed for result in objective_results):
        return GateDecision(
            phase=cfg.phase,
            status=GateStatus.NOT_CLEARED,
            final_reason="Objective checks did not meet configured thresholds.",
            run_id=run.run_id,
            identity={"git_commit": run.git_commit, "config_path": run.config_path, "seeds": run.seeds},
            artifacts={"wandb_run_url": run.wandb_run_url, "replay_artifacts": run.replay_artifacts, "viewer_command": run.viewer_command},
            objective_results=objective_results,
        )

    if cfg.subjective_checks.required and cfg.subjective_checks.trigger_if_objective_passed:
        if not review.available:
            return GateDecision(
                phase=cfg.phase,
                status=GateStatus.HUMAN_INSPECTION_REQUIRED,
                final_reason="Objective checks passed; awaiting human subjective review.",
                run_id=run.run_id,
                identity={"git_commit": run.git_commit, "config_path": run.config_path, "seeds": run.seeds},
                artifacts={"wandb_run_url": run.wandb_run_url, "replay_artifacts": run.replay_artifacts, "viewer_command": run.viewer_command},
                objective_results=objective_results,
                subjective={
                    "required": True,
                    "questions": [question.model_dump() for question in cfg.subjective_checks.questions],
                    "approval_rule": cfg.subjective_checks.approval_rule,
                },
            )
        if review.decision != "approved":
            return GateDecision(
                phase=cfg.phase,
                status=GateStatus.NOT_CLEARED,
                final_reason="Human subjective review rejected gate clearance.",
                run_id=run.run_id,
                identity={"git_commit": run.git_commit, "config_path": run.config_path, "seeds": run.seeds},
                artifacts={"wandb_run_url": run.wandb_run_url, "replay_artifacts": run.replay_artifacts, "viewer_command": run.viewer_command},
                objective_results=objective_results,
                subjective=review.model_dump(),
            )

    return GateDecision(
        phase=cfg.phase,
        status=GateStatus.CLEARED,
        final_reason="All objective checks passed and required subjective review is approved.",
        run_id=run.run_id,
        identity={"git_commit": run.git_commit, "config_path": run.config_path, "seeds": run.seeds},
        artifacts={"wandb_run_url": run.wandb_run_url, "replay_artifacts": run.replay_artifacts, "viewer_command": run.viewer_command},
        objective_results=objective_results,
        subjective=review.model_dump(),
    )
