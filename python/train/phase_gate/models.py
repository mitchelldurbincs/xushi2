from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class GateStatus(str, Enum):
    CLEARED = "CLEARED"
    NOT_CLEARED = "NOT_CLEARED"
    BLOCKED = "BLOCKED"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    HUMAN_INSPECTION_REQUIRED = "HUMAN_INSPECTION_REQUIRED"


class Comparator(str, Enum):
    GTE = ">="
    LTE = "<="
    GT = ">"
    LT = "<"
    EQ = "=="
    NE = "!="


class AggregationType(str, Enum):
    MEAN = "mean"
    STDDEV = "stddev"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


class IdentityRequirements(BaseModel):
    require_git_commit: bool = True
    require_config_path: bool = True
    require_seeds: bool = True
    min_unique_seeds: int = 1


class ArtifactRequirements(BaseModel):
    require_wandb_run_url: bool = True
    require_replay_artifacts: bool = True
    min_replay_count: int = 1
    viewer_command_template: str | None = "xushi2-viewer --replay {replay_path}"


class BlockersConfig(BaseModel):
    fail_on_crash: bool = True
    fail_on_nan: bool = True
    fail_on_missing_metrics: bool = True
    fail_on_import_error: bool = True
    fail_on_timeout_before_evidence: bool = True


class AggregationConfig(BaseModel):
    type: AggregationType = AggregationType.MEAN
    window: str = "last_n"
    n: int = 1

    @field_validator("window")
    @classmethod
    def validate_window(cls, value: str) -> str:
        if value not in {"last_n", "all"}:
            raise ValueError("aggregation.window must be one of: last_n, all")
        return value

    @field_validator("n")
    @classmethod
    def validate_n(cls, value: int) -> int:
        if value < 1:
            raise ValueError("aggregation.n must be >= 1")
        return value


class ObjectiveCheck(BaseModel):
    id: str
    metric: str
    source: str = "wandb"
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    comparator: Comparator
    threshold: float
    min_samples: int = 1
    on_missing: str = "EVIDENCE_INSUFFICIENT"

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        if value not in {"wandb", "local"}:
            raise ValueError("objective_checks.source must be one of: wandb, local")
        return value

    @field_validator("on_missing")
    @classmethod
    def validate_on_missing(cls, value: str) -> str:
        if value not in {"EVIDENCE_INSUFFICIENT", "FAIL"}:
            raise ValueError("objective_checks.on_missing must be EVIDENCE_INSUFFICIENT or FAIL")
        return value


class SubjectiveQuestion(BaseModel):
    id: str
    prompt: str


class SubjectiveChecks(BaseModel):
    required: bool = False
    trigger_if_objective_passed: bool = True
    questions: list[SubjectiveQuestion] = Field(default_factory=list)
    approval_rule: str = "all_yes"


class PhaseGateConfig(BaseModel):
    schema_version: int = 1
    phase: str
    identity_requirements: IdentityRequirements = Field(default_factory=IdentityRequirements)
    artifact_requirements: ArtifactRequirements = Field(default_factory=ArtifactRequirements)
    blockers: BlockersConfig = Field(default_factory=BlockersConfig)
    objective_checks: list[ObjectiveCheck] = Field(default_factory=list)
    required_tests: list[str] = Field(default_factory=list)
    subjective_checks: SubjectiveChecks = Field(default_factory=SubjectiveChecks)


class RunEvidence(BaseModel):
    run_id: str
    git_commit: str | None = None
    config_path: str | None = None
    seeds: list[int] = Field(default_factory=list)
    wandb_run_url: str | None = None
    replay_artifacts: list[str] = Field(default_factory=list)
    viewer_command: str | None = None
    crashed: bool = False
    saw_nan: bool = False
    import_error: bool = False
    timed_out_before_evidence: bool = False
    metrics: dict[str, list[float]] = Field(default_factory=dict)


class ObjectiveResult(BaseModel):
    id: str
    metric: str
    comparator: Comparator
    threshold: float
    value: float | None = None
    passed: bool
    reason: str | None = None


class HumanReview(BaseModel):
    available: bool = False
    decision: str | None = None
    checks: dict[str, str] = Field(default_factory=dict)
    comment: str | None = None


class GateDecision(BaseModel):
    schema_version: int = 1
    phase: str
    status: GateStatus
    final_reason: str
    run_id: str
    identity: dict
    artifacts: dict
    blockers: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    objective_results: list[ObjectiveResult] = Field(default_factory=list)
    subjective: dict = Field(default_factory=dict)
