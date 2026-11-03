"""Shared public contracts; checkpoints and answers remain in separate artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from . import VERSION


class Condition(StrEnum):
    UNAIDED = "A"
    ANALYSIS = "B"
    INSTRUMENTED = "C"


class WeightView(StrEnum):
    FULL = "full"
    BOUNDARY = "boundary_only"
    MASKED = "masked"
    WRONG = "matched_wrong"


FAMILIES = {
    "rule_recovery": "read",
    "causal_forecast": "read",
    "edit_effects": "read",
    "coupled_repair": "repair",
    "compositional_synthesis": "write",
    "loop_repair": "repair",
}
CORE_FAMILIES = tuple(f for f in FAMILIES if f != "loop_repair")


@dataclass(frozen=True)
class Limits:
    model_turns: int = 16
    analysis_calls: int = 12
    input_tokens: int = 160_000
    output_tokens: int = 32_000
    cpu_seconds: int = 120
    memory_mb: int = 2048
    wall_seconds: int = 900
    example_evaluations: int = 128
    candidate_patches: int = 8
    activation_scalars: int = 256_000
    episodes: int = 16
    episode_steps: int = 64


@dataclass
class Task:
    id: str
    parent_problem_id: str
    mechanism_family_id: str
    split: str
    family: str
    skill: str
    target_stratum: str
    capacity_tier: str
    architecture: dict[str, Any]
    instruction: str
    public: dict[str, Any]
    directory: Path
    conditions: tuple[str, ...] = ("A", "B", "C")
    limits: Limits = field(default_factory=Limits)

    @property
    def weights_path(self) -> Path:
        return self.directory / "target.npz"

    @property
    def private_path(self) -> Path:
        return self.directory / "private.json"

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "2", "benchmark_version": VERSION,
            "suite_tier": "challenge", "admission_status": "candidate",
            **{key: getattr(self, key) for key in (
                "id", "parent_problem_id", "mechanism_family_id", "split", "family", "skill",
                "target_stratum", "capacity_tier", "architecture", "instruction", "public")},
            "conditions": list(self.conditions), "limits": asdict(self.limits),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], directory: Path) -> "Task":
        required = ("id", "parent_problem_id", "mechanism_family_id", "split", "family", "skill",
                    "target_stratum", "capacity_tier", "architecture", "instruction", "public")
        if data.get("benchmark_version") != VERSION or data.get("suite_tier") != "challenge":
            raise ValueError("Not a v0.2 challenge-candidate manifest")
        if data["family"] not in FAMILIES or FAMILIES[data["family"]] != data["skill"]:
            raise ValueError("Unknown challenge family/skill")
        conditions = tuple(data.get("conditions", ("A", "B", "C")))
        if conditions != (("C",) if data["family"] == "loop_repair" else ("A", "B", "C")):
            raise ValueError("Invalid access conditions for task family")
        return cls(**{key: data[key] for key in required}, directory=directory,
                   conditions=conditions, limits=Limits(**data.get("limits", {})))


@dataclass
class Score:
    success: bool
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "scored"
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
