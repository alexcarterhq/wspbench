"""Public task and score contracts. Private ground truth never enters prompts."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VERSION = "0.1.0"

SANITY_FAMILIES = frozenset({"read_numeric", "read_visual", "repair_fault", "repair_shortcut",
                             "write_piecewise", "write_controller"})

@dataclass(frozen=True)
class Limits:
    max_turns: int = 16
    python_calls: int = 12
    output_tokens: int = 32_000
    input_tokens: int = 160_000
    cpu_seconds: int = 120
    memory_mb: int = 2048
    wall_seconds: int = 900
    practice_examples: int = 128
    practice_episodes: int = 16
    episode_steps: int = 256

@dataclass
class Task:
    id: str
    parent_id: str
    split: str
    skill: str
    family: str
    architecture: dict[str, Any]
    instruction: str
    public: dict[str, Any]
    directory: Path
    limits: Limits = field(default_factory=Limits)

    @property
    def weights_path(self) -> Path:
        return self.directory / "target.npz"

    @property
    def private_path(self) -> Path:
        return self.directory / "private.json"

    @property
    def practice_path(self) -> Path:
        return self.directory / "practice.npz"

    def to_public_dict(self) -> dict[str, Any]:
        return {"version": VERSION, "id": self.id, "parent_id": self.parent_id,
                "split": self.split, "skill": self.skill, "family": self.family,
                "architecture": self.architecture, "instruction": self.instruction,
                "public": self.public, "limits": asdict(self.limits)}

    @classmethod
    def from_dict(cls, data: dict[str, Any], directory: Path) -> "Task":
        return cls(**{k: data[k] for k in ("id", "parent_id", "split", "skill", "family", "architecture", "instruction", "public")},
                   directory=directory, limits=Limits(**data.get("limits", {})))

@dataclass
class Score:
    success: bool
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "scored"
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
