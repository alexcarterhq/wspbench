"""Public-data controls with aggregate-only reporting for held-out tasks."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import numpy as np

from .domain import Task, VERSION
from .networks import load_weights
from .scoring import baseline_submission, reference_submission, score
from .tasks import load_tasks

FEATURE_NAMES = ["mean", "standard_deviation", "l2_norm", "positive_fraction", "q10", "median", "q90"]


def metadata_only_prediction(task: Task) -> str:
    """Fixed first-option guess. It inspects neither tensors nor answer keys."""
    if task.skill != "read":
        raise ValueError("The metadata-only label baseline applies only to Read")
    return task.public["options"][0]


def tensor_statistics(task: Task) -> np.ndarray:
    """Stable features, with the same tensor order for each fixed architecture."""
    features = []
    for name, array in sorted(load_weights(task.weights_path).items()):
        values = np.asarray(array, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"Cannot compute statistics for tensor {name}")
        features.extend([float(np.mean(values)), float(np.std(values)), float(np.linalg.norm(values)),
                         float(np.mean(values > 0)), *np.quantile(values, [.1, .5, .9]).tolist()])
    return np.asarray(features, dtype=np.float64)


class WeightStatisticsBaseline:
    """Nearest public-development checkpoint in standardized statistic space.

    This is intentionally a weak control. The pilot has only two development
    examples per family, covering only two of the four Read labels.
    """

    def __init__(self, development_tasks: list[Task]):
        if any(task.split != "dev" for task in development_tasks):
            raise ValueError("Statistics baseline may fit only public development tasks")
        self.families = {}
        self.training_summary = {}
        for family in sorted({task.family for task in development_tasks if task.skill == "read"}):
            tasks = sorted([task for task in development_tasks if task.skill == "read" and task.family == family], key=lambda t: t.id)
            features = np.stack([tensor_statistics(task) for task in tasks])
            labels = [reference_submission(task) for task in tasks]
            if any(not isinstance(label, str) or label not in task.public["options"] for label, task in zip(labels, tasks)):
                raise ValueError("Public development Read labels are invalid")
            mean = features.mean(axis=0)
            scale = features.std(axis=0)
            scale = np.where(scale > 1e-8, scale, 1.0)
            self.families[family] = {"features": (features - mean) / scale,
                                     "mean": mean, "scale": scale, "labels": labels}
            self.training_summary[family] = {"development_examples": len(tasks),
                                             "observed_label_count": len(set(labels)),
                                             "available_label_count": len(tasks[0].public["options"]),
                                             "feature_count": int(features.shape[1])}

    def predict(self, task: Task) -> str:
        if task.skill != "read" or task.family not in self.families:
            raise ValueError("No public-development statistic baseline for this Read family")
        fitted = self.families[task.family]
        features = tensor_statistics(task)
        if features.shape != fitted["mean"].shape:
            raise ValueError("Checkpoint architecture differs from its development family")
        standardized = (features - fitted["mean"]) / fitted["scale"]
        distances = np.sum((fitted["features"] - standardized) ** 2, axis=1)
        return fitted["labels"][int(np.argmin(distances))]


def _artifact_hash(submission: dict) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(submission.items()):
        digest.update(name.encode())
        digest.update(np.asarray(array, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _aggregate(records: list[dict]) -> dict:
    results = {}
    for baseline in sorted({row["baseline"] for row in records}):
        rows = [row for row in records if row["baseline"] == baseline]
        metrics = sorted({key for row in rows for key in row["score"]["metrics"]})
        results[baseline] = {
            "task_count": len(rows),
            "successes": sum(row["score"]["success"] for row in rows),
            "success_rate": float(np.mean([row["score"]["success"] for row in rows])),
            "mean_metrics": {key: float(np.mean([row["score"]["metrics"][key] for row in rows if key in row["score"]["metrics"]])) for key in metrics},
            "statuses": dict(Counter(row["score"]["status"] for row in rows)),
        }
    return results


def evaluate_baselines(root: str | Path) -> dict:
    """Evaluate controls, retaining held-out predictions only in a private file.

    Public returns contain split/skill/family aggregates and development fit
    metadata. They never contain official task IDs, predictions, or answers.
    """
    root = Path(root)
    tasks = load_tasks(root, "all")
    statistics = WeightStatisticsBaseline([task for task in tasks if task.split == "dev" and task.skill == "read"])
    records = []
    for task in tasks:
        methods = ["noop", "random"]
        if task.skill == "read":
            methods += ["metadata_only", "weight_statistics"]
        elif task.skill == "repair":
            methods += ["no_weight"]
        else:
            methods += ["constructive"]
        for method in methods:
            if method == "metadata_only":
                prediction = metadata_only_prediction(task)
            elif method == "weight_statistics":
                prediction = statistics.predict(task)
            elif method == "constructive":
                prediction = baseline_submission(task, "no_weight")
            else:
                prediction = baseline_submission(task, method)
            outcome = score(task, prediction)
            records.append({"task_id": task.id, "parent_id": task.parent_id, "split": task.split,
                            "skill": task.skill, "family": task.family, "baseline": method,
                            "prediction": prediction if isinstance(prediction, str) else {"artifact_sha256": _artifact_hash(prediction)},
                            "score": outcome.to_dict()})
    results = {}
    for split in sorted({task.split for task in tasks}):
        results[split] = {}
        for skill in sorted({task.skill for task in tasks if task.split == split}):
            selected = [row for row in records if row["split"] == split and row["skill"] == skill]
            results[split][skill] = {"task_count": len({row["task_id"] for row in selected}),
                                     "baselines": _aggregate(selected),
                                     "families": {family: {"task_count": len({row["task_id"] for row in selected if row["family"] == family}),
                                                           "baselines": _aggregate([row for row in selected if row["family"] == family])}
                                                  for family in sorted({row["family"] for row in selected})}}
    suite = json.loads((root / "suite.json").read_text())
    summary = {
        "version": VERSION, "suite_fingerprint": suite["fingerprint"], "status": "local_controls",
        "results": results,
        "weight_statistics": {"method": "1-nearest-neighbor with development-only feature standardization",
                              "fit_split": "dev", "features_per_tensor": FEATURE_NAMES,
                              "training": statistics.training_summary,
                              "limitation": "Only two public development examples per pilot Read family; only two of four labels observed. No held-out labels are used to fit or predict."},
        "metadata_only": {"method": "Always choose the first listed option without reading tensors.",
                          "balanced_four_choice_chance_rate": .25},
        "notes": ["One deterministic attempt per baseline and task; random baseline uses a fixed task-ID seed.",
                  "Results are controls, not provider model evaluations. No combined skill score is reported.",
                  "Constructive Write is the public-specification reference construction.",
                  "Per-task held-out predictions and results remain in baseline-details.private.json until retirement."],
    }
    private_path = root / "baseline-details.private.json"
    private_path.write_text(json.dumps({"version": VERSION, "records": records}, indent=2) + "\n")
    private_path.chmod(0o600)
    (root / "baseline-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
