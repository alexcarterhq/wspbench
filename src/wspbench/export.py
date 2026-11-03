"""Allowlisted public export. Official task files are never copied."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from wspbench.networks import save_weights
from wspbench.scoring import reference_submission
from wspbench.tasks import load_tasks


def export_development(data_root: Path, destination: Path) -> dict:
    tasks = load_tasks(data_root, "dev")
    if len(tasks) != 12:
        raise ValueError("Public release requires exactly twelve development examples")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Choose an empty destination; refusing to merge a public export")
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for task in tasks:
        folder = destination / task.id
        folder.mkdir()
        manifest = task.to_public_dict()
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        shutil.copyfile(task.directory / "target.npz", folder / "target.npz")
        shutil.copyfile(task.directory / "practice.npz", folder / "practice.npz")
        solution = reference_submission(task)
        if isinstance(solution, str):
            (folder / "reference.json").write_text(json.dumps({"answer": solution}) + "\n")
        else:
            save_weights(folder / "reference.npz", solution)
        entries.append({"id": task.id, "skill": task.skill, "family": task.family})
    (destination / "index.json").write_text(json.dumps({"license": "CC-BY-4.0", "tasks": entries}, indent=2) + "\n")
    archive = destination.parent / "wspbench-development-v0.1.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(destination.rglob("*")):
            if path.is_file() and not path.name.startswith("._"):
                output.write(path, path.relative_to(destination))
    return {"n_tasks": len(tasks), "archive": str(archive), "destination": str(destination)}
