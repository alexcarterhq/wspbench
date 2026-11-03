"""Fresh, locally generated targets. Only development cases may be exported."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import secrets
import numpy as np
from .domain import Task, Limits, VERSION
from .networks import forward, load_weights, save_weights, mlp_architecture, rnn_architecture, zeros, validate_weights

FAMILIES = {"read_numeric": "read", "read_visual": "read", "repair_fault": "repair",
            "repair_shortcut": "repair", "write_piecewise": "write", "write_controller": "write"}
NUMERIC_LABELS = ["mean", "maximum", "minimum", "last"]
VISUAL_LABELS = ["horizontal_bar", "vertical_bar", "main_diagonal", "anti_diagonal"]


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def numeric_truth(x: np.ndarray, label: str) -> np.ndarray:
    functions = {"mean": lambda a: a.mean(axis=1), "maximum": lambda a: a.max(axis=1),
                 "minimum": lambda a: a.min(axis=1), "last": lambda a: a[:, -1]}
    return functions[label](x).reshape(-1, 1)


def visual_patterns() -> np.ndarray:
    patterns = np.zeros((4, 8, 8), dtype=np.float32)
    patterns[0, 3, :] = 1
    patterns[1, :, 3] = 1
    patterns[2, np.arange(8), np.arange(8)] = 1
    patterns[3, np.arange(8), 7 - np.arange(8)] = 1
    return patterns.reshape(4, 64)


def visual_cases(rng: np.random.Generator, count: int, label: int, nuisance: bool = False,
                 correlation: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    y = np.tile([0, 1], (count + 1) // 2)[:count]
    rng.shuffle(y)
    classes = rng.integers(0, 3, size=count)
    classes += classes >= label
    classes[y == 1] = label
    x = visual_patterns()[classes] + rng.normal(0, .08, (count, 64))
    if nuisance:
        stamp = y.copy()
        flip = rng.random(count) >= correlation
        stamp[flip] = 1 - stamp[flip]
        x = np.column_stack([x, (2 * stamp - 1) * 5.0])
    return x.astype(np.float32), y.astype(np.float32).reshape(-1, 1)


def _train_mlp(seed: int, arch: dict, label: int, nuisance: bool = False,
               correlation: float = .5, steps: int = 220) -> dict[str, np.ndarray]:
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    layers = []
    for i, (nin, nout) in enumerate(zip(arch["sizes"][:-1], arch["sizes"][1:])):
        layers.append(torch.nn.Linear(nin, nout))
        if i < len(arch["sizes"]) - 2:
            layers.append(torch.nn.ReLU())
    net = torch.nn.Sequential(*layers)
    optimizer = torch.optim.Adam(net.parameters(), lr=.012)
    linear_layers = [m for m in net if isinstance(m, torch.nn.Linear)]
    auxiliary = arch["sizes"][-1] == 2
    teacher = rng.normal(0, .3, 64).astype(np.float32)
    if auxiliary:
        with torch.no_grad():
            linear_layers[0].weight[:4] = 0
            linear_layers[0].bias[:4] = 0
            linear_layers[-1].weight[:, :4] = 0
    for _ in range(steps):
        x, y = visual_cases(rng, 128, label, nuisance, correlation)
        predictions = net(torch.from_numpy(x))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions[:, :1], torch.from_numpy(y))
        if auxiliary:
            aux_y = torch.from_numpy((x[:, :64] @ teacher).reshape(-1, 1))
            loss = loss + .5 * ((predictions[:, 1:2] - aux_y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    values = {}
    for i, layer in enumerate(m for m in net if isinstance(m, torch.nn.Linear)):
        values[f"w{i}"] = layer.weight.detach().numpy().copy()
        values[f"b{i}"] = layer.bias.detach().numpy().copy()
    return values


def _train_rnn(seed: int, arch: dict, label: str, steps: int = 850) -> dict[str, np.ndarray]:
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    hidden = arch["hidden"]
    wi = torch.nn.Parameter(torch.randn(hidden, 1) * .2)
    wh = torch.nn.Parameter(torch.randn(hidden, hidden) * (.3 / np.sqrt(hidden)))
    b = torch.nn.Parameter(torch.zeros(hidden))
    wo = torch.nn.Parameter(torch.randn(1, hidden) * .2)
    bo = torch.nn.Parameter(torch.zeros(1))
    params = [wi, wh, b, wo, bo]
    optimizer = torch.optim.Adam(params, lr=.008)
    for _ in range(steps):
        x_np = rng.uniform(-1, 1, (128, arch["length"])).astype(np.float32)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(numeric_truth(x_np, label).astype(np.float32))
        h = torch.zeros((len(x), hidden))
        for t in range(arch["length"]):
            h = torch.tanh(x[:, t:t + 1] @ wi.T + h @ wh.T + b)
        loss = ((h @ wo.T + bo - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {k: v.detach().numpy().copy() for k, v in zip(arch["shapes"], params)}


def _visual_reference(arch: dict, label: int, base: dict | None = None) -> dict[str, np.ndarray]:
    weights = zeros(arch) if base is None else {k: v.copy() for k, v in base.items()}
    weights["w0"][0] = 0
    weights["w1"][0] = 0
    pattern = visual_patterns()[label]
    # Every positive motif has eight active pixels; all other motifs overlap at <=2.
    weights["w0"][0, :64] = pattern
    weights["b0"][0] = -3.0
    weights["w1"][0, 0] = 2
    weights["b1"][0] = -2
    return weights


def _numeric_case(seed: int, index: int) -> tuple[dict, dict, dict, dict, dict]:
    arch = rnn_architecture(24, 5)
    label = NUMERIC_LABELS[index % 4]
    weights = _train_rnn(seed, arch, label, steps=2400)
    rng = np.random.default_rng(seed + 1)
    x = rng.uniform(-1, 1, (2048, 5)).astype(np.float32)
    predictions = forward(arch, weights, x)
    errors = {name: float(np.mean((predictions - numeric_truth(x, name)) ** 2)) for name in NUMERIC_LABELS}
    correct = errors[label]
    alternative = min(v for k, v in errors.items() if k != label)
    if correct > .006 or alternative < correct * 2 + .01:
        raise ValueError(f"RNN failed behavior admission: {label}, errors={errors}")
    public = {"options": NUMERIC_LABELS, "input_semantics": "Five real values in [-1,1], in sequence order. The scalar output approximates one listed statistic.", "training": "Fresh supervised training; no optimizer or training allowed during solving."}
    private = {"answer": label, "admission": {"mse": correct, "alternative_mse": alternative, "training_steps": 2400}}
    return arch, weights, weights, public, {"private": private, "hidden": {"x": x}, "practice": {}}


def _visual_case(seed: int, index: int, family: str, split: str,
                 clean_control: bool = False, fault_sign: int = 1) -> tuple[dict, dict, dict, dict, dict]:
    nuisance = family == "repair_shortcut"
    repairing = family.startswith("repair")
    arch = mlp_architecture([65 if nuisance else 64, 24 if repairing else 16, 2 if repairing else 1])
    label = index % 4
    correlation = .5 if clean_control else 1.0
    steps = 320 if repairing else 220
    weights = _train_mlp(seed, arch, label, nuisance, correlation if nuisance else .5, steps)
    reference = _visual_reference(arch, label, weights) if nuisance else {k: v.copy() for k, v in weights.items()}
    rng = np.random.default_rng(seed + 1)
    x, y = visual_cases(rng, 2048, label, nuisance, .0 if nuisance else .5)
    clean_x, clean_y = visual_cases(rng, 1024, label, nuisance, 1.0 if nuisance else .5)
    practice_x, practice_y = visual_cases(rng, 128, label, nuisance, .0 if nuisance else .5)
    target_accuracy = float(np.mean((forward(arch, weights, x)[:, :1] >= 0) == y))
    if family in {"read_visual", "repair_fault"} and target_accuracy < .98:
        raise ValueError("Trained visual target failed behavior admission")
    if family == "repair_fault" and not clean_control:
        weights["b1"][0] += 32.0 * fault_sign
    if nuisance and not clean_control and target_accuracy > .8:
        raise ValueError("Shortcut target did not acquire a verified nuisance dependence")
    if clean_control:
        reference = {k: v.copy() for k, v in weights.items()}
    if repairing and float(np.std(forward(arch, weights, clean_x)[:, 1])) < .1:
        raise ValueError("Learned auxiliary behavior is degenerate")
    public = {"input_semantics": "8x8 grayscale pixels flattened row-major" + (" followed by one nuisance stamp value, which the motif output must ignore." if nuisance else "."),
              "patterns": {name: pattern.reshape(8, 8).astype(int).tolist() for name, pattern in zip(VISUAL_LABELS, visual_patterns())},
              "output_semantics": "Output 0 is a motif logit: nonnegative predicts presence." + (" Output 1 is an unrelated learned real-valued auxiliary function. Preserve its original behavior; its formula is intentionally unspecified." if repairing else "")}
    if family == "read_visual":
        public["options"] = VISUAL_LABELS
        matches = {}
        for candidate in range(4):
            cx, cy = visual_cases(rng, 512, candidate)
            matches[VISUAL_LABELS[candidate]] = float(np.mean((forward(arch, weights, cx) >= 0) == cy))
        if matches[VISUAL_LABELS[label]] < .98 or max(v for k, v in matches.items() if k != VISUAL_LABELS[label]) > .85:
            raise ValueError("Visual target has ambiguous behavioral identity")
        private = {"answer": VISUAL_LABELS[label], "admission": {"hypothesis_accuracy": matches, "training_steps": steps}}
    else:
        public["target_motif"] = VISUAL_LABELS[label]
        public["success_criteria"] = {"target_accuracy": .95, "max_retention_loss": .02,
                                      "auxiliary_absolute_tolerance": .001, "auxiliary_retention_accuracy": .98}
        private = {"clean_control": clean_control, "admission": {"training_steps": steps,
                   "auxiliary_standard_deviation": float(np.std(forward(arch, weights, clean_x)[:, 1]))}}
    return arch, weights, reference, public, {"private": private,
        "hidden": {"x": x, "y": y, "preserve_x": clean_x, "preserve_y": clean_y},
        "practice": {"x": practice_x, "y": practice_y}}


def _piecewise_case(seed: int) -> tuple[dict, dict, dict, dict, dict]:
    arch = mlp_architecture([1, 8, 1])
    rng = np.random.default_rng(seed)
    knots = np.linspace(0, 1, 6)
    values = rng.uniform(.1, .9, 6)
    slopes = np.diff(values) / np.diff(knots)
    reference = zeros(arch)
    reference["w0"][:5, 0] = 1
    reference["b0"][:5] = -knots[:-1]
    reference["w1"][0, :5] = np.r_[slopes[0], np.diff(slopes)]
    reference["b1"][0] = values[0]
    x = np.unique(np.r_[np.linspace(0, 1, 4097), rng.uniform(0, 1, 1024), knots,
                         np.clip(knots - 1e-6, 0, 1), np.clip(knots + 1e-6, 0, 1)]).reshape(-1, 1)
    px = rng.uniform(0, 1, (128, 1))
    public = {"knots": knots.tolist(), "values": values.tolist(), "input_domain": [0, 1],
              "output_semantics": "Linearly interpolate the specified knot/value pairs. No extrapolation is scored.",
              "success_criteria": {"max_absolute_error": .001}}
    return arch, zeros(arch), reference, public, {"private": {},
        "hidden": {"x": x, "y": np.interp(x[:, 0], knots, values).reshape(-1, 1)},
        "practice": {"x": px, "y": np.interp(px[:, 0], knots, values).reshape(-1, 1)}}


def _controller_case(seed: int) -> tuple[dict, dict, dict, dict, dict]:
    arch = mlp_architecture([3, 4, 1])
    rng = np.random.default_rng(seed)
    mass = float(rng.uniform(.8, 1.2))
    drag = float(rng.uniform(.05, .2))
    kp, kd = 4 * mass, 3 * mass
    reference = zeros(arch)
    reference["w0"][:2] = np.array([-kp, -kd, kp])
    reference["b0"][:2] = [1, -1]
    reference["w1"][0, :2] = [1, -1]
    reference["b1"][0] = -1
    initial = np.column_stack([rng.uniform(-1, 1, 256), rng.uniform(-.3, .3, 256), rng.uniform(-.6, .6, 256)])
    practice = np.column_stack([rng.uniform(-1, 1, 16), rng.uniform(-.3, .3, 16), rng.uniform(-.6, .6, 16)])
    dynamics = {"mass": mass, "drag": drag, "dt": .05, "steps": 256,
                "position_tolerance": .04, "velocity_tolerance": .04}
    public = {"input_semantics": "[position, velocity, goal_position]", "output_semantics": "Force, clipped by the evaluator to [-1,1].", "dynamics": dynamics,
              "initial_conditions": {"position": [-1, 1], "velocity": [-.3, .3], "goal": [-.6, .6]},
              "update_rule": "v <- v + dt * (force/mass - drag*v); x <- x + dt*v. Goal remains fixed.",
              "success_criteria": {"episode_success_rate": .95}}
    return arch, zeros(arch), reference, public, {"private": {}, "hidden": {"initial": initial}, "practice": {"initial": practice}}


def _suite_fingerprint(root: Path, entries: list[dict]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        directory = root / entry["split"] / entry["id"]
        if not directory.resolve().is_relative_to(root.resolve()):
            raise ValueError("Task directory leaves the suite root")
        for filename in ("manifest.json", "target.npz", "reference.npz", "private.json", "private.npz", "practice.npz"):
            digest.update((directory / filename).read_bytes())
    return digest.hexdigest()


def generate_suite(output: str | Path, official_per_family: int = 8, dev_per_family: int = 2,
                   seed: int | None = None) -> dict:
    output = Path(output)
    if official_per_family < 1 or dev_per_family < 1:
        raise ValueError("Both splits must contain at least one task per family")
    if (output / "suite.json").exists():
        raise ValueError("Suite already exists; use a new directory to preserve frozen artifacts")
    output.mkdir(parents=True, exist_ok=True)
    seed = secrets.randbits(63) if seed is None else int(seed)
    _json(output / "private-suite.json", {"seed": seed, "version": VERSION})
    entries = []
    for family, skill in FAMILIES.items():
        for split, count in [("dev", dev_per_family), ("official", official_per_family)]:
            base = 17011 if split == "dev" else seed
            schedule_seed = int.from_bytes(hashlib.sha256(f"{base}:{family}:{split}:schedule".encode()).digest()[:8], "little")
            schedule_rng = np.random.default_rng(schedule_seed)
            labels = schedule_rng.permutation(np.arange(count) % 4)
            controls = np.zeros(count, dtype=bool)
            number_clean = count // 4 if split == "official" else count // 2
            # Prefer distinct labels for controls, and never derive status from a label.
            distinct = schedule_rng.permutation(np.unique(labels))
            chosen = []
            for motif in distinct:
                if len(chosen) == number_clean:
                    break
                chosen.append(int(schedule_rng.choice(np.flatnonzero(labels == motif))))
            if len(chosen) < number_clean:
                remaining = [i for i in range(count) if i not in chosen]
                chosen.extend(schedule_rng.choice(remaining, number_clean - len(chosen), replace=False).tolist())
            controls[chosen] = True
            signs = schedule_rng.permutation(np.where(np.arange(count) % 2, -1, 1))
            for index in range(count):
                task_seed = int.from_bytes(hashlib.sha256(f"{base}:{family}:{index}".encode()).digest()[:8], "little") & ((1 << 63) - 1)
                task_id = hashlib.sha256(f"{split}:{task_seed}:{family}".encode()).hexdigest()[:32]
                directory = output / split / task_id
                directory.mkdir(parents=True, exist_ok=False)
                if family == "read_numeric":
                    arch, target, reference, public, payload = _numeric_case(task_seed, int(labels[index]))
                elif family in {"read_visual", "repair_fault", "repair_shortcut"}:
                    arch, target, reference, public, payload = _visual_case(task_seed, int(labels[index]), family, split, bool(controls[index]), int(signs[index]))
                elif family == "write_piecewise":
                    arch, target, reference, public, payload = _piecewise_case(task_seed)
                else:
                    arch, target, reference, public, payload = _controller_case(task_seed)
                instruction = {
                    "read_numeric": "Identify the statistic computed by this freshly trained recurrent network. Analyze weights only; do not execute any layer on real or synthetic input. Return exactly one listed option.",
                    "read_visual": "Identify the motif recognized by this freshly trained classifier. Analyze weights only; do not execute any layer on real or synthetic input. Return exactly one listed option.",
                    "repair_fault": "Repair any parameter fault so the network recognizes the specified motif. It may already be correct. Preserve performance on previously successful motif cases and preserve the unrelated learned auxiliary output. Submit tensors only.",
                    "repair_shortcut": "Remove any dependence on the nuisance stamp while retaining recognition of the specified motif. It may already be correct. Preserve the unrelated learned auxiliary output. Submit tensors only.",
                    "write_piecewise": "Write weights implementing the specified piecewise-linear function in this fixed zero-initialized ReLU network. Submit tensors only.",
                    "write_controller": "Write weights for a controller that reaches the goal and stops within 256 steps in this fixed zero-initialized ReLU network. Submit tensors only."
                }[family]
                task = Task(task_id, task_id, split, skill, family, arch, instruction, public, directory)
                validate_weights(arch, target)
                save_weights(directory / "target.npz", target)
                save_weights(directory / "reference.npz", reference)
                _json(directory / "manifest.json", task.to_public_dict())
                _json(directory / "private.json", {**payload["private"], "generation_seed": task_seed})
                np.savez(directory / "private.npz", **payload["hidden"])
                np.savez(directory / "practice.npz", **payload["practice"])
                entries.append({"id": task_id, "parent_id": task_id, "split": split, "skill": skill,
                                "family": family, "manifest": f"{split}/{task_id}/manifest.json"})
    summary = {"version": VERSION, "status": "generated", "fingerprint": _suite_fingerprint(output, entries),
               "counts": {"official": 6 * official_per_family, "dev": 6 * dev_per_family},
               "families": FAMILIES, "tasks": entries,
               "notice": "Held-out local pilot. Ground truth and official artifacts must not be published until retirement."}
    _json(output / "suite.json", summary)
    return summary


def load_tasks(root: str | Path, split: str = "dev") -> list[Task]:
    root = Path(root)
    if split not in {"dev", "official", "all"}:
        raise ValueError("Split must be dev, official, or all")
    suite = json.loads((root / "suite.json").read_text())
    tasks = []
    for entry in suite["tasks"]:
        if split != "all" and entry["split"] != split:
            continue
        path = root / entry["manifest"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Manifest path leaves the suite directory")
        tasks.append(Task.from_dict(json.loads(path.read_text()), path.parent))
    return tasks


def validate_suite(root: str | Path) -> dict:
    from .scoring import score, reference_submission, baseline_submission
    root = Path(root)
    suite = json.loads((root / "suite.json").read_text())
    if _suite_fingerprint(root, suite["tasks"]) != suite["fingerprint"]:
        raise ValueError("Suite fingerprint mismatch: frozen artifacts were modified")
    tasks = load_tasks(root, "all")
    results = []
    parents = set()
    for task in tasks:
        if task.parent_id in parents:
            raise ValueError("Duplicate parent task in primary suite")
        parents.add(task.parent_id)
        validate_weights(task.architecture, load_weights(task.weights_path))
        reference = score(task, reference_submission(task))
        baseline = score(task, baseline_submission(task, "noop"))
        random = score(task, baseline_submission(task, "random"))
        no_weight = score(task, baseline_submission(task, "no_weight"))
        results.append({"id": task.id, "split": task.split, "skill": task.skill,
                        "family": task.family, "reference": reference.to_dict(),
                        "noop": baseline.to_dict(), "random": random.to_dict(), "no_weight": no_weight.to_dict()})
    passed = all(item["reference"]["success"] for item in results)
    result = {"version": VERSION, "status": "passed" if passed else "failed", "task_count": len(tasks),
              "reference_successes": sum(item["reference"]["success"] for item in results), "tasks": results}
    _json(Path(root) / "validation.json", result)
    return result

