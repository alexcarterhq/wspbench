from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from wspbench.audit import audit_trace, review_run, screen_code
from wspbench.budget import BudgetExceeded, BudgetLedger, maximum_session_cost, normalize_usage, empty_usage, add_usage, session_cost
from wspbench.domain import Limits, Task
from wspbench.networks import mlp_architecture, save_weights, zeros
from wspbench.runner import preflight, suite_digest
from wspbench.sandbox import DockerWorkspace, WorkspaceFailure


def test_budget_parallel_reservations_and_crash_safety(tmp_path):
    path = tmp_path / "ledger.sqlite"
    ledger = BudgetLedger(path, total=1, phases={"calibration": 1})
    def reserve(i):
        try:
            return ledger.reserve("calibration", .2, str(i))
        except BudgetExceeded:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = [r for r in pool.map(reserve, range(20)) if r is not None]
    assert len(ids) == 5
    assert ledger.snapshot()["committed_usd"] == 1
    restarted = BudgetLedger(path, total=1, phases={"calibration": 1})
    assert len(restarted.snapshot()["unsettled_reservations"]) == 5
    restarted.settle(ids[0], .1)
    restarted.settle(ids[0], .1)  # Idempotent reconciliation.
    with pytest.raises(ValueError):
        restarted.settle(ids[0], 0)
    assert restarted.snapshot()["committed_usd"] == .9


def test_actual_overrun_is_recorded_and_reported(tmp_path):
    ledger = BudgetLedger(tmp_path / "b.sqlite", total=1, phases={"primary": 1})
    identity = ledger.reserve("primary", .1)
    with pytest.raises(BudgetExceeded):
        ledger.settle(identity, 1.1)
    assert ledger.snapshot()["spent_usd"] == 1.1
    with pytest.raises(BudgetExceeded):
        ledger.reserve("primary", .01)


def test_trace_audit_requires_review_and_detects_prohibited_ops(tmp_path):
    trace = [{"tool": "python", "code": "for i in range(3):\n    weights['w0'][i] = 0"}]
    audit = audit_trace(trace)
    assert audit["status"] == "pending"
    assert "prohibited_attribute:backward" in screen_code("loss.backward()")
    assert audit_trace([{"tool": "python", "code": "import socket"}])["status"] == "rejected"
    (tmp_path / "trace.json").write_text(json.dumps(trace))
    (tmp_path / "result.json").write_text(json.dumps({"audit": audit, "success": True}))
    reviewed = review_run(tmp_path, "independent-reviewer", "approved", "Analytic construction; no target execution or fitting")
    assert reviewed["audit"]["status"] == "approved"
    (tmp_path / "trace.json").write_text("[]")
    with pytest.raises(ValueError, match="Trace changed"):
        review_run(tmp_path, "reviewer", "approved", "Must fail")


def test_price_confirmation_required():
    with pytest.raises(ValueError):
        maximum_session_cost({"id": "example", "pricing": {}}, Limits())
    config = {"id": "example", "pricing": {"verified_at": "2026-11-02", "source": "https://example.test/prices", "input_per_million": 10, "cache_write_per_million": 12.5, "cache_read_per_million": 1, "output_per_million": 50}}
    assert maximum_session_cost(config, Limits()) == 3.6


def test_actual_cached_token_fixture_is_counted_and_priced_once():
    model = {"id": "fixture", "pricing": {"verified_at": "2026-11-02", "source": "https://developers.openai.com/api/docs/models/gpt-6-astra", "input_per_million": 10, "cache_write_per_million": 12.5, "cache_read_per_million": 1, "output_per_million": 50}}
    observed = {"input_tokens": 3, "input_tokens_cache_write": 1198, "input_tokens_cache_read": 0,
                "output_tokens": 13, "reasoning_tokens": 0, "total_tokens": 1214}
    counts = normalize_usage(observed)
    assert counts["input_tokens_total"] == 1201
    assert session_cost(model, counts) == pytest.approx(.015655)
    second = {"input_tokens": 3, "input_tokens_cache_write": 1198, "input_tokens_cache_read": 500,
              "output_tokens": 50, "reasoning_tokens": 20, "total_tokens": 1751}
    aggregate = add_usage(add_usage(empty_usage(), observed), second)
    assert aggregate["input_tokens_total"] == 2902
    assert aggregate["output_tokens"] == 63  # Reasoning is a subset, not an extra charge.
    assert aggregate["reasoning_tokens"] == 20
    assert session_cost(model, aggregate) == pytest.approx(.015655 + .018005)
    with pytest.raises(ValueError, match="reconcile"):
        normalize_usage({**observed, "total_tokens": 1215})


def test_explicit_billing_correction_preserves_original_charge(tmp_path):
    ledger = BudgetLedger(tmp_path / "bill.sqlite", total=10, phases={"calibration": 10})
    identity = ledger.reserve("calibration", 3.2)
    ledger.settle(identity, .005)
    evidence = "a" * 64
    change = ledger.correct_cost(identity, .025, reason="Cached usage omitted in original accounting", evidence_sha256=evidence, reviewer="automated-trace-reconciliation")
    assert change["old_actual_usd"] == .005
    assert ledger.snapshot()["spent_usd"] == .025
    assert ledger.snapshot()["corrections"][0]["old_actual_usd"] == .005
    assert ledger.correct_cost(identity, .025, reason="Same authoritative trace", evidence_sha256=evidence, reviewer="reviewer")["already_applied"]
    outstanding = ledger.reserve("calibration", 3.6)
    with pytest.raises(ValueError, match="unknown billing"):
        ledger.correct_cost(outstanding, 0, reason="No authoritative receipt", evidence_sha256="b" * 64, reviewer="reviewer")
    assert outstanding in ledger.snapshot()["unsettled_reservations"]


def test_failed_workspace_rpc_cannot_consume_a_late_reply():
    class Input:
        def write(self, data):
            pass
        async def drain(self):
            pass
    class Output:
        async def readline(self):
            await asyncio.sleep(5)
    class Process:
        returncode = None
        stdin = Input()
        stdout = Output()
    async def check():
        workspace = DockerWorkspace(Limits())
        workspace.process = Process()
        workspace.rpc_timeout = .001
        closed = []
        async def close():
            closed.append(True)
            workspace.process = None
        workspace.close = close
        with pytest.raises(WorkspaceFailure):
            await workspace._request({"op": "exec", "code": "pass"})
        assert closed and workspace.failed
        with pytest.raises(WorkspaceFailure, match="invalidated"):
            await workspace._request({"op": "weights"})
    asyncio.run(check())


def test_neuron_permutation_preserves_both_supported_architectures():
    from wspbench.runner import _permuted_weights
    from wspbench.networks import forward, rnn_architecture
    rng = np.random.default_rng(123)
    for arch in [mlp_architecture([3, 5, 4, 2]), rnn_architecture(6, 5)]:
        weights = {k: rng.normal(0, .2, s).astype(np.float32) for k, s in arch["shapes"].items()}
        sizes = [arch["hidden"]] if arch["kind"] == "rnn" else arch["sizes"][1:-1]
        transformed = _permuted_weights(arch, weights, [rng.permutation(s) for s in sizes])
        x = rng.uniform(-1, 1, (64, 5 if arch["kind"] == "rnn" else 3))
        assert np.allclose(forward(arch, weights, x), forward(arch, transformed, x), atol=1e-7)


def test_suite_digest_ignores_reports_but_freezes_declared_artifacts(tmp_path):
    root = tmp_path / "suite"
    directory = root / "dev" / "neutral-id"
    directory.mkdir(parents=True)
    (root / "private-suite.json").write_text('{"seed":123}')
    (root / "suite.json").write_text(json.dumps({"tasks": [{"manifest": "dev/neutral-id/manifest.json"}]}))
    for name in ("manifest.json", "target.npz", "reference.npz", "private.json", "private.npz", "practice.npz"):
        (directory / name).write_bytes(b"immutable test fixture")
    digest = suite_digest(root)
    (root / "baseline-details.private.json").write_text('{"report":"not a task"}')
    (root / "validation.json").write_text('{"status":"passed"}')
    assert suite_digest(root) == digest
    (directory / "target.npz").write_bytes(b"changed")
    assert suite_digest(root) != digest
    (root / "suite.json").write_text(json.dumps({"tasks": [{"manifest": "../outside/manifest.json"}]}))
    with pytest.raises(ValueError, match="leaves"):
        suite_digest(root)


def test_calibration_preflight_can_select_only_one_provider(tmp_path, monkeypatch):
    import wspbench.runner as runner
    config = json.loads(Path("configs/models.json").read_text())
    for model in config["models"]:
        monkeypatch.delenv(model["api_key_env"], raising=False)
    chosen = config["models"][0]
    chosen.update(snapshot="test-snapshot", verified_at="2026-11-02")
    chosen["pricing"].update(input_per_million=10, cache_write_per_million=12.5,
                             cache_read_per_million=1, output_per_million=50, verified_at="2026-11-02")
    path = tmp_path / "models.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    monkeypatch.setattr(runner, "docker_status", lambda: {"available": True, "image_ready": True, "image_digest": "sha256:test"})
    assert preflight(path, model_ids=[chosen["id"]])["ready"]
    assert not preflight(path)["ready"]


def test_inspect_scaffold_scores_mock_submission_without_claiming_real_results(tmp_path, monkeypatch):
    monkeypatch.setenv("INSPECT_TRACE_FILE", str(tmp_path / "inspect-trace.log"))
    monkeypatch.setenv("INSPECT_CACHE_DIR", str(tmp_path / "inspect-cache"))
    import wspbench.inspect_task as integration
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, ModelUsage, ChatMessageAssistant
    from inspect_ai.tool import ToolCall
    arch = mlp_architecture([1, 2, 1])
    save_weights(tmp_path / "target.npz", zeros(arch))
    (tmp_path / "private.json").write_text(json.dumps({"answer": "mean"}))
    task = Task("mock-task", "mock-task", "dev", "read", "read_numeric", arch,
                "Identify behavior", {"options": ["mean", "maximum"]}, tmp_path)
    class NumericFixture:
        """No arbitrary solver code executes in this testing fixture."""
        def __init__(self, limits, **kwargs):
            self.cpu_seconds = 0
            self.calls = 0
        async def start(self, weights):
            return self
        async def close(self):
            pass
    monkeypatch.setattr(integration, "DockerWorkspace", NumericFixture)
    record = {}
    evaluation = integration.build_inspect_task(task, tmp_path / "run", {"generate_config": {}}, record)
    output = ModelOutput(model="mockllm", usage=ModelUsage(input_tokens=3, input_tokens_cache_write=1198,
                         input_tokens_cache_read=0, output_tokens=13, total_tokens=1214),
                         choices=[{"message": ChatMessageAssistant(content="", tool_calls=[ToolCall(id="1", function="submit", arguments={"answer": "mean"})])}])
    logs = inspect_eval(evaluation, model="mockllm/model", model_args={"custom_outputs": [output]},
                        log_dir=str(tmp_path / "inspect"), log_realtime=False, ctl_server=False)
    assert logs[0].status == "success"
    assert record["status"] == "completed", record.get("error")
    assert record["success"] is True
    assert record["audit"]["status"] == "pending"
    assert record["usage"]["input_tokens"] == 3
    assert record["usage"]["input_tokens_total"] == 1201


def test_outer_inspect_cancellation_retains_unknown_billing(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import wspbench.inspect_task as integration
    arch = mlp_architecture([1, 2, 1])
    save_weights(tmp_path / "target.npz", zeros(arch))
    task = Task("cancel-fixture", "cancel-fixture", "dev", "read", "read_numeric", arch,
                "Identify behavior", {"options": ["mean", "maximum"]}, tmp_path)
    closed = []
    class FixtureWorkspace:
        def __init__(self, limits, **kwargs):
            self.cpu_seconds = 0
            self.calls = 0
        async def start(self, weights):
            pass
        async def close(self):
            closed.append(True)
    monkeypatch.setattr(integration, "DockerWorkspace", FixtureWorkspace)
    record = {}
    evaluation = integration.build_inspect_task(task, tmp_path / "run", {}, record)
    async def interrupted(*args, **kwargs):
        raise asyncio.CancelledError()
    state = SimpleNamespace(messages=[], metadata={}, completed=False)
    async def check():
        with pytest.raises(asyncio.CancelledError):
            await evaluation.solver(state, interrupted)
    asyncio.run(check())
    assert record["usage_unknown"] is True
    assert record["status"] == "timeout"
    assert closed
