"""
Mock benchmark test: validates the benchmark infrastructure without real models.

Uses mock implementations to verify:
- Method registry dispatch
- BenchmarkReport aggregation
- Per-method and per-organ statistics
- JSON serialization
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medsam_agent3d.benchmark import (
    BenchmarkEntry, MethodResult, BenchmarkReport, METHOD_REGISTRY,
)


def mock_method_good(entry, args) -> MethodResult:
    return MethodResult(
        method="Mock-Method-A",
        volume_id=entry.volume_id,
        target_name=entry.target_name,
        iou=0.85,
        dice=0.92,
        runtime_sec=1.2,
        num_turns=3,
        success=True,
    )


def mock_method_ok(entry, args) -> MethodResult:
    return MethodResult(
        method="Mock-Method-B",
        volume_id=entry.volume_id,
        target_name=entry.target_name,
        iou=0.65,
        dice=0.78,
        runtime_sec=0.5,
        num_turns=1,
        success=True,
    )


def test_benchmark():
    """Run mock benchmark and validate all infrastructure."""
    print("=" * 60)
    print("Benchmark Infrastructure Test")
    print("=" * 60)

    # Create synthetic entries
    organs = ["liver", "spleen", "kidney_right"]
    entries = []
    for organ in organs:
        for case in ["s0001", "s0002"]:
            entries.append(BenchmarkEntry(
                volume_path=f"data/{case}/ct.nii.gz",
                mask_path=f"data/{case}/segmentations/{organ}.nii.gz",
                target_name=organ,
                volume_id=case,
            ))

    print(f"\n[1/4] Created {len(entries)} evaluation cases")
    for e in entries:
        print(f"  {e.volume_id}/{e.target_name}")

    # Run mock methods
    print(f"\n[2/4] Running mock methods...")
    report = BenchmarkReport(
        methods=["Mock-Method-A", "Mock-Method-B"],
        total_cases=len(entries),
    )

    for entry in entries:
        report.add_result(mock_method_good(entry, None))
        report.add_result(mock_method_ok(entry, None))

    print(f"  Collected {len(report.results)} results")

    # Compute aggregate stats
    print(f"\n[3/4] Computing aggregate statistics...")
    report.compute()
    print(report.summary_table())

    # Validate per-method stats
    assert len(report.per_method) == 2
    for method in ["Mock-Method-A", "Mock-Method-B"]:
        assert method in report.per_method
        assert report.per_method[method]["num_cases"] == 6

    # Validate per-organ stats
    for organ in organs:
        assert organ in report.per_organ
        assert len(report.per_organ[organ]) == 2

    print("[4/4] Validating JSON serialization...")
    report.to_json("outputs/tests/benchmark_test.json")
    assert os.path.exists("outputs/tests/benchmark_test.json")
    print("  [OK] JSON saved")

    # Verify METHOD_REGISTRY is accessible
    print(f"\n  Available real methods: {list(METHOD_REGISTRY.keys())}")

    print("\n" + "=" * 60)
    print("*** BENCHMARK INFRASTRUCTURE OK ***")
    print("=" * 60)


if __name__ == "__main__":
    os.makedirs("outputs/tests", exist_ok=True)
    test_benchmark()


def test_failures_and_matched_cases(tmp_path):
    report = BenchmarkReport(methods=["agent", "sam-point"], total_cases=2)
    report.results = [
        MethodResult("agent", "a", "liver", iou=.8, dice=.9, success=True),
        MethodResult("agent", "b", "liver", iou=.4, dice=.5, success=False, error="limit"),
        MethodResult("sam-point", "a", "liver", iou=.6, dice=.7, success=True),
        MethodResult("sam-point", "b", "liver", iou=.9, dice=.95, success=True),
    ]
    report.compute()
    assert report.per_method["agent"]["mean_iou"] == .4
    assert report.per_method["agent"]["num_failed"] == 1
    assert report.per_method["agent"]["failure_rate"] == .5
    assert report.per_method["agent"]["matched_mean_iou"] == .8
    assert report.per_method["sam-point"]["matched_mean_iou"] == .6
    assert "agent" in report.summary_table()
    report.to_json(str(tmp_path / "report.json"))
    import json
    payload = json.loads((tmp_path / "report.json").read_text())
    assert len(payload["results"]) == 4
    assert payload["results"][1]["error"] == "limit"


def test_batch_ct_path_optional_gt_and_persistence(tmp_path, monkeypatch):
    from medsam_agent3d import evaluation
    from medsam_agent3d.types import AgentResult, TurnRecord
    from medsam_agent3d import io as io_utils
    seen = {}
    volume = np.zeros((3, 4, 5), dtype=np.float32)
    def fake_load(path, **kwargs):
        assert path == "scan.nii.gz"
        return volume, (2., 3., 4.), np.eye(4)
    def fake_loop(**kwargs):
        seen.update(kwargs)
        return AgentResult(volume_id="case", target_name="liver", success=True,
            predicted_mask=np.ones_like(volume), turns=[TurnRecord(0, "view.png",
                {"tool": "stop_action"}, "raw response", None, None, 60, (1, 2, 2))])
    def fake_save(data, path, **kwargs):
        assert kwargs["reference_path"] == "scan.nii.gz"
        assert data.shape == volume.shape
        seen["saved"] = path
    monkeypatch.setattr(io_utils, "load_nifti", fake_load)
    monkeypatch.setattr(io_utils, "save_nifti", fake_save)
    monkeypatch.setattr(evaluation, "run_agent_loop", fake_loop)
    results = evaluation.evaluate_dataset(None, None,
        [{"ct_path": "scan.nii.gz", "volume_id": "case", "target_name": "liver"}],
        str(tmp_path), seed=19)
    assert seen["volume"] is volume and seen["gt_mask"] is None
    assert seen["seed"] == 19
    assert results[0].mask_path == seen["saved"]
    import json
    payload = json.loads((tmp_path / "case" / "liver" / "trajectory.json").read_text())
    assert payload["turns"][0]["raw_output"] == "raw response"
    assert "predicted_mask" not in payload


def test_batch_failures_remain_in_summary(tmp_path, monkeypatch):
    from medsam_agent3d import evaluation
    def fail(**kwargs):
        raise RuntimeError("segmentation failed")
    monkeypatch.setattr(evaluation, "run_agent_loop", fail)
    results = evaluation.evaluate_dataset(None, None, [{
        "volume": np.zeros((3, 4, 5)), "mask": np.ones((3, 4, 5)),
        "volume_id": "bad", "target_name": "liver"}], str(tmp_path))
    assert not results[0].success
    assert results[0].final_iou == 0.
    assert "segmentation failed" in results[0].error


def test_binary_metrics_and_empty_convention():
    from medsam_agent3d.types import compute_metrics
    empty = np.zeros((2, 3, 4))
    assert compute_metrics(empty, empty) == (1., 1.)
    assert compute_metrics(np.ones_like(empty), np.full_like(empty, 255)) == (1., 1.)
    import pytest
    with pytest.raises(ValueError, match="matching 3D"):
        compute_metrics(empty, np.zeros((3, 4)))


def test_cli_single_forwards_sam_device_and_exits_on_failure(monkeypatch, tmp_path):
    from medsam_agent3d import cli, evaluation
    from medsam_agent3d.types import AgentResult
    import pytest
    seen = {}
    def fake_evaluate(**kwargs):
        seen.update(kwargs)
        return AgentResult("case", "liver", error="empty mask", termination_reason="empty_prediction")
    monkeypatch.setattr(evaluation, "evaluate_single", fake_evaluate)
    monkeypatch.setattr(cli, "_build_model_kwargs", lambda args: {})
    with pytest.raises(SystemExit) as exc:
        cli.main_single(["--volume", "scan.nii.gz", "--sam-device", "cpu", "--output-dir", str(tmp_path)])
    assert exc.value.code == 1
    assert seen["sam_device"] == "cpu"


def test_cli_single_returns_success(monkeypatch):
    from medsam_agent3d import cli, evaluation
    from medsam_agent3d.types import AgentResult
    expected = AgentResult("case", "liver", success=True)
    monkeypatch.setattr(evaluation, "evaluate_single", lambda **kwargs: expected)
    monkeypatch.setattr(cli, "_build_model_kwargs", lambda args: {})
    assert cli.main(["single", "--volume", "scan.nii.gz"]) is expected


def test_cli_cuda_request_has_actionable_error(monkeypatch):
    from medsam_agent3d import cli
    from types import SimpleNamespace
    import pytest
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="use --device cpu or auto"):
        cli._build_model_kwargs(SimpleNamespace(device="cuda", load_in_4bit=False))


def test_shared_sam_forwards_device_and_reuses_instance(monkeypatch):
    from medsam_agent3d.benchmark import _shared_sam
    from types import SimpleNamespace
    import sys
    calls = []
    class FakeSAM:
        def __init__(self, **kwargs):
            calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "medsam_agent3d.sam", SimpleNamespace(FrozenSAMMed3D=FakeSAM))
    args = SimpleNamespace(sam_ckpt="weights.pth", sam_device="cpu")
    first = _shared_sam(args)
    assert _shared_sam(args) is first
    assert calls == [{"checkpoint_path": "weights.pth", "device": "cpu"}]


def test_cli_batch_reports_failure_exit(monkeypatch, tmp_path):
    from medsam_agent3d import cli, evaluation, agent
    from medsam_agent3d.types import AgentResult
    from medsam_agent3d import io as io_utils
    from types import SimpleNamespace
    import sys
    import pytest
    seen = {}
    def make_sam(**kwargs):
        seen.update(kwargs)
        return object()
    monkeypatch.setitem(sys.modules, "medsam_agent3d.sam", SimpleNamespace(FrozenSAMMed3D=make_sam))
    monkeypatch.setattr(cli, "seed_everything", lambda seed: None)
    monkeypatch.setattr(cli, "_build_model_kwargs", lambda args: {})
    monkeypatch.setattr(io_utils, "get_totalsegmentator_dataset", lambda *args, **kwargs: [
        {"volume_id": "case", "target_name": "liver", "volume_path": "scan.nii.gz"}])
    monkeypatch.setattr(agent, "QwenVLAgent", lambda *args, **kwargs: object())
    monkeypatch.setattr(evaluation, "evaluate_dataset", lambda **kwargs: [
        AgentResult("case", "liver", error="prediction failed")])
    with pytest.raises(SystemExit) as exc:
        cli.main_batch(["--ts-dir", "dataset", "--sam-device", "cpu", "--output-dir", str(tmp_path)])
    assert exc.value.code == 1
    assert seen["device"] == "cpu"
