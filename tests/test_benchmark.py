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
    report.to_json("test_outputs/benchmark_test.json")
    assert os.path.exists("test_outputs/benchmark_test.json")
    print("  [OK] JSON saved")

    # Verify METHOD_REGISTRY is accessible
    print(f"\n  Available real methods: {list(METHOD_REGISTRY.keys())}")

    print("\n" + "=" * 60)
    print("*** BENCHMARK INFRASTRUCTURE OK ***")
    print("=" * 60)


if __name__ == "__main__":
    os.makedirs("test_outputs", exist_ok=True)
    test_benchmark()
