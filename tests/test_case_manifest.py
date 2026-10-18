"""Dependency-free tests for the CLI manifest boundary (no model imports)."""
import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import types


# Execute the real CLI functions without its optional numerical/model imports.
_SOURCE = Path(__file__).resolve().parents[1] / "medsam_agent3d" / "cli.py"
_TREE = ast.parse(_SOURCE.read_text(encoding="utf-8-sig"), filename=str(_SOURCE))
_FUNCTIONS = ast.Module(body=[node for node in _TREE.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
_NAMESPACE = dict(argparse=argparse, os=os, random=random, sys=sys, __file__=str(_SOURCE), __name__="medsam_agent3d.cli", __package__="medsam_agent3d")
exec(compile(_FUNCTIONS, str(_SOURCE), "exec"), _NAMESPACE)
_NAMESPACE["seed_everything"] = lambda seed: None


class CaseManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scan.nii.gz").touch()
        (self.root / "mask.nii.gz").touch()

    def manifest(self, data):
        path = self.root / "cases.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def case(self, **kwargs):
        return dict(volume_id="case", target_name="liver", volume_path="scan.nii.gz", **kwargs)

    def test_preserves_order_and_resolves_relative_paths(self):
        cases = [self.case(mask_path="mask.nii.gz"),
                 dict(volume_id="second", target_name="spleen", ct_path="scan.nii.gz")]
        result = _NAMESPACE["_load_case_manifest"](self.manifest(cases))
        self.assertEqual([entry["target_name"] for entry in result], ["liver", "spleen"])
        self.assertEqual(result[0]["mask_path"], str(self.root / "mask.nii.gz"))
        self.assertEqual(result[1]["volume_path"], str(self.root / "scan.nii.gz"))
        self.assertEqual(result[1]["ct_path"], result[1]["volume_path"])

    def test_rejects_empty_and_nonlist_manifests(self):
        for data in ([], {}, None):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "nonempty JSON list"):
                _NAMESPACE["_load_case_manifest"](self.manifest(data))

    def test_rejects_missing_files(self):
        case = self.case()
        case["volume_path"] = "missing.nii.gz"
        with self.assertRaisesRegex(ValueError, "volume_path does not exist"):
            _NAMESPACE["_load_case_manifest"](self.manifest([case]))

    def test_rejects_duplicate_destinations(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _NAMESPACE["_load_case_manifest"](self.manifest([self.case(), self.case()]))

    def test_rejects_unsafe_or_missing_identifiers(self):
        for key, value in (("target_name", ""), ("volume_id", "../outside"), ("target_name", None)):
            case = self.case()
            case[key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                _NAMESPACE["_load_case_manifest"](self.manifest([case]))

    def test_manifest_cli_preserves_prefix_without_dataset_loader(self):
        cases = [self.case(), dict(volume_id="second", target_name="spleen", volume_path="scan.nii.gz")]
        seen = {}
        def evaluate(**kwargs):
            seen.update(kwargs)
            return [types.SimpleNamespace(success=True, quality_status="unscored", target_overlap=None)]
        def sam(**kwargs):
            seen["sam_loads"] = seen.get("sam_loads", 0) + 1
            return object()
        package = types.ModuleType("medsam_agent3d")
        package.__path__ = []
        modules = {"torch": types.SimpleNamespace(), "medsam_agent3d": package,
                   "medsam_agent3d.sam": types.SimpleNamespace(FrozenSAMMed3D=sam),
                   "medsam_agent3d.agent": types.SimpleNamespace(QwenVLAgent=lambda *args, **kwargs: object()),
                   "medsam_agent3d.evaluation": types.SimpleNamespace(evaluate_dataset=evaluate)}
        with patch.dict(sys.modules, modules), patch.dict(_NAMESPACE, {"_build_model_kwargs": lambda args: {}}), contextlib.redirect_stdout(io.StringIO()):
            _NAMESPACE["main_batch"](["--cases-json", str(self.manifest(cases)),
                "--max-samples", "1", "--output-dir", str(self.root / "outputs")])
        self.assertEqual([entry["target_name"] for entry in seen["dataset_entries"]], ["liver"])
        self.assertEqual(seen["sam_loads"], 1)
    def test_invalid_manifest_cli_exits_before_model_imports(self):
        stderr = io.StringIO()
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace()}), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            _NAMESPACE["main_batch"](["--cases-json", str(self.manifest([])),
                                       "--output-dir", str(self.root / "outputs")])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("Invalid --cases-json", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
