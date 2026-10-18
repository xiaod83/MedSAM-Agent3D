"""Checkpoint selection regressions without model dependencies or network calls."""
import ast
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

class CheckpointSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.valid = b'verified checkpoint fixture'
        self.runtime = self.root / 'runtime.pth'
        self.drive = self.root / 'drive.pth'
        source = Path(__file__).resolve().parents[1] / 'notebooks/colab_runner.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('verify_checkpoint', 'ensure_checkpoint')]
        self.scope = {'hashlib': hashlib, 'shutil': shutil,
                      'CHECKPOINT_SHA256': hashlib.sha256(self.valid).hexdigest(),
                      'CHECKPOINT_URL': 'https://example.invalid/checkpoint'}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), self.scope)

    def download(self, url, path):
        path.write_bytes(self.valid)

    def test_unrecognized_drive_file_is_preserved_and_download_used(self):
        self.drive.write_bytes(b'older checkpoint')
        with patch('urllib.request.urlretrieve', side_effect=self.download) as download:
            result = self.scope['ensure_checkpoint'](self.runtime, self.drive)
        self.assertEqual(result.read_bytes(), self.valid)
        self.assertEqual(self.drive.read_bytes(), b'older checkpoint')
        download.assert_called_once()

    def test_verified_runtime_reused_before_unrecognized_drive(self):
        self.runtime.write_bytes(self.valid)
        self.drive.write_bytes(b'older checkpoint')
        with patch('urllib.request.urlretrieve') as download:
            self.scope['ensure_checkpoint'](self.runtime, self.drive)
        download.assert_not_called()

    def test_verified_drive_copy_needs_no_download(self):
        self.drive.write_bytes(self.valid)
        with patch('urllib.request.urlretrieve') as download:
            self.scope['ensure_checkpoint'](self.runtime, self.drive)
        self.assertEqual(self.runtime.read_bytes(), self.valid)
        download.assert_not_called()

    def test_bad_download_does_not_replace_existing_runtime(self):
        self.runtime.write_bytes(b'previous runtime checkpoint')
        def invalid_download(url, path):
            path.write_bytes(b'bad response')
        with patch('urllib.request.urlretrieve', side_effect=invalid_download):
            with self.assertRaises(ValueError):
                self.scope['ensure_checkpoint'](self.runtime, self.drive)
        self.assertEqual(self.runtime.read_bytes(), b'previous runtime checkpoint')

if __name__ == '__main__':
    unittest.main()
