"""Local-to-Colab transfer/control tests; no ML libraries or cloud account needed."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('colab_remote', ROOT / 'scripts/colab_remote.py')
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)

class RemoteLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.volume = self.directory / 'scan.nii.gz'
        self.volume.write_bytes(b'volume fixture')
        self.output = self.directory / 'results'

    def test_payload_contains_selected_inputs_and_no_weights_or_credentials(self):
        destination = self.directory / 'payload.zip'
        volume, mask = remote.create_payload(destination, self.volume)
        with zipfile.ZipFile(destination) as archive:
            self.assertEqual(archive.read(volume), b'volume fixture')
            self.assertIn('run.py', archive.namelist())
            self.assertFalse(any(name.endswith(('.pth', '.env')) or name.startswith('.git/') for name in archive.namelist()))
        self.assertIsNone(mask)

    def test_generated_remote_program_compiles_with_quoted_parameters(self):
        code = remote.remote_script({'target': "name'\\\n", 'model': 'local model'})
        compile(code, '<generated remote worker>', 'exec')

    def test_results_download_even_when_remote_inference_fails(self):
        operations = []
        def fake_run(command, **kwargs):
            operation = next(word for word in command if word in ('status', 'upload', 'exec', 'download'))
            operations.append(operation)
            if operation == 'upload':
                self.assertTrue(Path(command[-2]).is_file())
            if operation == 'download':
                with zipfile.ZipFile(command[-1], 'w') as archive:
                    archive.writestr('job_status.json', json.dumps({'success': False, 'error': 'Model failed'}))
                    archive.writestr('remote.log', 'failure details')
            return subprocess.CompletedProcess(command, 1 if operation == 'exec' else 0)
        with patch.object(remote.shutil, 'which', return_value='colab'), patch.object(remote.subprocess, 'run', side_effect=fake_run):
            code = remote.main(['--volume', str(self.volume), '--output-dir', str(self.output)])
        self.assertEqual(code, 1)
        self.assertEqual(operations, ['status', 'upload', 'exec', 'download'])
        self.assertEqual((self.output / 'remote.log').read_text(), 'failure details')

    def test_output_archive_cannot_escape_local_directory(self):
        archive = self.directory / 'unsafe.zip'
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr('../escape.txt', 'bad')
        self.output.mkdir()
        with self.assertRaises(ValueError):
            remote.extract_results(archive, self.output)
        self.assertFalse((self.directory / 'escape.txt').exists())

if __name__ == '__main__':
    unittest.main()
