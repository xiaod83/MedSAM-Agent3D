"""Single entry point. Run `python run.py --help` from the project folder."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
HELP = '''MedSAM-Agent3D

Run commands from this folder. On Windows PowerShell, use:
  .\\.venv\\Scripts\\python.exe run.py <command>

Commands:
  doctor       Show installed dependencies, GPU and checkpoint access
  setup        Install dependencies and run checks (default: CPU)
  test         Run tests; --light runs tests without ML dependencies
  single       Segment one volume (--volume PATH --target NAME)
  batch        Evaluate a dataset (--ts-dir PATH)
  benchmark    Compare segmentation methods
  validate     Exercise the actual SAM checkpoint on the bundled sample
  bundle       Build the optional Colab source ZIP
  remote       Optional Colab CLI launcher (Linux/macOS, not native Windows)

Examples:
  python run.py doctor
  python run.py test --light
  python run.py single --volume scan.nii.gz --target liver

Folder guide: README.md
'''


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help', 'help'):
        print(HELP)
        return 0
    command, arguments = argv[0], argv[1:]
    if command == 'test':
        if arguments and arguments[0] == '--light':
            for pattern in ('test_actions.py', 'test_colab_checkpoint.py', 'test_colab_remote.py', 'test_views.py', 'test_agent_model.py', 'test_case_manifest.py'):
                status = subprocess.call([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern], cwd=ROOT)
                if status:
                    return status
            return 0
        return subprocess.call([sys.executable, '-m', 'pytest', *arguments], cwd=ROOT)
    scripts = {'doctor': ('bootstrap.py', ['--doctor']), 'setup': ('bootstrap.py', []),
               'validate': ('validate_checkpoint.py', []), 'bundle': ('build_colab_bundle.py', []),
               'remote': ('colab_remote.py', [])}
    if command in scripts:
        script, defaults = scripts[command]
        return subprocess.call([sys.executable, str(ROOT / 'scripts' / script), *defaults, *arguments], cwd=ROOT)
    try:
        if command == 'single':
            from medsam_agent3d.cli import main_single
            main_single(arguments)
        elif command == 'batch':
            from medsam_agent3d.cli import main_batch
            main_batch(arguments)
        elif command == 'benchmark':
            from medsam_agent3d.benchmark import main as benchmark
            sys.argv = [str(ROOT / 'run.py'), *arguments]
            benchmark()
        else:
            print(f'Unknown command: {command}\nRun python run.py --help', file=sys.stderr)
            return 2
    except ModuleNotFoundError as exc:
        print(f'Missing dependency: {exc.name}. Run python run.py doctor for details.\n'
              'Install requirements using python -m pip install -r requirements.txt.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
