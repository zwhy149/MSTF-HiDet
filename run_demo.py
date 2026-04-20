from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'

sample_files = [
    '10惟_Charging Short-circuit.xlsx',
    '1惟Charging Short-circuit.xlsx',
    '0.1惟Charging Short-circuit.xlsx',
    '0.01惟Charging Short-circuit.xlsx',
    '10惟_Full-SOC Resting Short-circuit.xlsx',
    '1惟_Full-SOC Resting Short-circuit.xlsx',
    '0.1惟_Full-SOC Resting Short-circuit.xlsx',
    '0.01惟_Full-SOC Resting Short-circuit.xlsx',
]

for name in sample_files:
    file_path = DATA_DIR / name
    if not file_path.exists():
        print(f'[SKIP] Missing sample file: {name}')
        continue

    print(f'\n=== Demo on: {name} ===')
    subprocess.run(
        [sys.executable, str(ROOT / 'detection.py'), '--file', str(file_path)],
        check=True
    )
