from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'

sample_files = [
    '10Ω_Charging Short-circuit.xlsx',
    '1ΩCharging Short-circuit.xlsx',
    '0.1ΩCharging Short-circuit.xlsx',
    '0.01ΩCharging Short-circuit.xlsx',
    '10Ω_Full-SOC Resting Short-circuit.xlsx',
    '1Ω_Full-SOC Resting Short-circuit.xlsx',
    '0.1Ω_Full-SOC Resting Short-circuit.xlsx',
    '0.01Ω_Full-SOC Resting Short-circuit.xlsx',
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
