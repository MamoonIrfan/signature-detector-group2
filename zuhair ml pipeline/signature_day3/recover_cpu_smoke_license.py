"""Paste this cell into the existing Colab session, then rerun Section 4."""
import ast
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT = Path('/content/signature_day3')
RUN_ID = 'signature_day3_cpu_smoke_e2_b2_20261109'
OUT = PROJECT/'outputs'/RUN_ID

saved = json.loads((OUT/'config.json').read_text(encoding='utf-8'))
if saved.get('run_id') != RUN_ID or not saved.get('smoke'):
    raise ValueError('Expected the saved CPU smoke run configuration')

# Restore the Apache license from the pinned YOLOX source already downloaded.
source_license = Path(saved['source_dir'])/'LICENSE'
if not source_license.is_file():
    raise FileNotFoundError('The pinned YOLOX LICENSE is missing: '+str(source_license))
shutil.copy2(source_license, PROJECT/'LICENSE')

# Revision 0 also filtered out extensionless LICENSE while repacking the ZIP.
script = PROJECT/'train_day3.py'
text = script.read_text(encoding='utf-8')
old = "if file.is_file() and file.suffix in ('.onnx','.json','.py','.txt','.md'):"
new = "if file.is_file() and (file.name == 'LICENSE' or file.suffix in ('.onnx','.json','.py','.txt','.md')):"
if old in text:
    repaired = text.replace(old, new, 1)
    ast.parse(repaired)
    script.write_text(repaired, encoding='utf-8')
elif new not in text and 'package_handoff(artifact)' not in text:
    raise RuntimeError('Unexpected script version; retain checkpoints and use the corrected package')

PROFILE = 'cpu_smoke'
RESUME = True
DEVICE = 'cpu'
EPOCHS = saved['epochs']
BATCH_SIZE = saved['batch_size']
DATASET_REVISION = saved['dataset_revision']
limits = saved['smoke_limits']

COMMAND = [sys.executable, '-u', str(PROJECT/'train_day3.py'),
    '--dataset', saved['dataset_repo'], '--dataset-revision', DATASET_REVISION,
    '--epochs', str(EPOCHS), '--batch-size', str(BATCH_SIZE),
    '--device', 'cpu', '--workers', '0', '--output', str(OUT),
    '--r2-prefix', saved['r2_prefix'],
    '--registry-name', saved['registry_name'].removesuffix('-smoke'),
    '--smoke', '--train-limit', str(limits['train']),
    '--val-limit', str(limits['val']), '--test-limit', str(limits['test']),
    '--resume']

print('LICENSE restored; handoff packaging repaired.')
print('Resume:', RUN_ID, '| target epochs:', EPOCHS, '| batch:', BATCH_SIZE)
print('Now rerun Section 4. The saved completed epochs are retained.')
