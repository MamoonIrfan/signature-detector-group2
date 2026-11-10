"""Shared helpers. Configuration files contain no credentials."""
from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

YOLOX_COMMIT = '419778480ab6ec0590e5d3831b3afb3b46ab2aa3'
DATASET_REPO = 'AimaArshad/signature-detection-yolox'
MODEL_SIZES = {'s': (0.33, 0.50)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value):
    item = PurePosixPath(value)
    if not value or '\\' in value or ':' in value or item.is_absolute() or '..' in item.parts:
        raise ValueError('Unsafe relative path: ' + value)
    return item


def safe_id(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}', value):
        raise ValueError('Run ID must contain only letters, numbers, underscores or hyphens')
    return value


def resolve_device_name(requested='auto', cuda_available=False):
    if requested not in ('auto', 'cuda', 'cuda:0', 'cpu'):
        raise ValueError('device must be auto, cuda or cpu')
    if requested in ('cuda', 'cuda:0') and not cuda_available:
        raise RuntimeError('Select a Colab GPU, or explicitly use device=cpu')
    return 'cuda:0' if requested != 'cpu' and cuda_available else 'cpu'


def identity(config, audit):
    """Settings that must stay unchanged during an interrupted run."""
    return {'schema': 'day2-v1', 'yolox_commit': YOLOX_COMMIT,
            'dataset_repo': audit['repo'], 'dataset_revision': audit['revision'],
            'annotation_hashes': {k: v['annotation_sha256'] for k, v in audit['splits'].items()},
            **{k: config[k] for k in ('epochs', 'batch_size', 'img_size', 'seed',
                                      'conf_threshold', 'nms_threshold')}, 'model_size': 's'}


def validate_config(config):
    safe_id(config['run_id'])
    if config.get('model_size', 's') != 's' or config['img_size'] != 640:
        raise ValueError('This workbook is YOLOX-S with fixed 640 input')
    for key in ('epochs', 'batch_size'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if config['batch_size'] < 2:
        raise ValueError('Use batch_size >= 2 for YOLOX BatchNorm training')
    for key in ('conf_threshold', 'nms_threshold'):
        if not 0 < config[key] <= 1:
            raise ValueError(key + ' must be in (0,1]')
    if not config.get('drive_dir') and not config.get('enable_r2'):
        raise ValueError('Enable Drive or R2 so every completed epoch survives a runtime reset')
    return config
