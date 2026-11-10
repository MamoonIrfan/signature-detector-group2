"""Explicit CPU subsets, separate from the complete training dataset."""
from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
from pathlib import Path

from common import read_json, safe_relative, write_json
from dataset import SPLITS, audit_local_dataset, prepare_dataset, validate_coco


def select_subset(coco, limit):
    if limit < 2:
        raise ValueError('Smoke split limits must be at least 2')
    rows = sorted(coco['images'], key=lambda row: str(row['file_name']))
    positive_ids = {ann['image_id'] for ann in coco['annotations']}
    positive = next((row for row in rows if row['id'] in positive_ids), None)
    negative = next((row for row in rows if row['id'] not in positive_ids), None)
    if positive is None:
        raise ValueError('Each smoke split requires a labeled positive')
    selected = {positive['id']: positive}
    if negative is not None:
        selected[negative['id']] = negative
    for row in rows:
        if len(selected) >= limit:
            break
        selected[row['id']] = row
    return {**coco, 'images': sorted(selected.values(), key=lambda row: str(row['file_name'])),
            'annotations': [ann for ann in coco['annotations'] if ann['image_id'] in selected]}


def prepare_day3_dataset(project, config):
    if not config['smoke']:
        return prepare_dataset(project, config['dataset_repo'], config['dataset_revision'])
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get('HF_TOKEN') or None
    info = HfApi(token=token).dataset_info(repo_id=config['dataset_repo'], revision=config['dataset_revision'])
    revision = info.sha
    if not revision:
        raise RuntimeError('HF returned no immutable dataset revision')
    limits = config['smoke_limits']
    suffix = hashlib.sha256(str(sorted(limits.items())).encode()).hexdigest()[:12]
    destination = Path(project)/'datasets'/f'{revision}_day3_smoke_{suffix}'
    destination.mkdir(parents=True, exist_ok=True)
    # File listing avoids masking permission/network failures as a missing archive.
    files = set(HfApi(token=token).list_repo_files(config['dataset_repo'], repo_type='dataset', revision=revision))

    def fetch(name):
        return Path(hf_hub_download(repo_id=config['dataset_repo'], filename=name,
                    repo_type='dataset', revision=revision, token=token,
                    cache_dir=str(Path(project)/'downloads'/'hf_smoke_cache')))

    for split, directory in SPLITS.items():
        name = f'annotations/instances_{directory}.json'
        coco, clipped = validate_coco(read_json(fetch(name)))
        subset = select_subset(coco, limits[split])
        if f'{directory}.zip' in files:
            print(f'Smoke: downloading {directory}.zip; extracting only {len(subset["images"])} images.', flush=True)
            # Archives may need a full download; training and pixel audit remain bounded.
            with zipfile.ZipFile(fetch(f'{directory}.zip')) as archive:
                members = {member.filename: member for member in archive.infolist()}
                for row in subset['images']:
                    relative = safe_relative(f'{directory}/{row["file_name"]}')
                    member = members[str(relative)]
                    if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                        raise ValueError('ZIP symbolic links are not supported')
                    target = destination.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open('wb') as output:
                        shutil.copyfileobj(source, output)
        else:
            for row in subset['images']:
                relative = safe_relative(f'{directory}/{row["file_name"]}')
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fetch(str(relative)), target)
        write_json(destination/name, subset)
    report = audit_local_dataset(destination, verify_pixels=True)
    card_data = getattr(info, 'card_data', None) or {}
    license_value = card_data.get('license', 'unspecified') if hasattr(card_data, 'get') else 'unspecified'
    report.update(repo=config['dataset_repo'], revision=revision, smoke_test=True,
                  smoke_limits=limits, schema_version=1, dataset_license=str(license_value),
                  license_note='Copied from the pinned HF dataset card')
    for split in SPLITS:
        report['splits'][split]['source_boxes_clipped'] = None
    write_json(destination/'audit.json', report)
    return destination, report
