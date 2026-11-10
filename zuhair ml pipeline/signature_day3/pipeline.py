"""Reusable Day 2 training pipeline. See README.md for staged and one-command use."""
from __future__ import annotations
import argparse
import datetime
import os
import shutil
import sys
import zipfile
from pathlib import Path
from common import read_json, write_json, validate_config, YOLOX_COMMIT, sha256_file

ROOT = Path(__file__).resolve().parent


def initialize(config_path):
    config = validate_config(read_json(config_path))
    source = Path(config['source_dir'])
    if not (source/'.source_revision').is_file() or (source/'.source_revision').read_text().strip() != YOLOX_COMMIT:
        raise RuntimeError('Run bootstrap.py first; source must match the pinned YOLOX commit')
    os.environ['YOLOX_SOURCE_DIR'] = str(source)
    os.environ['PYTHONPATH'] = str(ROOT)+os.pathsep+str(source)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    out = ROOT/'outputs'/config['run_id']
    out.mkdir(parents=True, exist_ok=True)
    store = None
    if config.get('enable_r2'):
        from r2_storage import R2Store
        store = R2Store.from_env(config.get('r2_prefix', 'signature-detection-day2'))
        store.check_connection()
    return config, out, store


def prepare(config, out, store=None, resume=False):
    state = None
    if resume:
        from checkpoints import recover
        state = recover(out, config, store)
    else:
        assert_fresh_run_available(out, config, store)
    if store:
        # Test actual read/write permissions before downloading data or using GPU time.
        key = 'runs/'+config['run_id']+'/connection_check.json'
        marker = {'run_id': config['run_id'], 'status': 'r2_read_write_verified'}
        store.put_json(marker, key)
        if store.get_json(key) != marker:
            raise RuntimeError('R2 read/write preflight did not return the written marker')
        print('R2 read/write verified. Every-epoch R2 backup is required.', flush=True)
    from dataset import prepare_dataset
    dataset, audit = prepare_dataset(ROOT, config['dataset_repo'], config['dataset_revision'])
    config['dataset_revision'] = audit['revision']
    config['dataset_dir'] = str(dataset)
    write_json(out/'config.json', config)
    write_json(out/'reports'/'dataset_audit.json', audit)
    print('Complete split counts:', {k: v['images'] for k, v in audit['splits'].items()}, flush=True)
    print('Unsigned counts:', {k: v['negative_images'] for k, v in audit['splits'].items()}, flush=True)
    return dataset, audit, state


def assert_fresh_run_available(out, config, store=None):
    """A new run must not overwrite completed checkpoints from a previous session."""
    if (Path(out)/'checkpoints'/'latest.pth').is_file():
        raise ValueError('This local RUN_ID already has training state. Use resume=true or a new RUN_ID.')
    if config.get('drive_dir') and (Path(config['drive_dir'])/config['run_id']/'latest.json').is_file():
        raise ValueError('This Drive RUN_ID already has checkpoints. Use resume=true or a new RUN_ID.')
    if store and store.exists('runs/'+config['run_id']+'/checkpoints/latest.json'):
        raise ValueError('This R2 RUN_ID already has checkpoints. Use resume=true or a new RUN_ID.')


def existing_data(config, out):
    saved = read_json(out/'config.json')
    config['dataset_revision'] = saved['dataset_revision']
    config['dataset_dir'] = saved['dataset_dir']
    dataset = Path(config['dataset_dir'])
    if not dataset.is_dir():
        raise FileNotFoundError('Run prepare again after reconnecting to restore the pinned dataset')
    audit = read_json(out/'reports'/'dataset_audit.json')
    return dataset, audit


def publish(artifact, out, config, store=None):
    artifact = Path(artifact)
    metadata = read_json(artifact/'metadata.json')
    parity = read_json(artifact/'onnx_parity.json')
    acceptance = read_json(out/'acceptance'/'acceptance_tests.json')
    if not parity.get('passed') or not acceptance.get('passed'):
        raise RuntimeError('Publish requires passing parity and the positive/negative image acceptance tests')
    receipt = {'run_id': config['run_id'], 'model_sha256': metadata['model_sha256'],
               'validation_metrics': read_json(out/'evaluation'/'onnx'/'val_metrics.json'),
               'test_metrics': read_json(out/'evaluation'/'onnx'/'test_metrics.json'),
               'destinations': {}}
    # This is a model upload; private R2 keys are not public HTTP links.
    if store:
        root = 'runs/'+config['run_id']+'/model/'
        objects = {}
        for filename in ('signature_yolox_s.onnx', 'metadata.json', 'farhan_handoff.zip', 'onnx_parity.json'):
            objects[filename] = store.put_file(artifact/filename, root+filename)
        for name in ('metrics_table.csv', 'metrics_table.png', 'errors.csv', 'ERRORS.md', 'error_summary.json'):
            store.put_file(out/'reports'/name, root+name)
        store.put_file(out/'acceptance'/'acceptance_tests.json', root+'acceptance_tests.json')
        store.put_json({**receipt, 'objects': objects}, root+'manifest.json')
        receipt['destinations']['r2'] = objects
    if config.get('hf_model_repo'):
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get('HF_TOKEN') or None)
        repo = config['hf_model_repo']
        api.create_repo(repo_id=repo, repo_type='model', private=True, exist_ok=True)
        # Existing repository visibility is preserved.
        stage = out/'hf_upload'
        stage.mkdir(exist_ok=True)
        for name in ('signature_yolox_s.onnx', 'metadata.json', 'detect.py', 'requirements_inference.txt',
                     'onnx_parity.json', 'farhan_handoff.zip', 'FARHAN_HANDOFF.md', 'LICENSE', 'LICENSE-YOLOX.txt', 'NOTICE.txt'):
            shutil.copy2(artifact/name, stage/name)
        for name in ('metrics_table.csv', 'metrics_table.png', 'ERRORS.md'):
            shutil.copy2(out/'reports'/name, stage/name)
        model_card = ('---\nlibrary_name: onnx\npipeline_tag: object-detection\n'
                      'datasets:\n- '+config['dataset_repo']+'\n---\n\n'
                      '# YOLOX-S signature detector\n\n'
                      'One signature class. Fixed 640 BGR input, raw float32 0–255; no /255. '
                      'decode.py / detect.py applies grid/stride decode and NMS.\n\n'
                      'Use detect.py and metadata.json; see FARHAN_HANDOFF.md. '
                      'Measured metrics are in metrics_table.csv/png. '
                      'Dataset revision and declared license are recorded in metadata.json.\n\n'
                      'Detection locates signatures; it does not establish authenticity or writer identity.\n')
        (stage/'README.md').write_text(model_card.replace('decode.py / detect.py', 'detect.py'), encoding='utf-8')
        commit = api.upload_folder(repo_id=repo, repo_type='model', folder_path=str(stage),
                                   commit_message='Day 2 YOLOX-S ONNX model and measured evaluation')
        receipt['destinations']['huggingface'] = {'repo_id': repo, 'commit_url': str(commit.commit_url)}
    if not receipt['destinations']:
        raise ValueError('Enable R2 or set hf_model_repo for the requested model upload')
    write_json(out/'reports'/'upload_receipt.json', receipt)
    return receipt


def persist_deliverables(out, config):
    if not config.get('drive_dir'):
        return
    from checkpoints import copy_atomic
    destination = Path(config['drive_dir'])/config['run_id']/'deliverables'
    for directory in ('artifact', 'reports', 'acceptance', 'evaluation/onnx', 'early_handoff'):
        source = out/directory
        if not source.is_dir():
            continue
        for file in source.rglob('*'):
            if file.is_file():
                copy_atomic(file, destination/file.relative_to(out))
    if (out/'day2_evidence.zip').is_file():
        copy_atomic(out/'day2_evidence.zip', destination/'day2_evidence.zip')


def evidence_zip(out):
    path = Path(out)/'day2_evidence.zip'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for directory in ('reports', 'acceptance', 'evaluation/onnx'):
            root = Path(out)/directory
            if root.is_dir():
                for file in root.rglob('*'):
                    if file.is_file():
                        archive.write(file, file.relative_to(out))
        for name in ('config.json', 'artifact/metadata.json', 'artifact/onnx_parity.json'):
            if (Path(out)/name).is_file():
                archive.write(Path(out)/name, name)
    return path


def run(action, config_path):
    config, out, store = initialize(config_path)
    status_path = out/'reports'/'status.json'
    status = {'run_id': config['run_id'], 'stage': action, 'status': 'running',
              'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(status_path, status)
    try:
        if action in ('prepare', 'train', 'all'):
            resume = bool(config.get('resume'))
            if not resume and action in ('train', 'all') and (out/'checkpoints'/'latest.pth').is_file():
                raise ValueError('This RUN_ID already has a checkpoint. Set resume=true, or choose a new RUN_ID.')
            dataset, audit, state = prepare(config, out, store, resume=resume)
        else:
            dataset, audit = existing_data(config, out)
            state = None
        if action == 'prepare':
            status['status'] = 'prepared'
        if action in ('train', 'all'):
            from bootstrap import download
            initial = None
            if state is None:
                initial = Path(config['initial_checkpoint']) if config.get('initial_checkpoint') else ROOT/'pretrained'/'yolox_s.pth'
                if not initial.is_file():
                    if config.get('initial_checkpoint'):
                        raise FileNotFoundError('Put the prior best checkpoint in Drive and set its Colab path')
                    download('https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth', initial)
            from training import train
            train(dataset, out, config, audit, initial, state, store)
            status['status'] = 'training_completed'
        if action in ('export', 'all'):
            from export_model import export_and_verify
            checkpoint = Path(config['export_checkpoint']) if config.get('export_checkpoint') else out/'checkpoints'/'best.pth'
            if not checkpoint.is_file():
                raise FileNotFoundError('Train a completed epoch or set export_checkpoint to an existing single-class best.pth')
            export_and_verify(checkpoint, dataset, out/'artifact', config, audit, out)
            status['status'] = 'onnx_verified'
        if action in ('evaluate', 'all'):
            from onnx_evaluate import evaluate_onnx
            from reports import metric_tables, observed_errors
            for split in ('val', 'test'):
                evaluate_onnx(out/'artifact', dataset, split, out/'evaluation'/'onnx', config)
            metric_tables(out)
            observed_errors(dataset, out, config)
            status['status'] = 'evaluated'
        if action in ('test', 'all'):
            from reports import acceptance_tests
            acceptance = acceptance_tests(out/'artifact', dataset, out, config)
            status['status'] = 'acceptance_passed' if acceptance['passed'] else 'acceptance_failed'
        if action == 'publish' or (action == 'all' and config.get('publish_model')):
            publish(out/'artifact', out, config, store)
            status['status'] = 'uploaded'
        if action == 'all':
            progress = read_json(out/'reports'/'progress.json')
            parity = read_json(out/'artifact'/'onnx_parity.json')
            acceptance = read_json(out/'acceptance'/'acceptance_tests.json')
            done = progress['epoch'] >= config['epochs'] and parity['passed'] and acceptance['passed']
            uploaded = (out/'reports'/'upload_receipt.json').is_file()
            status.update(status='done' if done and uploaded else 'needs_followup',
                          training_complete=progress['epoch'] >= config['epochs'],
                          onnx_parity_passed=parity['passed'], acceptance_passed=acceptance['passed'], model_uploaded=uploaded)
        status['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(status_path, status)
        evidence_zip(out)
        persist_deliverables(out, config)
        if action == 'test' and status['status'] == 'acceptance_failed':
            raise AssertionError('Real model acceptance tests failed. Evidence was saved; inspect it before handoff.')
        if action == 'all' and status['status'] == 'needs_followup':
            raise RuntimeError('Training artifacts saved, but one or more requested completion checks remain. See status.json.')
    except Exception as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        write_json(status_path, status)
        persist_deliverables(out, config)
        raise
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train', 'export', 'evaluate', 'test', 'publish', 'all'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print('Output:', run(args.action, args.config))
