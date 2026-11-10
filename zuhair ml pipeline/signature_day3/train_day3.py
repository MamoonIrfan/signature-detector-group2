"""One command: HF download -> YOLOX training -> ONNX -> gate -> R2 + MLflow."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path

from common import DATASET_REPO, YOLOX_COMMIT, read_json, safe_id, sha256_file, validate_config, write_json

ROOT = Path(__file__).resolve().parent
PINNED_DATASET = 'a55352328a1fbc3442b54db0e377f5add470bcba'
DAY2_BASELINE = 'runs/signature_day2_fresh30_b16_20261108/model/manifest.json'


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--dataset', default=DATASET_REPO, help='HF dataset repo_id, not a URL')
    value.add_argument('--dataset-revision', default=PINNED_DATASET, help='Immutable HF commit or branch to resolve')
    value.add_argument('--epochs', type=int, default=30)
    value.add_argument('--output', required=True, help='Dedicated output directory; basename is the run ID')
    value.add_argument('--batch-size', type=int, default=None, help='Default: 16 full training, 2 smoke')
    value.add_argument('--device', choices=('auto','cuda','cpu'), default='auto')
    value.add_argument('--workers', type=int, default=0)
    value.add_argument('--resume', action='store_true')
    value.add_argument('--smoke', action='store_true', help='Small CPU integration trial; never promotes')
    value.add_argument('--train-limit', type=int, default=32)
    value.add_argument('--val-limit', type=int, default=16)
    value.add_argument('--test-limit', type=int, default=16)
    value.add_argument('--r2-prefix', default='signature-detection-day2')
    value.add_argument('--registry-name', default='signature-yolox-s')
    value.add_argument('--baseline-manifest', default=DAY2_BASELINE, help='R2 key relative to --r2-prefix')
    value.add_argument('--min-delta', type=float, default=0.0, help='Absolute AP difference; default requires strict improvement')
    value.add_argument('--initial-checkpoint', default='', help='Optional local warm-start .pth; default official COCO weights')
    value.add_argument('--negative-image', default='', help='Optional visually reviewed unsigned document')
    value.add_argument('--dry-run', action='store_true', help='Print the plan without downloading/installing/training')
    value.add_argument('--runtime-ready', action='store_true', help=argparse.SUPPRESS)
    return value


def make_config(args):
    if not math.isfinite(args.min_delta) or args.min_delta < 0:
        raise ValueError('--min-delta must be finite and >= 0')
    if args.workers < 0:
        raise ValueError('--workers must be >= 0')
    safe_id(args.registry_name)
    if not args.dataset_revision:
        raise ValueError('--dataset-revision must not be empty')
    output = Path(args.output).expanduser().resolve()
    batch = args.batch_size if args.batch_size is not None else (2 if args.smoke else 16)
    config = {'run_id': safe_id(output.name), 'resume': args.resume,
        'dataset_repo': args.dataset, 'dataset_revision': args.dataset_revision,
        'epochs': args.epochs, 'batch_size': batch, 'img_size': 640, 'model_size': 's',
        'seed': 42, 'workers': args.workers, 'device': args.device, 'fp16': args.device != 'cpu',
        'conf_threshold': 0.35, 'nms_threshold': 0.45, 'drive_dir': '',
        'enable_r2': True, 'require_r2_backup': True, 'r2_prefix': args.r2_prefix,
        'hf_model_repo': '', 'initial_checkpoint': args.initial_checkpoint,
        'export_checkpoint': '', 'negative_image': args.negative_image,
        'early_export_epoch': 0, 'publish_model': False, 'source_dir': str(ROOT/'vendor'/'YOLOX'),
        'smoke': args.smoke, 'smoke_limits': {'train': args.train_limit, 'val': args.val_limit, 'test': args.test_limit},
        'registry_name': args.registry_name+('-smoke' if args.smoke else ''), 'min_delta': args.min_delta,
        'baseline_manifest': args.baseline_manifest, 'yolox_commit': YOLOX_COMMIT}
    safe_id(config['registry_name'])
    if args.smoke and min(config['smoke_limits'].values()) < 2:
        raise ValueError('All smoke split limits must be >= 2')
    return validate_config(config), output


def download_baseline(store, record, destination):
    destination = Path(destination)
    objects = record.get('objects', {})
    for name in ('signature_yolox_s.onnx','metadata.json','onnx_parity.json'):
        if name not in objects:
            raise ValueError('Baseline manifest lacks '+name+'; complete Day 2 Section 13 first')
        store.download_verified(objects[name], destination/name)
    metadata = read_json(destination/'metadata.json')
    if metadata.get('model_sha256') != sha256_file(destination/'signature_yolox_s.onnx'):
        raise ValueError('Baseline model hash disagrees with metadata')
    if not read_json(destination/'onnx_parity.json').get('passed'):
        raise ValueError('Baseline ONNX parity was not verified')
    if not metadata.get('training_complete'):
        raise ValueError('Use the final trained Day 2 model as baseline, not its early integration export')
    return destination


def upload_candidate(store, out, config):
    # Rejected candidates still get an immutable run folder for audit/integration.
    root = 'runs/'+config['run_id']+'/candidate/'
    objects = {}
    for name in ('signature_yolox_s.onnx','metadata.json','onnx_parity.json','farhan_handoff.zip'):
        objects[name] = store.put_file(out/'artifact'/name, root+name)
    for directory in ('reports','acceptance','evaluation/onnx','evaluation/baseline'):
        for file in sorted((out/directory).rglob('*')):
            if file.is_file():
                store.put_file(file, root+file.relative_to(out).as_posix())
    record = {'run_id': config['run_id'], 'registry_name': config['registry_name'],
              'smoke': config['smoke'], 'objects': objects}
    store.put_json(record, root+'manifest.json')
    return record


def run(config, out):
    import torch
    from bootstrap import download
    from checkpoints import recover
    from day3_data import prepare_day3_dataset
    from day3_gate import evaluation_protocol, promotion_decision
    from day3_registry import RegistryLock, Tracking
    from export_model import export_and_verify, package_handoff, validate_handoff_sources
    from onnx_evaluate import evaluate_onnx
    from pipeline import assert_fresh_run_available
    from r2_storage import R2Store
    from reports import acceptance_tests, metric_tables, observed_errors
    from training import train
    from common import resolve_device_name

    validate_handoff_sources()
    revision_file = Path(config['source_dir'])/'.source_revision'
    if not revision_file.is_file() or revision_file.read_text().strip() != YOLOX_COMMIT:
        raise RuntimeError('Pinned YOLOX source missing. Run without --runtime-ready to bootstrap it.')
    os.environ['YOLOX_SOURCE_DIR'] = config['source_dir']
    if config['source_dir'] not in sys.path:
        sys.path.insert(0, config['source_dir'])
    config['device'] = resolve_device_name(config['device'], torch.cuda.is_available()).split(':')[0]
    config['fp16'] = config['device'] == 'cuda'
    if config['device'] == 'cpu':
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
    out.mkdir(parents=True, exist_ok=True)
    store = R2Store.from_env(config['r2_prefix'])
    store.check_connection()
    status = {'run_id': config['run_id'], 'smoke': config['smoke'], 'status': 'running',
              'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    tracking = None
    champion_key = f'registry/{config["registry_name"]}/champion.json'
    with RegistryLock(store, config['registry_name'], config['run_id']):
        if not config['resume']:
            assert_fresh_run_available(out, config, store)
            if store.exists('runs/'+config['run_id']+'/tracking.json'):
                raise ValueError('Run ID already exists. Use --resume or a new --output directory name.')
        try:
            if config['resume']:
                state = recover(out, config, store)
                if state['identity']['epochs'] != config['epochs'] or state['identity']['batch_size'] != config['batch_size']:
                    raise ValueError('Resume must keep the saved total epochs and batch size')
            else:
                state = None
            write_json(out/'reports'/'day3_status.json', status)
            store.put_file(out/'reports'/'day3_status.json', f'runs/{config["run_id"]}/day3_status.json')
            # Fail early if the mandatory full-run baseline has not been published.
            champion = store.get_json(champion_key) if store.exists(champion_key) else None
            if not config['smoke']:
                if champion is None:
                    if not store.exists(config['baseline_manifest']):
                        raise FileNotFoundError('Day 2 baseline manifest missing in R2: '+
                            store.key(config['baseline_manifest'])+'. Complete Day 2 Section 13 before a full Day 3 run.')
                    baseline_record = store.get_json(config['baseline_manifest'])
                else:
                    baseline_record = champion
                download_baseline(store, baseline_record, out/'baseline')

            tracking = Tracking(store, config['registry_name'], out, config['resume'])
            marker_key = f'runs/{config["run_id"]}/connection_check.json'
            marker = {'run_id': config['run_id'], 'r2_read_write': True}
            store.put_json(marker, marker_key)
            if store.get_json(marker_key) != marker:
                raise RuntimeError('R2 read/write check failed')
            dataset, audit = prepare_day3_dataset(ROOT, config)
            config.update(dataset_dir=str(dataset), dataset_revision=audit['revision'])
            write_json(out/'config.json', config)
            write_json(out/'reports'/'dataset_audit.json', audit)
            protocol = evaluation_protocol(config, audit)
            write_json(out/'reports'/'evaluation_protocol.json', protocol)
            tracking.log_params(config)
            print('Run:', config['run_id'], '| device:', config['device'], '| smoke:', config['smoke'], flush=True)
            print('Epochs:', config['epochs'], '| batch:', config['batch_size'],
                  '| split counts:', {k:v['images'] for k,v in audit['splits'].items()}, flush=True)

            baseline_metrics = None
            if not config['smoke']:
                baseline_metrics = evaluate_onnx(out/'baseline', dataset, 'val', out/'evaluation'/'baseline', config)
                baseline_metrics['protocol_sha256'] = protocol['sha256']
                write_json(out/'evaluation'/'baseline'/'val_metrics.json', baseline_metrics)
                if champion is None:
                    # Import the already deployed baseline as version 1, with honest lineage.
                    with tracking.mlflow.start_run(experiment_id=tracking.experiment_id,
                            run_name='import-existing-Day2', nested=True):
                        tracking.mlflow.set_tags({'imported_existing_model': True, 'training_executed_here': False})
                        tracking.mlflow.log_metric('validation_mAP50', baseline_metrics['mAP50'])
                        info = tracking.log_model(out/'baseline')
                    version = tracking.register(info.model_uri, {'origin': 'existing-Day2-import'})
                    tracking.set_champion(version)
                    champion = {**baseline_record, 'registry_name': config['registry_name'],
                                'model_version': version, 'model_uri': info.model_uri,
                                'validation_metrics': baseline_metrics, 'origin': 'existing-Day2-import'}
                    store.put_json(champion, champion_key)
                else:
                    # Repair an interrupted alias update from the authoritative R2 pointer.
                    tracking.set_champion(champion['model_version'])

            initial = None
            if state is None:
                initial = Path(config['initial_checkpoint']).resolve() if config['initial_checkpoint'] else ROOT/'pretrained'/'yolox_s.pth'
                if not initial.is_file():
                    if config['initial_checkpoint']:
                        raise FileNotFoundError(initial)
                    download('https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth', initial)
            best = train(dataset, out, config, audit, initial, state, store, epoch_callback=tracking.epoch)
            artifact = export_and_verify(best, dataset, out/'artifact', config, audit, out)
            metadata = read_json(artifact/'metadata.json')
            metadata.update(smoke_run=config['smoke'], promotion_eligible=not config['smoke'])
            write_json(artifact/'metadata.json', metadata)
            # Repack so the handoff carries the same explicit smoke metadata.
            package_handoff(artifact)
            for split in ('val','test'):
                evaluate_onnx(artifact, dataset, split, out/'evaluation'/'onnx', config)
            candidate = read_json(out/'evaluation'/'onnx'/'val_metrics.json')
            candidate['protocol_sha256'] = protocol['sha256']
            write_json(out/'evaluation'/'onnx'/'val_metrics.json', candidate)
            metric_tables(out)
            observed_errors(dataset, out, config)
            acceptance = acceptance_tests(artifact, dataset, out, config)
            progress = read_json(out/'reports'/'progress.json')
            gate = promotion_decision(candidate, baseline_metrics, smoke=config['smoke'],
                complete=progress['epoch'] >= config['epochs'],
                parity=read_json(artifact/'onnx_parity.json')['passed'],
                acceptance=acceptance['passed'], min_delta=config['min_delta'])
            write_json(out/'reports'/'gate.json', gate)
            for split in ('val','test'):
                metrics = read_json(out/'evaluation'/'onnx'/(split+'_metrics.json'))
                tracking.mlflow.log_metrics({split+'_'+key: float(metrics[key])
                    for key in ('mAP50','mAP50_95','precision','recall','f1')})
            if baseline_metrics:
                tracking.mlflow.log_metric('baseline_val_mAP50', baseline_metrics['mAP50'])
            tracking.mlflow.set_tags({'gate_promote': gate['promote'],
                'gate_reasons': ','.join(gate['reasons']), 'onnx_parity_passed': True,
                'acceptance_passed': acceptance['passed']})
            tracking.mlflow.log_artifacts(str(out/'reports'), artifact_path='reports')
            tracking.mlflow.log_artifacts(str(out/'acceptance'), artifact_path='acceptance')
            tracking.mlflow.log_artifact(str(out/'config.json'))
            tracking.mlflow.log_artifacts(str(out/'artifact'), artifact_path='handoff')
            model_info = tracking.log_model(artifact)
            record = upload_candidate(store, out, config)
            record.update(gate=gate, validation_metrics=candidate, mlflow_run_id=tracking.run_id,
                          model_uri=model_info.model_uri, model_version=None)
            if gate['promote']:
                version = tracking.register(model_info.model_uri, {
                    'validation_mAP50': candidate['mAP50'], 'dataset_revision': config['dataset_revision'],
                    'model_sha256': metadata['model_sha256'], 'gate': 'passed'})
                record['model_version'] = version
                store.put_json(record, f'registry/{config["registry_name"]}/versions/{version}.json')
                # R2 pointer is the deployment source of truth; aliases are reconciled on restart.
                store.put_json(record, champion_key)
                tracking.set_champion(version)
            write_json(out/'reports'/'upload_receipt.json', record)
            store.put_json(record, f'runs/{config["run_id"]}/candidate/manifest.json')
            store.put_file(out/'reports'/'upload_receipt.json', f'runs/{config["run_id"]}/candidate/upload_receipt.json')
            tracking.mlflow.log_artifact(str(out/'reports'/'upload_receipt.json'), artifact_path='reports')
            status.update(status='smoke_completed' if config['smoke'] else ('promoted' if gate['promote'] else 'rejected'),
                          gate=gate, mlflow_run_id=tracking.run_id, candidate_uploaded=True,
                          model_version=record['model_version'], acceptance_passed=acceptance['passed'])
            tracking.finish()
            tracking = None
        except BaseException as exc:
            status.update(status='failed', error_type=type(exc).__name__)
            if tracking is not None:
                try:
                    tracking.finish('KILLED' if isinstance(exc, KeyboardInterrupt) else 'FAILED')
                except Exception:
                    print('Tracking finalization failed; the last R2 SQLite snapshot remains available.', flush=True)
            raise
        finally:
            status['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            write_json(out/'reports'/'day3_status.json', status)
            store.put_file(out/'reports'/'day3_status.json', f'runs/{config["run_id"]}/day3_status.json')
    print(json.dumps(status, indent=2), flush=True)
    if status['status'] == 'rejected' and not status['acceptance_passed']:
        raise RuntimeError('Candidate uploaded for audit, but image acceptance failed. Production model retained.')
    return out


def main(argv=None):
    args = parser().parse_args(argv)
    config, out = make_config(args)
    if args.dry_run:
        print(json.dumps({'config': config, 'output': str(out),
            'stages': ['HF dataset','training','ONNX export/parity','val/test evaluation',
                       'positive/negative tests','R2 candidate upload','strict validation gate','MLflow registry'],
            'promotion_enabled': not args.smoke}, indent=2))
        return 0
    required = ('R2_ACCOUNT_ID','R2_ACCESS_KEY_ID','R2_SECRET_ACCESS_KEY','R2_BUCKET')
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise ValueError('Set private environment variables before running: '+', '.join(missing))
    if not args.runtime_ready:
        from bootstrap import setup
        runtime = setup(ROOT, device=args.device)
        env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'YOLOX_SOURCE_DIR': runtime['source_dir'],
               'PYTHONPATH': str(ROOT)+os.pathsep+runtime['source_dir']}
        command = [runtime['python'], '-u', str(Path(__file__).resolve()),
                   *(argv if argv is not None else sys.argv[1:]), '--output', str(out), '--runtime-ready']
        result = subprocess.run(command, cwd=ROOT, env=env)
        return result.returncode
    run(config, out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
