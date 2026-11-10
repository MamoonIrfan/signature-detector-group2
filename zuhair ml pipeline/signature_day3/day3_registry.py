"""Single-writer R2 lock and durable MLflow tracking/registry snapshots.

Artifact URIs point to R2, so restored SQLite metadata has no ephemeral local paths.
The production R2 champion manifest is the deployment source of truth.
"""
from __future__ import annotations

import datetime
import json
import math
import os
import sqlite3
import uuid
from pathlib import Path

from common import read_json, safe_id, write_json


class RegistryLock:
    def __init__(self, store, registry_name, run_id):
        self.store = store
        self.relative = f'registry/{safe_id(registry_name)}/writer_lock.json'
        self.owner = uuid.uuid4().hex
        self.payload = {'owner': self.owner, 'run_id': run_id,
                        'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}

    def __enter__(self):
        try:
            self.store.client.put_object(Bucket=self.store.bucket, Key=self.store.key(self.relative),
                Body=json.dumps(self.payload).encode(), ContentType='application/json', IfNoneMatch='*')
        except Exception as exc:
            status = getattr(exc, 'response', {}).get('ResponseMetadata', {}).get('HTTPStatusCode')
            if status in (409, 412):
                raise RuntimeError('Another writer holds '+self.relative+
                    '. Inspect it in R2. Remove a stale lock only after its process has stopped.') from None
            raise self.store._error(exc, 'Registry lock acquisition') from None
        return self

    def __exit__(self, *args):
        current = self.store.get_json(self.relative)
        if current.get('owner') != self.owner:
            raise RuntimeError('Registry lock ownership changed; refusing to delete it')
        self.store.client.delete_object(Bucket=self.store.bucket, Key=self.store.key(self.relative))


def configure_artifact_credentials(store):
    # MLflow's S3 artifact repository uses the same private R2 credentials.
    os.environ['AWS_ACCESS_KEY_ID'] = os.environ['R2_ACCESS_KEY_ID']
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['R2_SECRET_ACCESS_KEY']
    os.environ['AWS_DEFAULT_REGION'] = 'auto'
    os.environ['AWS_REQUEST_CHECKSUM_CALCULATION'] = 'when_required'
    os.environ['AWS_RESPONSE_CHECKSUM_VALIDATION'] = 'when_required'
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = getattr(store.client.meta, 'endpoint_url', None) or (
        'https://'+os.environ['R2_ACCOUNT_ID']+'.r2.cloudflarestorage.com')
    # Do not accidentally combine these static credentials with a host's session token.
    os.environ.pop('AWS_SESSION_TOKEN', None)


class Tracking:
    def __init__(self, store, name, out, resume=False):
        import mlflow
        from mlflow.tracking import MlflowClient
        self.mlflow, self.store, self.name, self.out = mlflow, store, name, Path(out)
        self.root = f'registry/{name}'
        self.db = self.out/'tracking'/'mlflow.db'
        self.db.parent.mkdir(parents=True, exist_ok=True)
        if store.exists(self.root+'/mlflow_state.json'):
            store.download_verified(store.get_json(self.root+'/mlflow_state.json')['database'], self.db)
        elif self.db.is_file():
            raise RuntimeError('Local MLflow DB exists without a remote state. Use a clean output directory.')
        configure_artifact_credentials(store)
        self.uri = 'sqlite:///'+self.db.as_posix()
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_registry_uri(self.uri)
        self.client = MlflowClient(tracking_uri=self.uri, registry_uri=self.uri)
        experiment_name = 'signature-day3-'+name
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = self.client.create_experiment(experiment_name,
                artifact_location=f's3://{store.bucket}/{store.prefix}/tracking/{name}/artifacts')
        else:
            self.experiment_id = experiment.experiment_id
        self.remote_run = f'runs/{out.name}/tracking.json'
        if resume and store.exists(self.remote_run):
            old = store.get_json(self.remote_run)
            run = mlflow.start_run(run_id=old['mlflow_run_id'])
        else:
            run = mlflow.start_run(experiment_id=self.experiment_id, run_name=out.name)
        self.run_id = run.info.run_id
        store.put_json({'mlflow_run_id': self.run_id, 'registry_name': name,
                        'day3_run_id': out.name}, self.remote_run)
        write_json(self.out/'reports'/'tracking.json', {'mlflow_run_id': self.run_id,
                   'tracking_uri': self.uri, 'registry_name': name})
        self.snapshot('started')

    def snapshot(self, label):
        snapshot = self.out/'tracking'/'mlflow_snapshot.sqlite'
        with sqlite3.connect(str(self.db)) as source, sqlite3.connect(str(snapshot)) as target:
            source.backup(target)
        key = self.root+f'/snapshots/{self.out.name}/{label}-{uuid.uuid4().hex}.sqlite'
        record = self.store.put_file(snapshot, key)
        self.store.put_json({'database': record, 'label': label,
                            'run_id': self.out.name}, self.root+'/mlflow_state.json')

    def log_params(self, config):
        keys = ('run_id','dataset_repo','dataset_revision','epochs','batch_size','img_size',
                'seed','workers','device','fp16','conf_threshold','nms_threshold','smoke',
                'yolox_commit','model_size','baseline_manifest','min_delta')
        # Device may change on resume, so log it as a tag rather than an immutable param.
        self.mlflow.log_params({key: config[key] for key in keys if key not in ('device','workers','fp16')})
        self.mlflow.set_tags({'current_device': config['device'], 'workers': config['workers'],
                             'fp16': config['fp16'], 'smoke_run': config['smoke']})

    def epoch(self, record):
        values = {key: float(value) for key, value in record.items()
                  if key != 'epoch' and isinstance(value, (int, float)) and math.isfinite(value)}
        self.mlflow.log_metrics({'epoch_'+key: value for key, value in values.items()}, step=record['epoch'])
        self.mlflow.log_artifact(str(self.out/'reports'/'history.json'), artifact_path='reports')
        self.snapshot(f'epoch-{record["epoch"]:04d}')

    def log_model(self, artifact, path='model'):
        import onnx
        import mlflow.onnx as onnx_flavor
        return onnx_flavor.log_model(onnx_model=onnx.load(str(Path(artifact)/'signature_yolox_s.onnx')),
            artifact_path=path, onnx_execution_providers=['CPUExecutionProvider'],
            pip_requirements=['onnx==1.17.0', 'onnxruntime==1.20.1', 'numpy==1.26.4'])

    def register(self, source, tags):
        version = self.mlflow.register_model(source, self.name)
        for key, value in tags.items():
            self.client.set_model_version_tag(self.name, version.version, key, str(value))
        self.snapshot('registered-'+str(version.version))
        return str(version.version)

    def set_champion(self, version):
        self.client.set_registered_model_alias(self.name, 'champion', str(version))
        self.snapshot('champion-'+str(version))

    def finish(self, status='FINISHED'):
        self.mlflow.end_run(status=status)
        self.snapshot(status.lower())
