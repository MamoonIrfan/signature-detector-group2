"""Download the latest MLflow SQLite snapshot for a local read-only review session."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from common import safe_id
from day3_registry import configure_artifact_credentials
from r2_storage import R2Store

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry-name', default='signature-yolox-s-smoke')
    parser.add_argument('--output', default='tracking_review/mlflow.db')
    parser.add_argument('--r2-prefix', default='signature-detection-day2')
    parser.add_argument('--serve', action='store_true', help='Run a local review UI on port 5000')
    args = parser.parse_args()
    safe_id(args.registry_name)
    store = R2Store.from_env(args.r2_prefix)
    record = store.get_json(f'registry/{args.registry_name}/mlflow_state.json')['database']
    output = Path(args.output).resolve()
    store.download_verified(record, output)
    configure_artifact_credentials(store)
    print('Review database:', output)
    print('Run MLflow locally in this same environment; this copy does not write back to R2.')
    print('Tracking URI: sqlite:///'+output.as_posix())
    if args.serve:
        subprocess.run([sys.executable,'-m','mlflow','ui','--backend-store-uri',
                        'sqlite:///'+output.as_posix(),'--host','127.0.0.1','--port','5000'],check=True)
