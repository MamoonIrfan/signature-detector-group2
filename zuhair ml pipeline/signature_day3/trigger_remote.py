"""Farhan's backend: trigger the included CPU workflow without shell commands."""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request


def dispatch(repo, *, revision, epochs=2, ref='main', token=None, opener=urllib.request.urlopen):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}', repo):
        raise ValueError('repo must be OWNER/REPOSITORY')
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 3:
        raise ValueError('CPU remote trials allow 1-3 epochs')
    if not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        raise ValueError('Use an immutable 40-character HF dataset commit SHA')
    token = token or os.environ.get('GH_TRAINING_TOKEN')
    if not token:
        raise ValueError('Set GH_TRAINING_TOKEN privately in the backend environment')
    workflow = 'signature-day3.yml'
    url = f'https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches'
    payload = {'ref': ref, 'inputs': {'epochs': str(epochs), 'dataset_revision': revision}}
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method='POST', headers={
        'Authorization': 'Bearer '+token, 'Accept': 'application/vnd.github+json',
        'Content-Type': 'application/json', 'X-GitHub-Api-Version': '2022-11-28',
        'User-Agent': 'signature-day3-trigger'})
    try:
        with opener(request, timeout=30) as response:
            status = response.status
            response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'GitHub dispatch failed (HTTP {exc.code}). Check token permissions, workflow, repo and ref.') from None
    if status not in (200, 201, 202, 204):
        raise RuntimeError('Unexpected GitHub response: '+str(status))
    return {'status': 'trigger_accepted', 'http_status': status,
            'workflow_url': f'https://github.com/{repo}/actions/workflows/{workflow}',
            'training_completion_verified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--dataset-revision', default='a55352328a1fbc3442b54db0e377f5add470bcba')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--ref', default='main')
    args = parser.parse_args()
    print(json.dumps(dispatch(args.repo, revision=args.dataset_revision, epochs=args.epochs, ref=args.ref), indent=2))
