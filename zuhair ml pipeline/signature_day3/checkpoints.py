"""Each completed epoch is committed to Drive and/or R2 before training continues.

The latest pointer is written only after all checkpoint objects are durable.
An interrupted pointer write leaves the preceding completed backup recoverable.
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path
from common import read_json, write_json, sha256_file, safe_relative


def copy_atomic(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.part')
    shutil.copyfile(source, temporary)
    if sha256_file(source) != sha256_file(temporary):
        raise IOError('Checkpoint copy hash mismatch')
    os.replace(temporary, destination)
    return {'file': destination.name, 'sha256': sha256_file(destination),
            'size_bytes': destination.stat().st_size}


def backup_epoch(out, config, epoch, run_identity, store=None):
    out = Path(out)
    latest, best = out/'checkpoints'/'latest.pth', out/'checkpoints'/'best.pth'
    best_epoch = read_json(out/'reports'/'progress.json')['best_epoch']
    record = {'epoch': epoch, 'identity': run_identity, 'best_epoch': best_epoch}
    successful = []
    failures = []
    if config.get('drive_dir'):
        try:
            remote = Path(config['drive_dir'])/config['run_id']
            remote.mkdir(parents=True, exist_ok=True)
            record_drive = {**record, 'latest': copy_atomic(latest, remote/f'epoch_{epoch:04d}.pth')}
            best_destination = remote/f'best_epoch_{best_epoch:04d}.pth'
            if not best_destination.exists():
                record_drive['best'] = copy_atomic(best, best_destination)
            else:
                record_drive['best'] = {'file': best_destination.name,
                                        'sha256': sha256_file(best_destination)}
            for filename in ('config.json', 'reports/history.json', 'reports/progress.json',
                             'reports/dataset_audit.json'):
                source = out/filename
                if source.is_file():
                    copy_atomic(source, remote/filename)
            write_json(remote/f'epoch_{epoch:04d}.json', record_drive)
            write_json(remote/'latest.json', record_drive)
            successful.append('Drive')
        except Exception as exc:
            failures.append({'destination': 'Drive', 'error_type': type(exc).__name__})
    if store is not None:
        try:
            root = 'runs/'+config['run_id']+'/checkpoints/'
            previous_path = out/'reports'/'r2_backup.json'
            previous = read_json(previous_path) if previous_path.is_file() else {}
            latest_object = store.put_file(latest, root+f'epoch_{epoch:04d}.pth')
            best_object = previous.get('best') if previous.get('best_epoch') == best_epoch else None
            if not best_object:
                best_object = store.put_file(best, root+f'best_epoch_{best_epoch:04d}.pth')
            remote_record = {**record, 'latest': latest_object, 'best': best_object}
            store.put_file(out/'config.json', 'runs/'+config['run_id']+'/config.json')
            store.put_file(out/'reports'/'history.json', root+'history.json')
            store.put_json(remote_record, root+f'epoch_{epoch:04d}.json')
            store.put_json(remote_record, root+'latest.json')
            write_json(previous_path, remote_record)
            successful.append('R2')
        except Exception as exc:
            failures.append({'destination': 'R2', 'error_type': type(exc).__name__})
    write_json(out/'reports'/'backup_status.json',
               {**record, 'saved_to': successful, 'failures': failures})
    if config.get('require_r2_backup') and 'R2' not in successful:
        raise RuntimeError('Required R2 checkpoint was not committed. Local/Drive files were retained. Fix R2 and resume before another epoch.')
    if not successful:
        raise RuntimeError('No remote checkpoint was committed at this epoch. Local latest.pth is intact. Fix backup and resume.')
    if failures:
        print('Backup issue:', failures, '| checkpoint survived in:', successful, flush=True)
    print(f'Epoch {epoch} checkpoint verified: '+', '.join(successful), flush=True)


def recover(out, config, store=None):
    """Select the newest verified full checkpoint; never silently start from zero."""
    from modeling import load_state
    out = Path(out)
    latest, best = out/'checkpoints'/'latest.pth', out/'checkpoints'/'best.pth'
    candidates = []
    if latest.is_file():
        local = load_state(latest)
        candidates.append((int(local.get('next_epoch', -1)), 'local', local))
    if config.get('drive_dir'):
        remote = Path(config['drive_dir'])/config['run_id']
        if (remote/'latest.json').is_file():
            record = read_json(remote/'latest.json')
            candidates.append((int(record['epoch']), 'drive', record))
    if store:
        try:
            record = store.get_json('runs/'+config['run_id']+'/checkpoints/latest.json')
            candidates.append((int(record['epoch']), 'r2', record))
        except Exception:
            if not candidates:
                raise
            print('R2 manifest unavailable; inspecting surviving local/Drive checkpoint.', flush=True)
    if not candidates:
        raise FileNotFoundError('No full Day 2 checkpoint for this run. Use the same RUN_ID and backup destination.')
    epoch, source, record = max(candidates, key=lambda row: row[0])
    if source == 'drive':
        for key, destination in (('latest', latest), ('best', best)):
            item = record[key]
            path = remote.joinpath(*safe_relative(item['file']).parts)
            if sha256_file(path) != item['sha256']:
                raise ValueError('Drive checkpoint hash mismatch')
            copy_atomic(path, destination)
    elif source == 'r2':
        store.download_verified(record['latest'], latest)
        store.download_verified(record['best'], best)
        write_json(out/'reports'/'r2_backup.json', record)
    if source == 'local' and not best.is_file():
        # A reset may leave local latest but remove best. Restore the paired best
        # from a completed backup before proceeding; never rebase silently.
        for _, backup_source, backup_record in sorted(candidates, key=lambda row: -row[0]):
            if backup_source == 'drive' and backup_record.get('best'):
                item = backup_record['best']
                path = remote.joinpath(*safe_relative(item['file']).parts)
                if sha256_file(path) != item['sha256']:
                    raise ValueError('Drive best checkpoint hash mismatch')
                copy_atomic(path, best)
                break
            if backup_source == 'r2' and backup_record.get('best'):
                store.download_verified(backup_record['best'], best)
                break
    state = load_state(latest)
    required = ('training_model', 'model', 'optimizer', 'scaler', 'identity', 'next_epoch', 'ema_updates')
    if any(key not in state for key in required) or state['identity'].get('schema') != 'day2-v1':
        raise ValueError('Use Day 2 latest/epoch checkpoint for resume; older best.pth is a warm start only')
    if state['next_epoch'] != epoch:
        raise ValueError('Checkpoint and backup manifest epoch disagree')
    if not best.is_file():
        raise FileNotFoundError('Matching best.pth is missing; restore its backup before resuming')
    if load_state(best).get('identity') != state['identity']:
        raise ValueError('Best and latest checkpoints belong to different datasets/configurations')
    if load_state(best).get('epoch') != state.get('best_epoch'):
        raise ValueError('The recovered best weights do not match latest.pth best_epoch; restore its paired backup')
    config['dataset_revision'] = state['identity']['dataset_revision']
    write_json(out/'reports'/'recovery.json', {'source': source, 'epoch': epoch})
    print('Recovered full checkpoint after epoch', epoch, 'from', source, flush=True)
    return state
