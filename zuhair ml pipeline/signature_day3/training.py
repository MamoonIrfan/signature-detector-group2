"""Complete YOLOX-S fine-tuning, evaluation each epoch, and durable resume."""
from __future__ import annotations
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from common import identity, read_json, write_json, resolve_device_name
from modeling import build_model, load_state, warm_start
from data_loading import SignatureDataset, make_loader
from evaluation import evaluate
from checkpoints import backup_epoch


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state, device):
    random.setstate(state['python'])
    kind, keys, pos, gaussian, cached = state['numpy']
    np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda' and state.get('cuda') is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def train(dataset, out, config, audit, initial=None, resumed=None, store=None, epoch_callback=None):
    from tqdm.auto import tqdm
    from yolox.utils import ModelEMA
    out = Path(out)
    device = torch.device(resolve_device_name(config['device'], torch.cuda.is_available()))
    amp = config.get('fp16', True) and device.type == 'cuda'
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    else:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
    # Epoch-boundary recovery is reproducible to practical tolerances, not bitwise guaranteed.
    torch.backends.cudnn.benchmark = False
    model = build_model('s').to(device)
    if resumed is None:
        if initial is None:
            raise ValueError('An official pretrained or prior signature checkpoint is required')
        transfer = warm_start(model, initial)
        write_json(out/'reports'/'initialization.json', transfer)
    groups = {'decay': [], 'no_decay': []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups['no_decay' if parameter.ndim == 1 or name.endswith('.bias') else 'decay'].append(parameter)
    max_lr = 0.01 / 64 * config['batch_size']
    optimizer = torch.optim.SGD([
        {'params': groups['no_decay'], 'weight_decay': 0.0},
        {'params': groups['decay'], 'weight_decay': 0.0005}],
        lr=max_lr, momentum=0.9, nesterov=True)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    ema = ModelEMA(model, decay=0.9998)
    training = SignatureDataset(dataset, 'train', 640, training=True)
    validation = SignatureDataset(dataset, 'val', 640)
    if not training.coco['annotations'] or not validation.coco['annotations']:
        raise ValueError('Train and validation splits both need signature annotations')
    epochs = config['epochs']
    steps = math.ceil(len(training)/config['batch_size'])
    total_steps = steps*epochs
    warmup_steps = steps*min(3, max(0, epochs-1))
    no_aug_epochs = min(5, max(1, epochs//5))
    run_identity = identity(config, audit)
    latest, best = out/'checkpoints'/'latest.pth', out/'checkpoints'/'best.pth'
    start, best_ap, best_epoch, history = 0, -1.0, 0, []
    if resumed is not None:
        if resumed['identity'] != run_identity:
            raise ValueError('Resume configuration/dataset mismatch. Keep the saved settings or start a new RUN_ID.')
        model.load_state_dict(resumed['training_model'], strict=True)
        ema.ema.load_state_dict(resumed['model'], strict=True)
        ema.updates = resumed['ema_updates']
        optimizer.load_state_dict(resumed['optimizer'])
        if amp and resumed.get('scaler'):
            scaler.load_state_dict(resumed['scaler'])
        start, best_ap, best_epoch = resumed['next_epoch'], resumed['best_ap'], resumed['best_epoch']
        history = resumed['history']
        restore_rng(resumed['rng'], device)
        write_json(out/'reports'/'history.json', history)
        write_json(out/'reports'/'progress.json',
                   {'epoch': start, 'epochs': epochs, 'best_epoch': best_epoch, 'best_val_mAP50': best_ap})
        # Retry the most recent backup before spending another epoch on training.
        backup_epoch(out, config, start, run_identity, store)
    for epoch in range(start, epochs):
        started = time.perf_counter()
        training.augment = epoch < epochs-no_aug_epochs
        model.head.use_l1 = not training.augment
        ema.ema.head.use_l1 = model.head.use_l1
        model.train()
        sums, batches = {}, 0
        loader = make_loader(training, config['batch_size'], config['workers'], seed, epoch)
        bar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}')
        for iteration, (images, targets) in enumerate(bar):
            step = epoch*steps+iteration
            if warmup_steps and step < warmup_steps:
                lr = max_lr*((step+1)/warmup_steps)**2
            else:
                fraction = (step-warmup_steps)/max(1, total_steps-warmup_steps-1)
                lr = max_lr*(0.05+0.95*(1+math.cos(math.pi*min(1, max(0, fraction))))/2)
            for group in optimizer.param_groups:
                group['lr'] = lr
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(images, targets)
                loss = outputs['total_loss']
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss. Inspect data/boxes before resuming.')
            previous_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                ema.update(model)
            for name, value in outputs.items():
                value = float(value.detach().item()) if torch.is_tensor(value) else float(value)
                sums[name] = sums.get(name, 0.0)+value
            batches += 1
            if (iteration+1) % 20 == 0:
                bar.set_postfix(loss=f'{float(loss.detach()):.4f}', lr=f'{lr:.6f}')
        metrics = evaluate(ema.ema, validation, config, out/'evaluation'/'during_training', 'val')
        improved = metrics['mAP50'] > best_ap
        if improved:
            best_ap, best_epoch = metrics['mAP50'], epoch+1
            save_torch(best, {'model': cpu_state(ema.ema), 'identity': run_identity,
                              'epoch': best_epoch, 'validation': metrics})
        record = {'epoch': epoch+1, 'loss': sums['total_loss']/batches,
                  'learning_rate': lr, 'seconds': time.perf_counter()-started, **metrics}
        history.append(record)
        write_json(out/'reports'/'history.json', history)
        write_json(out/'reports'/'progress.json', {'epoch': epoch+1, 'epochs': epochs,
                   'best_epoch': best_epoch, 'best_val_mAP50': best_ap})
        save_torch(latest, {'training_model': cpu_state(model), 'model': cpu_state(ema.ema),
                            'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                            'ema_updates': int(ema.updates), 'identity': run_identity,
                            'next_epoch': epoch+1, 'best_ap': float(best_ap), 'best_epoch': best_epoch,
                            'history': history, 'rng': rng_state()})
        backup_epoch(out, config, epoch+1, run_identity, store)
        if epoch_callback is not None:
            epoch_callback(record)
        print(f"val mAP@0.5={metrics['mAP50']:.4f} precision={metrics['precision']:.4f} "
              f"recall={metrics['recall']:.4f} | best epoch={best_epoch}", flush=True)
        # Optional early handoff after a real completed epoch; full training then continues.
        if config.get('early_export_epoch', 1) == epoch+1:
            from export_model import export_and_verify
            early = out/'early_handoff'
            saved_rng = rng_state()
            try:
                export_and_verify(best, dataset, early, config, audit, out)
                print('EARLY HANDOFF READY:', early/'farhan_handoff.zip', flush=True)
            except Exception as exc:
                write_json(out/'reports'/'early_export_failure.json',
                           {'epoch': epoch+1, 'error_type': type(exc).__name__, 'message': str(exc)})
                print('Early export failed; full training continues. See early_export_failure.json.', flush=True)
            finally:
                restore_rng(saved_rng, device)
    if not best.is_file():
        raise RuntimeError('No evaluated best checkpoint exists')
    return best
