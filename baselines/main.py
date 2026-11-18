"""One-command, resumable external architecture benchmark for the frozen study."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import data
from models import make_model
from training import train_model, load_model, predict

HERE = Path(__file__).resolve().parent
MODELS = ('MLP', 'GRU', 'TCN', 'TRANSFORMER')
SEEDS = (2026, 2027, 2028)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def archive(output, suffix='return'):
    target = output.parent / (output.name + '_' + suffix + '.zip')
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file() and not path.name.endswith('.tmp'):
                z.write(path, path.relative_to(output.parent))
    temporary.replace(target)
    return target


class Tee:
    def __init__(self, terminal, stream):
        self.terminal, self.stream = terminal, stream
    def write(self, text):
        self.terminal.write(text)
        self.stream.write(text)
        self.stream.flush()
    def flush(self):
        self.terminal.flush()
        self.stream.flush()
    def isatty(self):
        return False


def environment(device):
    info = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                numpy=np.__version__, device=str(device),
                cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
    if device.type == 'cuda':
        info.update(gpu=torch.cuda.get_device_name(device),
                    compute_capability=torch.cuda.get_device_capability(device),
                    torch_architectures=torch.cuda.get_arch_list())
        # Exercise a kernel before a long experiment, including on Blackwell GPUs.
        a = torch.ones((32, 32), device=device)
        _ = a @ a
        torch.cuda.synchronize(device)
    return info


def model_config(name, width, maximum_prefix, input_dim):
    # A two-convolution residual block with k=3 has RF=1+4*(2**levels-1).
    levels = max(4, math.ceil(math.log2((maximum_prefix-1)/4+1)))
    return dict(input_dim=int(input_dim), width=width, hidden_dim=width, dropout=.1,
                num_layers=levels if name == 'TCN' else (2 if name in ('MLP', 'GRU') else 4),
                num_heads=4, kernel_size=3,
                num_classes=10221, max_prefix_length=maximum_prefix,
                use_cudnn=False if name == 'GRU' else True)


def package_sources(output):
    target = output / 'source'
    target.mkdir(exist_ok=True)
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in ('.py', '.json', '.txt', '.md', '.ps1'):
            shutil.copy2(path, target / path.name)


def attach_records(prediction, dataset):
    out = {k:np.asarray(v) for k,v in prediction.items() if isinstance(v, (np.ndarray, list, tuple))}
    index = np.asarray(prediction['index'], dtype=np.int64)
    np.testing.assert_array_equal(index, np.arange(len(dataset)))
    for key in ('series_id', 'round_id', 'bins', 'marks', 'time', 'cutoff_index'):
        out[key] = np.asarray(dataset.records[key])[index]
    expected = np.where(out['bins'] == 20, 10220, out['bins']*511+out['marks']-1)
    np.testing.assert_array_equal(prediction['y'], expected)
    if not np.isfinite(out['nll']).all() or not np.isfinite(out['time_nll']).all():
        raise ValueError('Non-finite forecast scores')
    np.testing.assert_allclose(out['time_mass'].sum(1)+out['tail'], 1., atol=2e-5, rtol=2e-5)
    return out


def save_prediction(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.npz.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def run(args, output):
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = data.verify_cache(args.data)
    manifest_hash = sha(args.data / 'manifest.json')
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    torch.set_float32_matmul_precision('high')
    write_json(output / 'environment.json', environment(device))
    package_sources(output)
    shutil.copy2(args.data / 'manifest.json', output / 'data_manifest.json')
    if args.verify_only:
        print('Data and GPU checks passed. No training performed.')
        return

    protocol = dict(config=config, data_manifest_sha256=manifest_hash,
                    sources={p.name:sha(p) for p in HERE.glob('*.py')},
                    seeds=list(SEEDS), models=list(MODELS),
                    existing_paper_test_scores_already_known=True,
                    baseline_selection='Internal 22/8 series only; test not used for hyperparameters',
                    temporal_support='20 x 511 nonempty sets plus one tail',
                    evaluation='cutoff -> round -> series; seed-score mean',
                    mode='smoke' if args.smoke else 'formal')
    protocol['fingerprint'] = canonical(protocol)
    protocol_path = output / 'run_protocol.json'
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding='utf-8'))
        if old['fingerprint'] != protocol['fingerprint']:
            raise ValueError('Output belongs to different code/data/config. Choose another --output directory.')
    else:
        write_json(protocol_path, protocol)
    if args.report_only:
        from report import generate_report
        generate_report(output, args.references, SEEDS, MODELS)
        print('Report regenerated:', output / 'summary.json')
        return

    # Keep prior status files for audit, but never leave an old success marker
    # in place while a resumed run is still in progress.
    for filename in ('FAILED.json', 'COMPLETED.json', 'SMOKE_OK.json'):
        previous = output / filename
        if previous.exists():
            history = output / 'status_history'
            history.mkdir(exist_ok=True)
            previous.replace(history / (str(time.time_ns()) + '_' + filename))

    limit = 48 if args.smoke else None
    fit = data.CutoffDataset(args.data, split='inner_fit', max_examples=limit)
    held = data.CutoffDataset(args.data, split='inner_val', max_examples=24 if args.smoke else None)
    maximum_prefix = max(int(getattr(fit, 'max_prefix_length', 0)),
                         int(manifest.get('max_prefix_length', manifest.get('max_round_frames', 0))), 1)
    widths = [32] if args.smoke else config['widths']
    training_config = dict(config['training'])
    if args.smoke:
        training_config.update(max_epochs=2, min_epochs=1, patience=2,
                               batch_size=4, eval_batch_size=4, amp=False, allow_cpu_full=True)
    print('Mode:', protocol['mode'], '| device:', device)
    print('Internal fitting:', len(fit), '| internal validation:', len(held))
    selected = {}
    for name in MODELS:
        candidates = []
        for width in widths:
            spec = model_config(name, int(width), maximum_prefix, manifest['dimensions'])
            directory = output / 'tuning' / f'{name}_w{width}'
            print(f'\nTUNING {name} width={width}')
            result = train_model(fit, held, name, spec, directory, 2026, training_config, device, resume=True)
            model = load_model(result['checkpoint_path'], device)
            prediction = predict(model, data.make_loader(held, training_config['eval_batch_size'], False, 2026), device)
            score = data.aggregate(held, prediction['index'], prediction['nll'])['macro_nll']
            candidates.append(dict(model_config=spec, selected_epoch=int(result['selected_epoch']),
                                   internal_joint_nll=float(score), width=width))
            del model, prediction
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        selected[name] = min(candidates, key=lambda c:(c['internal_joint_nll'], c['width']))
        write_json(output / 'tuning' / f'{name}_candidates.json', candidates)
    # All baseline hyperparameters/epochs are locked before any final-cohort scoring.
    write_json(output / 'selected_configurations.json', dict(
        fingerprint=canonical(selected), choices=selected,
        protocol_fingerprint=protocol['fingerprint'], selected_without_test_predictions=True))
    print('\nAll four model configurations locked. Beginning final refits.')

    full_fit = data.CutoffDataset(args.data, split='train', max_examples=limit)
    seeds = (2026,) if args.smoke else SEEDS
    for name in MODELS:
        choice = selected[name]
        for seed in seeds:
            train_config = dict(training_config, epochs=choice['selected_epoch'],
                                max_epochs=choice['selected_epoch'])
            print(f'\nFINAL FIT {name} seed={seed} epochs={choice["selected_epoch"]}')
            directory = output / 'final' / f'{name}_seed{seed}'
            result = train_model(full_fit, None, name, choice['model_config'], directory,
                                 seed, train_config, device, resume=True)
            model = load_model(result['checkpoint_path'], device)
            parameters = sum(p.numel() for p in model.parameters())
            for cohort, role in (('dev', 'validation'), ('rio', 'test')):
                if args.smoke and cohort == 'rio':
                    continue
                dataset = data.CutoffDataset(args.data, split=role, max_examples=24 if args.smoke else None)
                path = output / 'predictions' / cohort / f'{name}_seed{seed}.npz'
                meta_path = path.with_suffix('.json')
                checkpoint_hash = sha(result['checkpoint_path'])
                if path.exists() and meta_path.exists():
                    old = json.loads(meta_path.read_text(encoding='utf-8'))
                    if old.get('checkpoint_sha256') == checkpoint_hash and old.get('prediction_sha256') == sha(path):
                        print('Verified prediction reuse:', path.name, cohort)
                        continue
                loader = data.make_loader(dataset, training_config['eval_batch_size'], False, seed)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                tick = time.perf_counter()
                prediction = predict(model, loader, device)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter()-tick
                arrays = attach_records(prediction, dataset)
                save_prediction(path, arrays)
                write_json(meta_path, dict(model=name, seed=seed, parameters=parameters,
                           selected_epoch=choice['selected_epoch'], model_config=choice['model_config'],
                           prediction_examples=len(dataset), inference_seconds=seconds,
                           inference_timing='End-to-end DataLoader + transfers + metrics, no model fit',
                           checkpoint_sha256=checkpoint_hash, prediction_sha256=sha(path),
                           protocol_fingerprint=protocol['fingerprint']))
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    if args.smoke:
        write_json(output / 'SMOKE_OK.json', dict(status='smoke_only', models=list(MODELS),
                   seconds=time.perf_counter()-started, formal_results=False))
        print('Smoke passed for four models. This is not a formal experiment result.')
        print('Smoke archive:', archive(output, 'smoke'))
        return
    from report import generate_report
    generate_report(output, args.references, SEEDS, MODELS)
    required = [output / 'predictions' / c / f'{m}_seed{s}.npz'
                for c in ('dev', 'rio') for m in MODELS for s in SEEDS]
    if not all(p.is_file() for p in required):
        raise RuntimeError('Incomplete baseline/seed/cohort outputs')
    files = {str(p.relative_to(output)):dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ('COMPLETED.json', 'console.log')}
    write_json(output / 'COMPLETED.json', dict(status='complete', formal_results=True,
               prediction_files=len(required), final_training_runs=12, models=list(MODELS),
               seeds=list(SEEDS), elapsed_seconds=time.perf_counter()-started,
               protocol_fingerprint=protocol['fingerprint'], files=files,
               unhashed_live_files=['console.log']))
    print('\nFORMAL STUDY COMPLETED. Return this archive:', archive(output))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=HERE/'config.json')
    ap.add_argument('--data', type=Path, default=HERE/'data/cache')
    ap.add_argument('--references', type=Path, default=HERE.parent/'outputs/references',
                    help='Frozen SPARC-CS2 reference predictions; see export_references.py')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--device', default='auto')
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument('--smoke', action='store_true')
    modes.add_argument('--verify-only', action='store_true')
    modes.add_argument('--report-only', action='store_true')
    args = ap.parse_args()
    output = (args.output or HERE/('results_smoke' if args.smoke else 'results')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'console.log').open('a', encoding='utf-8') as stream:
        original_out, original_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(original_out, stream), Tee(original_err, stream)
        try:
            run(args, output)
        except Exception as exc:
            traceback.print_exc()
            write_json(output/'FAILED.json', dict(status='failed', error=repr(exc), traceback=traceback.format_exc()))
            print('Partial diagnostic archive:', archive(output, 'partial'))
            raise SystemExit(1)
        finally:
            sys.stdout, sys.stderr = original_out, original_err


if __name__ == '__main__':
    main()
