"""Device check + micro-benchmark for the CNN training path.

Verifies that the Arc 140V iGPU (XPU) is usable and measures CPU vs XPU
throughput on a synthetic batch shaped like the real data (80, 13, 21).

Runs on a CPU-only install too (XPU section is skipped), so it doubles as a
sanity check now and a benchmark once the XPU build of torch is installed.

Usage:
    python -m Models.cnn.device_benchmark
    python -m Models.cnn.device_benchmark --batch 128 --steps 50
"""
import argparse
import time

import torch

from Models.cnn.cnn import CNN

try:
    import intel_extension_for_pytorch as ipex
    _HAS_IPEX = True
except ImportError:
    ipex = None
    _HAS_IPEX = False


def available_devices():
    """Return the list of devices to benchmark, in preference order."""
    devices = []
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        devices.append(torch.device('xpu'))
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))
    devices.append(torch.device('cpu'))
    return devices


def sync(device):
    """Block until queued work on the device finishes (for honest timing)."""
    if device.type == 'xpu':
        torch.xpu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def benchmark(device, batch=64, steps=50, warmup=5, use_ipex=True):
    """Time a forward+backward+step loop on synthetic data. Returns ms/step."""
    torch.manual_seed(0)
    model = CNN().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    applied_ipex = False
    if use_ipex and _HAS_IPEX and device.type in ('xpu', 'cpu'):
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
        applied_ipex = True

    # Fixed synthetic batch matching the real input/target shapes.
    X = torch.randn(batch, 80, 13, 21, device=device)
    Y = torch.randn(batch, 6, device=device)

    model.train()
    last_loss = None
    for i in range(warmup + steps):
        if i == warmup:
            sync(device)
            t0 = time.time()
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()
        last_loss = loss
    sync(device)
    elapsed = time.time() - t0

    ms_per_step = elapsed / steps * 1000.0
    return {
        'ms_per_step': ms_per_step,
        'final_loss': float(last_loss.detach().cpu()),
        'ipex': applied_ipex,
    }


def correctness_check(batch=8):
    """If XPU is present, confirm one forward pass matches CPU within tolerance."""
    if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        print('[parity] XPU not available — skipping CPU/XPU parity check.')
        return

    torch.manual_seed(0)
    model = CNN().eval()
    X = torch.randn(batch, 80, 13, 21)

    with torch.no_grad():
        out_cpu = model(X)
        out_xpu = model.to('xpu')(X.to('xpu')).cpu()

    max_diff = (out_cpu - out_xpu).abs().max().item()
    ok = torch.allclose(out_cpu, out_xpu, atol=1e-3, rtol=1e-3)
    print(f'[parity] max |CPU - XPU| = {max_diff:.2e}  -> {"PASS" if ok else "FAIL"}')


def main():
    parser = argparse.ArgumentParser(description='CNN device check + benchmark')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--no-ipex', action='store_true', help='disable ipex.optimize')
    args = parser.parse_args()

    print('=' * 60)
    print('Environment')
    print('=' * 60)
    print(f'torch           : {torch.__version__}')
    print(f'IPEX available  : {_HAS_IPEX}' + (f' ({ipex.__version__})' if _HAS_IPEX else ''))
    has_xpu = hasattr(torch, 'xpu') and torch.xpu.is_available()
    print(f'XPU available   : {has_xpu}')
    if has_xpu:
        for i in range(torch.xpu.device_count()):
            print(f'  xpu:{i}        : {torch.xpu.get_device_name(i)}')
    print(f'CUDA available  : {torch.cuda.is_available()}')
    print()

    devices = available_devices()
    print('=' * 60)
    print(f'Benchmark  (batch={args.batch}, steps={args.steps}, warmup={args.warmup})')
    print('=' * 60)
    results = {}
    for device in devices:
        res = benchmark(device, batch=args.batch, steps=args.steps,
                        warmup=args.warmup, use_ipex=not args.no_ipex)
        results[device.type] = res
        tag = ' +ipex' if res['ipex'] else ''
        print(f'{device.type:>5}{tag:<6} : {res["ms_per_step"]:8.1f} ms/step   '
              f'(final loss {res["final_loss"]:.4f})')
    print()

    # Speedup summary vs CPU, plus a full-epoch extrapolation (280k-sample train
    # split at the given batch size).
    if 'cpu' in results:
        cpu_ms = results['cpu']['ms_per_step']
        batches_per_epoch = 280000 // args.batch
        print('=' * 60)
        print('Summary')
        print('=' * 60)
        for dtype, res in results.items():
            speedup = cpu_ms / res['ms_per_step']
            epoch_min = res['ms_per_step'] * batches_per_epoch / 1000 / 60
            print(f'{dtype:>5} : {speedup:5.2f}x vs CPU   ~{epoch_min:5.1f} min/epoch (train split)')
    print()

    correctness_check()


if __name__ == '__main__':
    main()
