"""Device selection and GPU introspection utilities (Phase 3).

Pure queries — no side effects (no ``empty_cache()``, no allocator warmup).
``gpu_summary()`` output is written into every checkpoint's metadata so a
result can always be traced back to the hardware that produced it, which
matters on RunPod where the GPU type varies between sessions.

XPU (Intel Arc) is supported via ``hasattr(torch, 'xpu')`` guards only; IPEX
is never imported here.
"""
import platform

import torch


def _xpu_available():
    return hasattr(torch, 'xpu') and torch.xpu.is_available()


def get_device(prefer='cuda'):
    """Return the best available torch device.

    Prefer order: 'cuda' -> 'xpu' -> 'cpu'. ``prefer='cpu'`` forces cpu.
    Logs a one-line summary of the selected device.
    """
    if prefer == 'cpu':
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif _xpu_available():
        device = torch.device('xpu')
    else:
        device = torch.device('cpu')

    s = gpu_summary(device)
    print(f"[device] {s['backend']}: {s['name']} "
          f"mem={s['total_memory_gb']}GB "
          f"capability={s['capability_major']}.{s['capability_minor']} "
          f"driver={s['driver_version']}")
    return device


def get_amp_dtype(device):
    """Preferred autocast dtype for ``device``.

    CUDA Ampere+ (compute capability >= 8.0) and XPU get bfloat16; older
    CUDA gets float16; CPU gets None (autocast disabled).
    """
    if device.type == 'cuda':
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    if device.type == 'xpu':
        return torch.bfloat16
    return None


def _nvidia_driver_version():
    try:
        import pynvml
        pynvml.nvmlInit()
        v = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()
        return v.decode() if isinstance(v, bytes) else str(v)
    except Exception:
        return None


def gpu_summary(device):
    """Dict describing ``device`` for checkpoint reproducibility metadata.

    Keys: name, total_memory_gb, capability_major, capability_minor,
    driver_version (None if unavailable), backend ('cuda' | 'xpu' | 'cpu').
    """
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        return {
            'name': props.name,
            'total_memory_gb': round(props.total_memory / 2 ** 30, 2),
            'capability_major': props.major,
            'capability_minor': props.minor,
            'driver_version': _nvidia_driver_version(),
            'backend': 'cuda',
        }
    if device.type == 'xpu':
        props = torch.xpu.get_device_properties(device)
        return {
            'name': props.name,
            'total_memory_gb': round(props.total_memory / 2 ** 30, 2),
            'capability_major': None,
            'capability_minor': None,
            'driver_version': str(getattr(props, 'driver_version', None)),
            'backend': 'xpu',
        }
    return {
        'name': platform.processor() or 'cpu',
        'total_memory_gb': None,
        'capability_major': None,
        'capability_minor': None,
        'driver_version': None,
        'backend': 'cpu',
    }
