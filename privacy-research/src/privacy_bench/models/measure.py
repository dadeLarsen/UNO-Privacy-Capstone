"""Device setup and synchronized timing shared by every backend."""

import time

import torch

GIB = 1024**3


def require_cuda(*, bf16=False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run inference in the CUDA environment")
    if bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device 0 does not support BF16")
    torch.cuda.set_device(0)


def use_exact_fp32_matmul() -> dict:
    """Disable TF32 so FP32 matrix products use full precision; returns the settings for metadata."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def gpu_metadata() -> dict:
    return {"cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory}


def measured(run, *, cuda: bool):
    """Call run() in inference mode, timed between CUDA synchronizations.

    Returns (result, measurements). On CUDA the peaks cover only this call, including
    the resident weights.
    """
    if cuda:
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    with torch.inference_mode():
        result = run()
    if cuda:
        torch.cuda.synchronize(0)
    measurements = {"seconds": time.perf_counter() - started}
    if cuda:
        measurements.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
    return result, measurements


def timed_load(load, *, cuda: bool):
    """Load a model between CUDA synchronizations and return it with the seconds taken."""
    if cuda:
        torch.cuda.synchronize(0)
    started = time.perf_counter()
    model = load()
    if cuda:
        torch.cuda.synchronize(0)
    return model, time.perf_counter() - started
