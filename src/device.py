"""Dynamic device detection with clean CPU fallback.

Order: CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU.
`FORCE_CPU=1` overrides everything (useful for reproducible CI / air-gapped tests).
ONNX providers mirror the same ordering.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def get_torch_device() -> str:
    """Return 'cuda' | 'mps' | 'cpu'. Never raises — always returns a usable device."""
    if os.getenv("FORCE_CPU", "0") == "1":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # MPS exists only on Apple builds; guard with hasattr for Linux/Windows.
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        log.info("torch not installed; using cpu-numpy fallback")
        return "cpu"
    except Exception as exc:  # defensive: device probing must never crash the pipeline
        log.warning("device probe failed (%s); falling back to cpu", exc)
        return "cpu"


def get_onnx_providers() -> list[str]:
    """ONNX Runtime execution providers ordered by preference, filtered by availability.

    Order reflects the 2026 accelerator landscape: TensorRT (NVIDIA dGPU) >
    OpenVINO (Intel iGPU/NPU) > CUDA (NVIDIA generic) > CoreML (Apple) > CPU.
    `FORCE_CPU=1` collapses to CPU only (reproducible air-gapped runs).
    """
    cpu_only = ["CPUExecutionProvider"]
    if os.getenv("FORCE_CPU", "0") == "1":
        return cpu_only
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        ordered = [
            "TensorRTExecutionProvider",
            "OpenVINOExecutionProvider",
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers = [p for p in ordered if p in available]
        return providers or cpu_only
    except ImportError:
        return cpu_only
    except Exception as exc:
        log.warning("onnx provider probe failed (%s)", exc)
        return cpu_only


def describe_compute() -> dict[str, str | list[str]]:
    """Small diagnostic dict for reports / logs."""
    device = get_torch_device()
    return {"torch_device": device, "onnx_providers": get_onnx_providers()}
