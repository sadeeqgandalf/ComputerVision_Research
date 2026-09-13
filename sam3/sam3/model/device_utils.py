# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Device helpers for CUDA / Apple Silicon (MPS) / CPU."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

import torch


def get_device() -> str:
    """Prefer CUDA, then MPS (Apple Silicon), then CPU.

    Override with SAM3_DEVICE=cpu|mps|cuda.
    """
    override = os.environ.get("SAM3_DEVICE", "").strip().lower()
    if override in {"cpu", "mps", "cuda"}:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_torch_device() -> torch.device:
    return torch.device(get_device())


def autocast_device_type(device: torch.device | str | None = None) -> str:
    """Map a torch device to an amp autocast device_type string."""
    if device is None:
        return get_device()
    if isinstance(device, torch.device):
        dtype = device.type
    else:
        dtype = str(device).split(":")[0]
    if dtype in ("cuda", "mps", "cpu"):
        return dtype
    return "cpu"


def amp_context(dtype: torch.dtype = torch.bfloat16, enabled: bool | None = None):
    """Autocast only when CUDA is available; otherwise no-op (CPU/MPS float32)."""
    if not torch.cuda.is_available():
        return nullcontext()
    if enabled is False:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def feature_storage_dtype() -> torch.dtype:
    """CUDA video path stores features in bf16; CPU/MPS keep float32."""
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def to_compute_device(tensor: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
    """Move a tensor to the active compute device (replaces hard-coded .cuda())."""
    return tensor.to(get_torch_device(), non_blocking=non_blocking)


_SHIM_INSTALLED = False


def install_cuda_compat_shim() -> str:
    """Redirect Tensor/Module .cuda() to the active Mac-safe device.

    Call once at process start before building video models. Idempotent.
    """
    global _SHIM_INSTALLED
    device = get_device()
    # Video path still hits Metal dtype issues on some Macs; prefer CPU unless forced.
    if device == "mps" and os.environ.get("SAM3_ALLOW_MPS_VIDEO", "").strip() != "1":
        os.environ["SAM3_DEVICE"] = "cpu"
        device = "cpu"

    if _SHIM_INSTALLED:
        return device

    torch_device = torch.device(device)

    def _tensor_cuda(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.to(torch_device)

    def _module_cuda(self: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.nn.Module:
        return self.to(torch_device)

    torch.Tensor.cuda = _tensor_cuda  # type: ignore[method-assign]
    torch.nn.Module.cuda = _module_cuda  # type: ignore[method-assign]
    _SHIM_INSTALLED = True
    return device
