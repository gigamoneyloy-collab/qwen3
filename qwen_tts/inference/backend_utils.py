# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


_MLX_ALIASES = {"mlx", "mlx:0", "apple-mlx"}
_MPS_ALIASES = {"mps", "mps:0"}
_CUDA_PREFIXES = ("cuda",)


def _has_mps() -> bool:
    return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if _has_mps():
        return "mps"
    return "cpu"


def device_supports_bfloat16(device: str) -> bool:
    device = normalize_device_name(device)
    if device.startswith("cuda"):
        return True
    if device == "cpu":
        return True
    return False


def normalize_device_name(device: Optional[str]) -> str:
    raw = (device or "").strip().lower()
    if not raw or raw == "auto":
        return default_device()
    if raw in _MLX_ALIASES:
        return "mps" if _has_mps() else "cpu"
    if raw in _MPS_ALIASES:
        return "mps" if _has_mps() else "cpu"
    if raw.startswith(_CUDA_PREFIXES):
        return raw if torch.cuda.is_available() else ("mps" if _has_mps() else "cpu")
    if raw == "cpu":
        return "cpu"
    return raw


def normalize_dtype_for_device(dtype: Optional[torch.dtype], device: str) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if dtype == torch.bfloat16 and not device_supports_bfloat16(device):
        return torch.float16 if device != "cpu" else torch.float32
    return dtype


def normalize_attn_implementation(attn_implementation: Optional[str], device: str) -> Optional[str]:
    device = normalize_device_name(device)
    if attn_implementation == "flash_attention_2" and not device.startswith("cuda"):
        return None
    return attn_implementation


def resolve_model_load_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(kwargs)
    device = normalize_device_name(resolved.get("device_map"))
    resolved["device_map"] = device

    dtype_key = "dtype" if "dtype" in resolved else "torch_dtype" if "torch_dtype" in resolved else None
    if dtype_key is not None:
        resolved[dtype_key] = normalize_dtype_for_device(resolved.get(dtype_key), device)

    if "attn_implementation" in resolved:
        resolved["attn_implementation"] = normalize_attn_implementation(resolved.get("attn_implementation"), device)

    return resolved


def synchronize_device(device: Optional[str]) -> None:
    name = normalize_device_name(device)
    if name.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        return
    if name == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
