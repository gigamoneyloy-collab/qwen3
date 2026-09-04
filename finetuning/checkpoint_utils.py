# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""Transactional checkpoint utilities for the 12 Hz fine-tuning script."""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
import uuid
from typing import Callable, Optional

_CODEC_EMBEDDING_KEY = 'talker.model.codec_embedding.weight'
_BASE_WEIGHT_NAMES = frozenset({
    'model.safetensors',
    'model.safetensors.index.json',
    'pytorch_model.bin',
    'pytorch_model.bin.index.json',
})


def build_checkpoint_state_dict(model,
                                target_speaker_embedding,
                                *,
                                speaker_id: int = 3000,
                                drop_prefix: str = 'speaker_encoder') -> dict[str, object]:
    """Materialize CPU weights without retaining the previous epoch's copy.

    The model's state dictionary is traversed once, dropping the speaker
    encoder and updating the reserved codec row in-place.  Keeping this work
    in a short-lived helper prevents a completed checkpoint from remaining
    live while the next epoch's CPU snapshot is built.
    """
    checkpoint_state = {}
    model_state = model.state_dict()
    try:
        for name, value in model_state.items():
            if name.startswith(drop_prefix):
                continue
            # ``Tensor.to('cpu')`` may return the original storage when the
            # model is already on CPU.  ``copy=True`` keeps the checkpoint
            # snapshot independent, so patching the reserved codec row never
            # mutates the live model.
            checkpoint_state[name] = value.detach().to(device='cpu', copy=True)
    finally:
        del model_state

    if target_speaker_embedding is not None:
        if _CODEC_EMBEDDING_KEY not in checkpoint_state:
            raise KeyError(f'Missing {_CODEC_EMBEDDING_KEY!r} in model state dict')
        codec_weight = checkpoint_state[_CODEC_EMBEDDING_KEY]
        speaker_vector = target_speaker_embedding[0].detach().to(
            device=codec_weight.device, dtype=codec_weight.dtype)
        codec_weight[speaker_id].copy_(speaker_vector)
        del speaker_vector

    return checkpoint_state


def _ignore_base_weights(source_root: str, directory: str, names):
    if os.path.realpath(directory) != os.path.realpath(source_root):
        return []
    return [name for name in names if name.lower() in _BASE_WEIGHT_NAMES]


def _sync_file(path: str) -> None:
    """Flush a staged file when the platform exposes fsync."""
    try:
        with open(path, 'rb') as file_obj:
            os.fsync(file_obj.fileno())
    except OSError:
        # Directory fsync is unavailable on some Windows filesystems.  The
        # rename below still provides atomic visibility of the checkpoint.
        pass


def _sync_directory(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _publish_directory(staged_dir: str, final_dir: str) -> None:
    """Publish a complete staged directory while preserving an old one."""
    backup_dir = None
    if os.path.lexists(final_dir):
        backup_dir = f'{final_dir}.previous-{uuid.uuid4().hex}'
        os.replace(final_dir, backup_dir)

    try:
        # The destination is absent here, so this is a single directory-name
        # transition.  Readers never observe the staged/base-weight mixture.
        os.replace(staged_dir, final_dir)
    except BaseException:
        if backup_dir is not None and os.path.lexists(backup_dir) and not os.path.lexists(final_dir):
            os.replace(backup_dir, final_dir)
        raise

    if backup_dir is not None:
        if os.path.isdir(backup_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)
        else:
            try:
                os.unlink(backup_dir)
            except FileNotFoundError:
                pass


def _update_config(config_path: str, speaker_name: str, speaker_id: int) -> None:
    with open(config_path, 'r', encoding='utf-8') as file_obj:
        config = json.load(file_obj)
    config['tts_model_type'] = 'custom_voice'
    talker_config = config.setdefault('talker_config', {})
    talker_config['spk_id'] = {speaker_name: speaker_id}
    talker_config['spk_is_dialect'] = {speaker_name: False}
    with open(config_path, 'w', encoding='utf-8') as file_obj:
        json.dump(config, file_obj, indent=2, ensure_ascii=False)
        file_obj.flush()
        try:
            os.fsync(file_obj.fileno())
        except OSError:
            pass


def save_finetuned_checkpoint(model_path: str,
                              output_model_path: str,
                              checkpoint_name: str,
                              speaker_name: str,
                              model,
                              target_speaker_embedding,
                              *,
                              speaker_id: int = 3000,
                              save_file_fn: Optional[Callable[[dict[str, object], str], None]] = None) -> str:
    """Write a complete fine-tuned checkpoint and publish it atomically.

    Metadata is copied into a temporary sibling directory, base weight files
    are omitted, and the trained weights are written there before the final
    directory name is installed.  The old checkpoint (if any) is retained
    until the new one is ready.  CPU snapshots are explicitly cleared in the
    ``finally`` block so repeated epochs do not accumulate host memory.

    Returns:
        The final checkpoint directory path.
    """
    os.makedirs(output_model_path, exist_ok=True)
    final_dir = os.path.join(output_model_path, checkpoint_name)
    staged_dir = tempfile.mkdtemp(prefix=f'.{checkpoint_name}.', dir=output_model_path)
    checkpoint_state = None
    try:
        shutil.copytree(
            model_path,
            staged_dir,
            dirs_exist_ok=True,
            ignore=lambda directory, names: _ignore_base_weights(model_path, directory, names),
        )
        _update_config(os.path.join(staged_dir, 'config.json'), speaker_name, speaker_id)

        checkpoint_state = build_checkpoint_state_dict(
            model,
            target_speaker_embedding,
            speaker_id=speaker_id,
        )
        if save_file_fn is None:
            from safetensors.torch import save_file
            save_file_fn = save_file
        weight_path = os.path.join(staged_dir, 'model.safetensors')
        save_file_fn(checkpoint_state, weight_path)
        _sync_file(weight_path)
        _sync_directory(staged_dir)
        _publish_directory(staged_dir, final_dir)
        staged_dir = None
        return final_dir
    finally:
        if checkpoint_state is not None:
            checkpoint_state.clear()
            del checkpoint_state
        if staged_dir is not None:
            shutil.rmtree(staged_dir, ignore_errors=True)
        gc.collect()
