import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from finetuning.checkpoint_utils import save_finetuned_checkpoint


class _Model:

    def __init__(self):
        self._state = {
            'speaker_encoder.weight': torch.full((2, 2), 9.0),
            'talker.model.codec_embedding.weight': torch.zeros((3001, 4)),
            'talker.model.projection.weight': torch.arange(8, dtype=torch.float32).reshape(2, 4),
        }

    def state_dict(self):
        return self._state


def _make_source(tmp_path: Path) -> Path:
    source = tmp_path / 'base'
    source.mkdir()
    (source / 'config.json').write_text(json.dumps({'model_type': 'qwen3_tts'}), encoding='utf-8')
    (source / 'tokenizer.json').write_text('metadata', encoding='utf-8')
    (source / 'model.safetensors').write_bytes(b'base weights')
    (source / 'model.safetensors.index.json').write_text('base index', encoding='utf-8')
    nested = source / 'speech_tokenizer'
    nested.mkdir()
    (nested / 'model.safetensors').write_bytes(b'nested tokenizer weights')
    return source


def _target_embedding():
    return torch.tensor([[1.0, 2.0, 3.0, 4.0]])


def test_checkpoint_publishes_complete_weights_and_metadata(tmp_path):
    source = _make_source(tmp_path)
    output = tmp_path / 'output'
    model = _Model()

    checkpoint = save_finetuned_checkpoint(
        str(source),
        str(output),
        'checkpoint-epoch-0',
        'speaker_test',
        model,
        _target_embedding(),
    )

    checkpoint_path = Path(checkpoint)
    assert checkpoint_path.is_dir()
    assert (checkpoint_path / 'tokenizer.json').read_text(encoding='utf-8') == 'metadata'
    assert (checkpoint_path / 'speech_tokenizer' / 'model.safetensors').read_bytes() == b'nested tokenizer weights'
    assert not (checkpoint_path / 'model.safetensors.index.json').exists()
    weights = load_file(str(checkpoint_path / 'model.safetensors'))
    assert 'speaker_encoder.weight' not in weights
    torch.testing.assert_close(weights['talker.model.codec_embedding.weight'][3000], _target_embedding()[0])
    config = json.loads((checkpoint_path / 'config.json').read_text(encoding='utf-8'))
    assert config['tts_model_type'] == 'custom_voice'
    assert config['talker_config']['spk_id'] == {'speaker_test': 3000}
    assert config['talker_config']['spk_is_dialect'] == {'speaker_test': False}
    torch.testing.assert_close(
        model._state['talker.model.codec_embedding.weight'][3000],
        torch.zeros(4),
    )
    assert not list(output.glob('.*'))


def test_failed_save_keeps_previous_checkpoint_and_cleans_stage(tmp_path):
    source = _make_source(tmp_path)
    output = tmp_path / 'output'
    old_checkpoint = output / 'checkpoint-epoch-0'
    old_checkpoint.mkdir(parents=True)
    (old_checkpoint / 'config.json').write_text('{"old": true}', encoding='utf-8')
    (old_checkpoint / 'model.safetensors').write_bytes(b'previous weights')

    def fail_after_partial_write(state_dict, path):
        assert state_dict
        Path(path).write_bytes(b'partial weights')
        raise RuntimeError('simulated interruption')

    with pytest.raises(RuntimeError, match='simulated interruption'):
        save_finetuned_checkpoint(
            str(source),
            str(output),
            'checkpoint-epoch-0',
            'speaker_test',
            _Model(),
            _target_embedding(),
            save_file_fn=fail_after_partial_write,
        )

    assert (old_checkpoint / 'config.json').read_text(encoding='utf-8') == '{"old": true}'
    assert (old_checkpoint / 'model.safetensors').read_bytes() == b'previous weights'
    assert not list(output.glob('.*'))
    assert not list(output.glob('checkpoint-epoch-0.previous-*'))


def test_replacing_checkpoint_never_copies_base_weights(tmp_path):
    source = _make_source(tmp_path)
    output = tmp_path / 'output'
    old_checkpoint = output / 'checkpoint-epoch-0'
    old_checkpoint.mkdir(parents=True)
    (old_checkpoint / 'config.json').write_text('{}', encoding='utf-8')
    (old_checkpoint / 'model.safetensors').write_bytes(b'old')

    save_finetuned_checkpoint(
        str(source),
        str(output),
        'checkpoint-epoch-0',
        'speaker_test',
        _Model(),
        _target_embedding(),
        save_file_fn=save_file,
    )

    weights = load_file(str(old_checkpoint / 'model.safetensors'))
    assert weights['talker.model.projection.weight'].shape == (2, 4)
    assert (old_checkpoint / 'model.safetensors').read_bytes() != b'old'
    assert not list(output.glob('.*'))
    assert not list(output.glob('checkpoint-epoch-0.previous-*'))


def test_cpu_snapshot_is_released_after_save(tmp_path):
    source = _make_source(tmp_path)
    output = tmp_path / 'output'
    captured = {}

    def capture_state(state_dict, path):
        captured['state_dict'] = state_dict
        Path(path).write_bytes(b'complete weights')

    save_finetuned_checkpoint(
        str(source),
        str(output),
        'checkpoint-epoch-0',
        'speaker_test',
        _Model(),
        _target_embedding(),
        save_file_fn=capture_state,
    )

    assert captured['state_dict'] == {}
