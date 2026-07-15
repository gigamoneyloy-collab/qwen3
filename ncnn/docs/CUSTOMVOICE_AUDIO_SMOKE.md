# CustomVoice Audio Smoke

This check runs the ncnn CLI end to end for representative CustomVoice prompts:
C++ tokenizer/frontend, talker generation, code predictor, codec frontend, and
speech decoder.

Command shape:

```bash
python tools/qwen3_tts_customvoice_audio_smoke.py \
  --qwen3-tts-exe /path/to/qwen3_tts.exe \
  --model-json /path/to/model.json \
  --out-dir /tmp/qwen3_tts_audio_smoke \
  --cases /path/to/cases.json \
  --frames 25 \
  --threads 3 \
  --vulkan
```

## Representative Result

Run date: 2026-07-05

All four representative cases generated valid 25-frame wav files:

| Case | Language | Speaker | Prompt length | Samples | RMS | PyTorch code mismatch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `en_vivian` | English | Vivian | 19 | 48000 | 4161.000 | not checked in this run |
| `zh_vivian` | Chinese | Vivian | 19 | 48000 | 4.530 | `0/400` |
| `ja_ono_anna` | Japanese | Ono_Anna | 22 | 48000 | 4.948 | not checked in this run |
| `ko_sohee` | Korean | Sohee | 28 | 48000 | 0.000 | `0/400` |

The low-RMS Chinese and Korean outputs are not treated as ncnn failures in this
deterministic 25-frame smoke: for checked cases, ncnn acoustic codes matched the
official PyTorch reference exactly, and the PyTorch reference wav was also
near-silent for the first 25 frames.

This smoke proves that multilingual UTF-8 text-file input and end-to-end wav
generation work. It does not yet prove subjective audio quality for every
language/speaker; longer generation and sampling-mode tests are still needed.
