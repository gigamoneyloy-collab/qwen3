# CustomVoice Prompt Matrix

This matrix validates the C++ CustomVoice frontend against the official PyTorch
prompt construction for representative multilingual/speaker cases.

Command shape:

```bash
python tools/qwen3_tts_customvoice_prompt_matrix.py \
  --model /path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --frontend-dir /path/to/ncnn/frontend \
  --frontend-check-exe /path/to/qwen3_tts_frontend_check.exe \
  --out-dir /tmp/qwen3_tts_prompt_matrix \
  --device cpu
```

Windows note: non-ASCII text is written to UTF-8 files and passed to C++ tools
as `@file` so the Windows process codepage cannot corrupt Chinese, Japanese, or
Korean prompts.

## Result

Run date: 2026-07-05

Summary:

```text
total: 12
passed: 12
failed: 0
threshold: 1e-4
max observed error: 2.38418579e-7
```

| Case | Language | Speaker | Prompt length | Max error |
| --- | --- | --- | ---: | ---: |
| `en_vivian` | English | Vivian | 19 | `2.38418579e-7` |
| `zh_vivian` | Chinese | Vivian | 19 | `2.38418579e-7` |
| `ja_ono_anna` | Japanese | Ono_Anna | 22 | `2.38418579e-7` |
| `ko_sohee` | Korean | Sohee | 28 | `2.38418579e-7` |
| `de_eric` | German | Eric | 25 | `2.38418579e-7` |
| `fr_serena` | French | Serena | 25 | `2.38418579e-7` |
| `ru_aiden` | Russian | Aiden | 33 | `2.38418579e-7` |
| `pt_ryan` | Portuguese | Ryan | 26 | `2.38418579e-7` |
| `es_dylan` | Spanish | Dylan | 25 | `2.38418579e-7` |
| `it_uncle_fu` | Italian | Uncle_Fu | 25 | `2.38418579e-7` |
| `zh_eric_dialect` | Chinese | Eric | 20 | `2.38418579e-7` |
| `zh_dylan_dialect` | Chinese | Dylan | 21 | `2.38418579e-7` |

This proves prompt-level parity for the 10 explicit CustomVoice languages and
the 9 built-in speakers. It does not prove generated-audio quality for every
case; audio regression coverage is tracked separately.
