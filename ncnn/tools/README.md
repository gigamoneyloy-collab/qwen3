# Tools

Python utilities used to export, convert, validate, and benchmark Qwen3-TTS for
the ncnn runtime.

Main conversion path:

```text
export_qwen3_tts_front_weights.py
export_qwen3_tts_frontend_nets.py
export_qwen3_tts_tokenizer_txt.py
export_qwen3_tts_decoder.py
export_qwen3_tts_talker_simple_rope.py
add_ncnn_sdpa_kvcache.py
make_qwen3_tts_talker_prefill_dynamic.py
export_qwen3_tts_code_predictor_body.py
split_qwen3_tts_codepred_kv_shared.py
```

Validation and benchmark helpers:

```text
export_qwen3_tts_long_ref.py
export_qwen3_tts_prompt_ref.py
qwen3_tts_customvoice_prompt_matrix.py
dump_qwen3_tts_tokenizer_ref.py
qwen3_tts_customvoice_audio_smoke.py
export_qwen3_tts_wave_prefix.py
benchmark_qwen3_tts_pytorch.py
time_command.py
```

Research/debug helpers are kept for reproducibility of the development trail.
They are not required for the minimal runtime package.

See `docs/CONVERSION.md` for the step-by-step workflow.
