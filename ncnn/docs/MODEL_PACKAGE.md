# Model Package Layout

This repository should only contain source code, build files, and conversion documentation.
Converted Qwen3-TTS model files are large and should be distributed as a separate model package.

The runtime expects a JSON file with absolute or relative paths to the following assets.

## Directory Layout

Recommended package layout:

```text
Qwen3-TTS-12Hz-0.6B-ncnn/
  model.json
  front_weights/
    first_embedding.f32.bin
    rest_embedding_0.f32.bin
    ...
    first_output_proj_weight.f32.bin
    rest_output_proj_weight.f32.bin
    pre_conv_weight.f32.bin
    pre_conv_bias.f32.bin
    manifest.json
  talker/
    tokenizer.txt
    talker_text_embed.ncnn.param
    talker_text_embed.ncnn.bin
    talker_codec_embed.ncnn.param
    talker_codec_embed.ncnn.bin
    talker_prefill_dynamic_kv.ncnn.param
    talker_prefill.ncnn.bin
    talker_decode_s1_kv.ncnn.param
    tts_pad_embed_f32.bin
    prefill_embed_f32.bin        # optional fallback / validation asset
    prefill_mask_f32.bin         # optional fallback / validation asset
    prefill_cos_f32.bin          # optional fallback / validation asset
    prefill_sin_f32.bin          # optional fallback / validation asset
  code_predictor/
    body/
      code_predictor_body_shared.ncnn.bin
      code_predictor_body_s02.ncnn.param
      ...
      code_predictor_body_s16.ncnn.param
    weights/
      talker_codec_embedding.f32
      codec_embedding_0.f32
      ...
      codec_embedding_14.f32
      lm_head_0.f32
      ...
      lm_head_14.f32
  decoder/
    speech_decoder_from_hidden_s325.ncnn.param
    speech_decoder_from_hidden_s325.ncnn.bin
```

The recommended prefill param is prompt-length dynamic. It is generated from a fixed-length pnnx export by rewriting only shape metadata:

- `Gemm 7=<seq_len>` to `7=0`
- `Reshape 0=128 1={8,16} 2=<seq_len>` to `2=-1`
- `Reshape 0=2048 1=<seq_len>` to `1=-1`

The talker `.ncnn.bin` weights are identical across prompt lengths and should be packaged once.

## Current Validated Long-Prompt Assets

The 20-second validation used these local files:

```text
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/long_prompt_20s/model_vulkan_all_raw_long_s97_250f.json
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/talker_simple_rope_dynamic_kv.ncnn.param
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/talker_simple_rope_s22.ncnn.bin
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/long_prompt_20s/talker_simple_input_f32.bin
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/long_prompt_20s/talker_simple_mask_f32.bin
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/long_prompt_20s/talker_simple_cos_f32.bin
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/long_prompt_20s/talker_simple_sin_f32.bin
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/long_prompt_20s/tts_pad_embed_f32.bin
```

Shared assets for that run:

```text
/workspace/Tencent/qwen3_tts_runs/baseline_001/front_weights
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/speech_decoder_from_hidden_s325.ncnn.param
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/speech_decoder_from_hidden_s325.ncnn.bin
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/talker_simple_rope_decode_s1_kv.ncnn.param
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/code_predictor_body_compact
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/code_predictor_weights
```

Approximate minimum package sizes observed locally:

```text
front_weights: 40M
code_predictor_body_compact: 301M
code_predictor_weights: 253M
speech_decoder_from_hidden_s325.ncnn.bin: 397M
talker prefill ncnn bin: 1.7G
```

The full experimental `nonstream` directory is much larger because it includes pnnx intermediate files, reference outputs, blob dumps, and failed/diagnostic variants. Do not publish that directory as-is.

## JSON Fields

```json
{
  "front_weights_dir": "front_weights",
  "decoder_param": "decoder/speech_decoder_from_hidden_s325.ncnn.param",
  "decoder_bin": "decoder/speech_decoder_from_hidden_s325.ncnn.bin",
  "talker_prefill_param": "talker/talker_prefill_dynamic_kv.ncnn.param",
  "talker_decode_param": "talker/talker_decode_s1_kv.ncnn.param",
  "talker_bin": "talker/talker_prefill.ncnn.bin",
  "codepred_body_dir": "code_predictor/body",
  "codepred_weights_dir": "code_predictor/weights",
  "tts_pad_embed": "talker/tts_pad_embed_f32.bin",
  "prefill_embed": "talker/prefill_embed_f32.bin",
  "prefill_mask": "talker/prefill_mask_f32.bin",
  "prefill_cos": "talker/prefill_cos_f32.bin",
  "prefill_sin": "talker/prefill_sin_f32.bin",
  "tokenizer": "talker/tokenizer.txt",
  "text_embed_param": "talker/talker_text_embed.ncnn.param",
  "text_embed_bin": "talker/talker_text_embed.ncnn.bin",
  "codec_embed_param": "talker/talker_codec_embed.ncnn.param",
  "codec_embed_bin": "talker/talker_codec_embed.ncnn.bin",
  "decoder_chunk_frames": 325,
  "decoder_context_frames": 25,
  "sample_rate": 24000,
  "samples_per_frame": 1920,
  "fp16": false
}
```

## Packaging Rules

- Use `fp16=0` in pnnx for parity-oriented conversion.
- Keep reference `.npy`, reference wav/code bins, pnnx generated Python, ONNX, and `.pt` export files out of the runtime package.
- Use `make_qwen3_tts_talker_prefill_dynamic.py` to convert the pnnx-exported prefill param to the dynamic form before packaging.
- The decode graph is shared because its sequence length is one token and it consumes KV cache from prefill/decode steps.
- Prefer relative paths in `model.json` for redistributable packages.
