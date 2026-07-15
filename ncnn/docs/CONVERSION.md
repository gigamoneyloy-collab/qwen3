# Conversion Workflow

This document records the current reproducible conversion path from the PyTorch Qwen3-TTS checkpoint to ncnn runtime assets.

Local paths used during development:

```text
PyTorch model: /workspace/Tencent/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
Python venv: /workspace/Tencent/Qwen3-TTS/.venv
Export scripts: /workspace/Tencent/tools
Output root: /workspace/Tencent/qwen3_tts_runs/baseline_001
ncnn build with pnnx: /workspace/Tencent/ncnn/build-vulkan
```

Use the Qwen3-TTS Python environment:

```bash
cd /workspace/Tencent/Qwen3-TTS
source .venv/bin/activate
```

## 1. Export Codec Front Weights

```bash
python /workspace/Tencent/tools/export_qwen3_tts_front_weights.py \
  --safetensors /workspace/Tencent/models/Qwen3-TTS-12Hz-0.6B-CustomVoice/speech_tokenizer_v2_25hz/model.safetensors \
  --out-dir /workspace/Tencent/qwen3_tts_runs/baseline_001/front_weights
```

This exports FP32 codebook, projection, and pre-conv weights used by the C++ codec front end.

## 2. Export Speech Decoder

```bash
python /workspace/Tencent/tools/export_qwen3_tts_decoder.py \
  --model /workspace/Tencent/models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --codes /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/talker_loop_25f_codes_ref.npy \
  --out /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/speech_decoder_from_hidden_s325.pt \
  --mode from-hidden \
  --trace-seq-len 325 \
  --device cpu \
  --dtype float32
```

Convert with pnnx using FP32 weights:

```bash
cd /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream
/workspace/Tencent/ncnn/build-vulkan/tools/pnnx speech_decoder_from_hidden_s325.pt \
  inputshape=[1,325,1024]f32 \
  fp16=0
```

Keep:

```text
speech_decoder_from_hidden_s325.ncnn.param
speech_decoder_from_hidden_s325.ncnn.bin
```

## 3. Export Talker Prefill

For the default short prompt:

```bash
python /workspace/Tencent/tools/export_qwen3_tts_talker_simple_rope.py \
  --model /workspace/Tencent/models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --out-dir /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/pnnx_talker_simple_rope \
  --text "你好，欢迎使用通义千问语音合成。" \
  --language Chinese \
  --speaker Vivian \
  --device cpu
```

For the validated long prompt:

```bash
python /workspace/Tencent/tools/export_qwen3_tts_talker_simple_rope.py \
  --model /workspace/Tencent/models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --out-dir /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/long_prompt_20s/pnnx_talker_simple_rope \
  --text "本次测试用于验证长音频场景下的语音合成一致性。我们将使用一段较长的中文内容，覆盖多个句子、停顿和连续表达，以便生成接近二十秒的音频。模型需要稳定地产生语音编码，随后通过声码器还原成最终波形。这个测试可以帮助确认转换后的 ncnn Vulkan 推理结果，在较长时间范围内仍然和 PyTorch 参考结果保持一致。" \
  --language Chinese \
  --speaker Vivian \
  --device cpu
```

The exporter writes:

```text
talker_simple_rope_s22.pt
talker_simple_input_f32.bin
talker_simple_mask_f32.bin
talker_simple_cos_f32.bin
talker_simple_sin_f32.bin
tts_pad_embed_f32.bin
```

The `.pt` filename is historical; check `talker_simple_rope_meta.json` for the actual `input_shape`. For the long prompt the prefill length is 97.

Convert the long-prompt prefill graph:

```bash
cd /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/long_prompt_20s/pnnx_talker_simple_rope
/workspace/Tencent/ncnn/build-vulkan/tools/pnnx talker_simple_rope_s22.pt \
  inputshape=[1,97,1024]f32,[1,1,97,97]f32,[1,97,128]f32,[1,97,128]f32 \
  fp16=0
python /workspace/Tencent/tools/add_ncnn_sdpa_kvcache.py \
  --in-param talker_simple_rope_s22.ncnn.param \
  --out-param talker_simple_rope_s97_kv.ncnn.param
```

Make the prefill graph prompt-length dynamic:

```bash
python /workspace/Tencent/tools/make_qwen3_tts_talker_prefill_dynamic.py \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/long_prompt_20s/talker_simple_rope_s97_kv.ncnn.param \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/talker_simple_rope_dynamic_kv.ncnn.param \
  --seq-len 97
```

The same rewrite from the short `s22` param and the long `s97` param produces identical dynamic params. The `.ncnn.bin` files also have identical sha256, so package only one talker bin.

Keep:

```text
talker_simple_rope_dynamic_kv.ncnn.param
talker_simple_rope_s22.ncnn.bin
talker_simple_input_f32.bin
talker_simple_mask_f32.bin
talker_simple_cos_f32.bin
talker_simple_sin_f32.bin
tts_pad_embed_f32.bin
```

Rename the bin in the package if desired:

```text
talker_prefill.ncnn.bin
```

## 4. Export Talker Decode Graph

Generate decode assets from the first-frame reference:

```bash
python /workspace/Tencent/tools/export_qwen3_tts_talker_simple_rope.py \
  --model /workspace/Tencent/models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --out-dir /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/pnnx_talker_decode \
  --text "你好，欢迎使用通义千问语音合成。" \
  --language Chinese \
  --speaker Vivian \
  --first-frame-codes /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/talker_loop_25f_codes_ref_i32.bin \
  --trace-decode \
  --device cpu
```

Convert with pnnx:

```bash
cd /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/pnnx_talker_decode
/workspace/Tencent/ncnn/build-vulkan/tools/pnnx talker_simple_rope_decode_s1.pt \
  inputshape=[1,1,1024]f32,[1,1,1,1]f32,[1,1,128]f32,[1,1,128]f32 \
  fp16=0
python /workspace/Tencent/tools/add_ncnn_sdpa_kvcache.py \
  --in-param talker_simple_rope_decode_s1.ncnn.param \
  --out-param talker_simple_rope_decode_s1_kv.ncnn.param
```

The decode graph shares the talker `.ncnn.bin` weights with the prefill graph when both are exported from the same model.

## 5. Export Code Predictor

```bash
python /workspace/Tencent/tools/export_qwen3_tts_code_predictor_body.py \
  --model /workspace/Tencent/models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --talker-hidden /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/talker_simple_hidden_ref_f32.bin \
  --first-code 0 \
  --out-dir /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/code_predictor_export \
  --device cpu \
  --all-lengths
```

Convert each fixed sequence length with pnnx and then compact the shared weights into one `.ncnn.bin`.
The current runtime package uses:

```text
code_predictor_body_compact/code_predictor_body_shared.ncnn.bin
code_predictor_body_compact/code_predictor_body_s02.ncnn.param
...
code_predictor_body_compact/code_predictor_body_s16.ncnn.param
code_predictor_weights/talker_codec_embedding.f32
code_predictor_weights/codec_embedding_0.f32
...
code_predictor_weights/lm_head_14.f32
```

## 6. Build Runtime

```bash
cmake -S /workspace/Tencent/Qwen3-TTS-ncnn \
  -B /workspace/Tencent/Qwen3-TTS-ncnn/build-vulkan \
  -DNCNN_ROOT=/workspace/Tencent/ncnn \
  -DNCNN_BUILD_DIR=/workspace/Tencent/ncnn/build-vulkan
cmake --build /workspace/Tencent/Qwen3-TTS-ncnn/build-vulkan -j$(nproc)
```

## 7. Validate

```bash
/workspace/Tencent/Qwen3-TTS-ncnn/build-vulkan/qwen3_tts_model_json_check \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/long_prompt_20s/model_vulkan_all_raw_long_s97_250f.json \
  250 \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/long_prompt_20s/talker_loop_250f_long_codes_ref_i32.bin \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/long_prompt_20s/talker_loop_250f_long_wav_ref_f32.bin \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/long_prompt_20s/runtime_full_250f_vulkan_all_raw_codes_i32.bin \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/long_prompt_20s/runtime_full_250f_vulkan_all_raw_wav_f32.bin \
  4 1
```

Expected long-prompt result:

```text
code mismatches=0/4000 generated=4000 ref=4000
wav mae=4.05714474e-06
wav rmse=6.609014e-06
wav maxe=0.0001267888
```
