# ncnn Vulkan operator fixes: PR split and repro notes

This directory preserves the current ncnn operator fixes as standalone patches and records how to reproduce the Qwen3-TTS failures that exposed them.

Do not reset the current dirty ncnn worktree unless the patches below have been copied or committed. The current fixes are also saved under:

- `patches/0001-rotaryembed-vulkan-cache-stride.patch`
- `patches/0002-sdpa-vulkan-broadcast-mask.patch`
- `patches/0003-gelu-vulkan-fast-gelu-param.patch`
- `patches/all-three-vulkan-operator-fixes.patch`

## Safe branch workflow

Use `git worktree` so the current dirty `/workspace/Tencent/ncnn` tree is not touched.

```bash
cd /workspace/Tencent/ncnn

git worktree add /workspace/Tencent/ncnn-pr-rotaryembed -b fix-vulkan-rotaryembed HEAD
git -C /workspace/Tencent/ncnn-pr-rotaryembed apply /workspace/Tencent/operator_pr_repro/patches/0001-rotaryembed-vulkan-cache-stride.patch

git worktree add /workspace/Tencent/ncnn-pr-sdpa -b fix-vulkan-sdpa-mask-broadcast HEAD
git -C /workspace/Tencent/ncnn-pr-sdpa apply /workspace/Tencent/operator_pr_repro/patches/0002-sdpa-vulkan-broadcast-mask.patch

git worktree add /workspace/Tencent/ncnn-pr-gelu -b fix-vulkan-gelu-fast-gelu HEAD
git -C /workspace/Tencent/ncnn-pr-gelu apply /workspace/Tencent/operator_pr_repro/patches/0003-gelu-vulkan-fast-gelu-param.patch
```

Rebuild each worktree with a separate build dir, for example:

```bash
cmake -S /workspace/Tencent/ncnn-pr-rotaryembed -B /workspace/Tencent/ncnn-pr-rotaryembed/build-vulkan \
  -DNCNN_VULKAN=ON -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=OFF
cmake --build /workspace/Tencent/ncnn-pr-rotaryembed/build-vulkan --target ncnn -j$(nproc)
```

## Shared repro environment

Existing paths:

- Qwen runtime: `/workspace/Tencent/Qwen3-TTS-ncnn`
- ncnn source with current fixes: `/workspace/Tencent/ncnn`
- Vulkan ncnn build: `/workspace/Tencent/ncnn/build-vulkan`
- Qwen Vulkan build: `/workspace/Tencent/Qwen3-TTS-ncnn/build-vulkan`
- pnnx executable: `/workspace/Tencent/Qwen3-TTS/.venv/bin/pnnx`
- Logs: `/workspace/Tencent/qwen3_tts_runs/logs`
- Converted graphs: `/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream`

The final parity conversions must use `fp16=0`. pnnx default fp16 conversion caused measurable waveform drift in this project.

The Qwen runtime currently disables fp16 compute/storage in `configure_net_options()` to keep FP32 parity:

- `use_packing_layout=false`
- `use_fp16_packed=false`
- `use_fp16_storage=false`
- `use_fp16_arithmetic=false`

## ONNX / pnnx conversion commands

These commands regenerate the ONNX and ncnn artifacts that contain the relevant operators. They use existing traced TorchScript `.pt` files.

### Decoder graph, covers all three fixes

This is the most useful single repro graph. It contains:

- `RotaryEmbed`
- `SDPA`
- `GELU`

```bash
cd /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream

/workspace/Tencent/Qwen3-TTS/.venv/bin/pnnx \
  speech_decoder_from_hidden_s325.pt \
  inputshape=[1,325,1024]f32 \
  fp16=0 \
  optlevel=2 \
  device=cpu \
  > /workspace/Tencent/qwen3_tts_runs/logs/pnnx_decoder_from_hidden_s325_fp32_repro.log 2>&1
```

Expected outputs:

- `speech_decoder_from_hidden_s325.pnnx.onnx`
- `speech_decoder_from_hidden_s325.pnnx.param`
- `speech_decoder_from_hidden_s325.pnnx.bin`
- `speech_decoder_from_hidden_s325.ncnn.param`
- `speech_decoder_from_hidden_s325.ncnn.bin`

Existing conversion log:

- `/workspace/Tencent/qwen3_tts_runs/logs/pnnx_decoder_from_hidden_s325_fp32.log`

### Talker graph, covers RotaryEmbed and SDPA

```bash
cd /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/pnnx_talker_simple_rope

/workspace/Tencent/Qwen3-TTS/.venv/bin/pnnx \
  ../talker_simple_rope_s22.pt \
  inputshape=[1,22,1024]f32,[1,1,22,22]f32,[1,22,128]f32,[1,22,128]f32 \
  fp16=0 \
  optlevel=2 \
  device=cpu \
  > /workspace/Tencent/qwen3_tts_runs/logs/talker_simple_rope_pnnx_repro.log 2>&1
```

Expected outputs are written one level up, matching the existing log:

- `../talker_simple_rope_s22.pnnx.onnx`
- `../talker_simple_rope_s22.ncnn.param`
- `../talker_simple_rope_s22.ncnn.bin`

Existing conversion log:

- `/workspace/Tencent/qwen3_tts_runs/logs/talker_simple_rope_pnnx.log`

### Code predictor body, covers RotaryEmbed and SDPA

Use one fixed-length graph for an operator repro. The full runtime uses `s02..s16`.

```bash
cd /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/code_predictor_body_by_len/pnnx_s02

/workspace/Tencent/Qwen3-TTS/.venv/bin/pnnx \
  ../code_predictor_body_s02.pt \
  inputshape=[1,2,1024]f32,[1,1,2,2]f32,[1,2]i64,[2]i64 \
  fp16=0 \
  optlevel=2 \
  device=cpu \
  > /workspace/Tencent/qwen3_tts_runs/logs/code_predictor_body_s02_pnnx_repro.log 2>&1
```

Expected outputs are written one level up:

- `../code_predictor_body_s02.pnnx.onnx`
- `../code_predictor_body_s02.ncnn.param`
- `../code_predictor_body_s02.ncnn.bin`

Existing multi-length conversion log:

- `/workspace/Tencent/qwen3_tts_runs/logs/code_predictor_body_by_len_pnnx.log`

## PR 1: RotaryEmbed Vulkan cache row stride

Patch:

- `patches/0001-rotaryembed-vulkan-cache-stride.patch`

Changed files:

- `src/layer/vulkan/rotaryembed_vulkan.cpp`
- `src/layer/vulkan/shader/rotaryembed.comp`
- `src/layer/vulkan/shader/rotaryembed_pack4.comp`

Root cause:

- CPU non-interleaved rotary consumes only `halfdim = embed_dim / 2` values from each cos/sin row, but row stride is the real cache width.
- The Vulkan shader used `gy * halfdim + gx`.
- Qwen3-TTS decoder uses cos/sin cache `w=64,h=325`, while `halfdim=32`, so all rows after row 0 read the wrong offset.

Fix:

- Pass `cos_cache.w` as `cos_dim` specialization/push constant.
- Use `gy * cos_dim + gx` in both rotary shaders.

Minimal repro command:

```bash
BIN=/workspace/Tencent/Qwen3-TTS-ncnn/build-vulkan/qwen3_tts_decoder_blob_diff
PARAM=/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/speech_decoder_from_hidden_s325.ncnn.param
MODEL=/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/speech_decoder_from_hidden_s325.ncnn.bin
HIDDEN=/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/ncnn_codes_to_wav_3f_hidden_f32.bin

$BIN "$PARAM" "$MODEL" "$HIDDEN" 3 325 1024 51 4
$BIN "$PARAM" "$MODEL" "$HIDDEN" 3 325 1024 52 4
```

Observed before fix:

- Blob `51` MAE around `0.7099`
- Blob `52` MAE around `0.4784`

Observed after fix:

- Blob `51` MAE `4.00898886e-08`
- Blob `52` MAE `3.16777205e-08`

## PR 2: SDPA Vulkan broadcast mask handling

Patch:

- `patches/0002-sdpa-vulkan-broadcast-mask.patch`

Changed file:

- `src/layer/vulkan/sdpa_vulkan.cpp`

Root cause:

- CPU SDPA broadcasts an attention mask with `dims=3,c=1` across all heads.
- Vulkan SDPA used `attn_mask_blob.dims` directly.
- The shader treats `attn_mask_dims == 3` as per-head mask and offsets by `gz * mask_cstep`.
- For `c=1`, heads after 0 read past the single mask channel, causing future-token leakage.

Fix:

- Add a small helper for shader mask dims.
- If mask is `dims=3,c=1`, report it as non per-head (`2`) to the shader.
- Apply the same helper to FA and non-FA SDPA paths.

Minimal repro command:

```bash
BIN=/workspace/Tencent/Qwen3-TTS-ncnn/build-vulkan/qwen3_tts_decoder_blob_diff
PARAM=/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/speech_decoder_from_hidden_s325.ncnn.param
MODEL=/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/speech_decoder_from_hidden_s325.ncnn.bin
HIDDEN=/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/ncnn_codes_to_wav_3f_hidden_f32.bin

$BIN "$PARAM" "$MODEL" "$HIDDEN" 3 325 1024 62 4 104022
```

Key probe:

- Index `104022` is head 5, row 0, col 22 in the first decoder SDPA output.
- With causal mask, row 0 should attend only to value row 0.

Observed before fix:

- Blob `62` MAE `0.00488107543`
- Probe:
  - CPU `-0.298230499`
  - GPU `0.327211589`

Observed after fix:

- Blob `62` MAE `1.01448435e-08`
- Probe:
  - CPU `-0.298230499`
  - GPU `-0.29823041`

## PR 3: GELU Vulkan honors fast_gelu parameter

Patch:

- `patches/0003-gelu-vulkan-fast-gelu-param.patch`

Changed files:

- `src/layer/vulkan/gelu_vulkan.cpp`
- `src/layer/vulkan/shader/gelu.comp`

Root cause:

- CPU `GELU` has `fast_gelu` param.
- Default `fast_gelu=0` uses the erf/erfc GELU formula.
- Vulkan shader always used the tanh fast-GELU approximation.
- This is numerically different enough for Qwen3-TTS postnet to amplify the error.

Fix:

- Add `fast_gelu` specialization constant to `GELU_vulkan`.
- In `gelu.comp`, use erf-form GELU when `fast_gelu=0`, and keep the existing tanh approximation for `fast_gelu=1`.

Minimal repro command:

```bash
BIN=/workspace/Tencent/Qwen3-TTS-ncnn/build-vulkan/qwen3_tts_decoder_blob_diff
PARAM=/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/speech_decoder_from_hidden_s325.ncnn.param
MODEL=/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/speech_decoder_from_hidden_s325.ncnn.bin
HIDDEN=/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/ncnn_codes_to_wav_3f_hidden_f32.bin

$BIN "$PARAM" "$MODEL" "$HIDDEN" 3 325 1024 352 4
$BIN "$PARAM" "$MODEL" "$HIDDEN" 3 325 1024 353 4
$BIN "$PARAM" "$MODEL" "$HIDDEN" 3 325 1024 354 4
```

Observed before fix:

- Blob `352` MAE `6.43740107e-07`
- Blob `353` after GELU MAE `0.000158915031`
- Blob `354` after following Gemm MAE `0.00314360646`

Observed after fix:

- Blob `353` MAE `9.77839133e-08`
- Blob `354` MAE `2.60294162e-06`

## End-to-end validation after all three fixes

Build:

```bash
cmake --build /workspace/Tencent/ncnn/build-vulkan --target ncnn -j$(nproc)
cmake --build /workspace/Tencent/Qwen3-TTS-ncnn/build-vulkan \
  --target qwen3_tts_decoder_blob_diff qwen3_tts_runtime_codes_to_wav_check qwen3_tts_model_json_check \
  -j$(nproc)
```

Decoder-only original param, 25 frames:

```bash
/workspace/Tencent/Qwen3-TTS-ncnn/build-vulkan/qwen3_tts_runtime_codes_to_wav_check \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/front_weights \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/speech_decoder_from_hidden_s325.ncnn.param \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/speech_decoder_from_hidden_s325.ncnn.bin \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/talker_loop_25f_codes_ref_i32.bin \
  25 0 25 \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/talker_loop_25f_wav_ref_f32.bin \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/runtime_codes_to_wav_25f_vulkan_rotary_sdpa_gelu_fixed_wav_f32.bin \
  4 325 1
```

Observed:

- wav MAE `1.08142368e-07`
- RMSE `2.58423427e-07`
- MaxE `3.69548798e-06`

Full raw Vulkan config:

- `/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/model_vulkan_all_raw_after_ncnn_fixes.json`

Full pipeline, 25 frames:

```bash
/workspace/Tencent/Qwen3-TTS-ncnn/build-vulkan/qwen3_tts_model_json_check \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/model_vulkan_all_raw_after_ncnn_fixes.json \
  25 \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/talker_loop_25f_codes_ref_i32.bin \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/talker_loop_25f_wav_ref_f32.bin \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/runtime_full_25f_vulkan_all_raw_codes_i32.bin \
  /workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/runtime_full_25f_vulkan_all_raw_wav_f32.bin \
  4 1
```

Observed:

- code mismatches `0/400`
- wav MAE `1.08142368e-07`
- RMSE `2.58423427e-07`
- MaxE `3.69548798e-06`

## Notes for PR descriptions

Suggested PR split:

1. `RotaryEmbed_vulkan`: fix cos/sin cache row stride for non-interleaved cache width larger than halfdim.
2. `SDPA_vulkan`: align broadcast mask semantics with CPU when `dims=3,c=1`.
3. `GELU_vulkan`: honor `fast_gelu` param and use erf GELU for default mode.

Each PR can mention Qwen3-TTS as a repro model, but the fixes are general ncnn Vulkan correctness fixes.
