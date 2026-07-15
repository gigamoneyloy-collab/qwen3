# C++ Text Frontend Plan

The current validated runtime starts from precomputed prompt assets:

```text
prefill_embed
prefill_mask
prefill_cos
prefill_sin
```

That proves the ncnn talker/code-predictor/decoder path, but it is not a full C++ text-to-speech pipeline.

For a complete issue deliverable, the runtime should accept:

```text
text + language + speaker
```

and construct the talker prompt in C++.

## Target Scope

Primary target:

```text
Qwen3-TTS-12Hz-0.6B-CustomVoice
```

This is the best first target because built-in CustomVoice speakers do not require a C++ speaker encoder or Mimi reference encoder.

Required C++ frontend pieces:

1. Qwen tokenizer:
   - load `vocab.json` + `merges.txt`, or export an easier `tokenizer.txt`
   - encode `"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"`
2. Text embedding path:
   - export `talker.get_text_embeddings() + talker.text_projection` as ncnn
   - C++ runs text token ids through this net
3. Codec embedding path:
   - export `talker.get_input_embeddings()` as ncnn
   - C++ runs language/speaker/control codec ids through this net
4. Prompt assembly:
   - port `build_nonstream_prefill()` from `/workspace/Tencent/tools/export_qwen3_tts_talker_next.py`
   - for CustomVoice:

```python
role = text_proj(text_embed(input_id[:, :3]))
codec0 = codec_embed([codec_think, codec_think_bos, language_id, codec_think_eos])
speaker_embed = codec_embed([spk_id])
codec1 = codec_embed([codec_pad, codec_bos])
body = [tts_pad * 4, tts_bos] + codec_input[:-1]
text_part = text_proj(text_embed(input_id[:, 3:-5]) + tts_eos) + codec_pad
codec_bos = tts_pad + codec_embed([codec_bos])
prefill = concat(role, body, text_part, codec_bos)
```

5. Mask/RoPE:
   - generate causal mask in C++
   - generate RoPE cos/sin in C++
6. Talker runtime:
   - stop relying on one fixed pnnx prefill graph per prompt length
   - use a KV-cache decoder graph that supports prefill and decode with generated mask/RoPE

## Better Runtime Architecture

Current architecture:

```text
Python prompt exporter -> fixed prefill graph/assets -> C++ AR loop -> wav
```

Target architecture:

```text
C++ tokenizer
  -> text_embed/text_projection ncnn
  -> codec_embed ncnn
  -> C++ prompt assembly
  -> C++ mask/RoPE
  -> talker decoder ncnn with KV cache
  -> C++ code predictor
  -> C++ codec front end + decoder ncnn
  -> wav
```

## Third-Party Baseline

Reviewed:

```text
https://github.com/mingshi2333/Qwen3-TTS-ncnn
```

Observations:

- It already has a C++ BPE tokenizer.
- It exports text/codec embedding nets instead of precomputing prompt embeddings.
- It constructs CustomVoice prompts in C++.
- It has C++ x-vector and ICL paths.
- It has Linux/Windows CI, but CI only runs tokenizer/sampler without model assets.
- Its default FetchContent ncnn build sets `NCNN_VULKAN=OFF`; Vulkan requires a suitable external ncnn build.

Implication:

- To be clearly stronger, we need more than a fixed-prompt runtime.
- The strongest differentiator is full C++ CustomVoice frontend plus our already validated H100 Vulkan 20-second final-audio parity.

## Implemented CustomVoice Frontend

Current implementation files:

```text
src/qwen_bpe.h
src/qwen_bpe.cpp
src/qwen3_tts_frontend.h
src/qwen3_tts_frontend.cpp
src/qwen3_tts_frontend_check.cpp
```

Notes:

- `qwen_bpe.*` is adapted from `mingshi2333/Qwen3-TTS-ncnn` under Apache-2.0.
- `Qwen3TTSFrontend` builds CustomVoice prompt embeddings in C++ from:
  - text
  - language
  - speaker
  - tokenizer file
  - text embedding/projection ncnn net
  - codec embedding ncnn net
- `Qwen3TTSNcnn::generate_codes_from_prompt()` runs the prompt Mat directly through the talker prefill graph and no longer needs `prefill_embed/mask/cos/sin` files for the public text path.

Frontend export helpers:

```text
/workspace/Tencent/tools/export_qwen3_tts_tokenizer_txt.py
/workspace/Tencent/tools/export_qwen3_tts_frontend_nets.py
```

Generated local assets:

```text
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/frontend/tokenizer.txt
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/frontend/talker_text_embed.ncnn.param
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/frontend/talker_text_embed.ncnn.bin
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/frontend/talker_codec_embed.ncnn.param
/workspace/Tencent/qwen3_tts_runs/baseline_001/nonstream/frontend/talker_codec_embed.ncnn.bin
```

Prompt embedding check:

```text
prompt_shape=(22,1024)
got=22528 ref=22528 compare=22528
prompt mae=1.09023479e-08
prompt rmse=3.1012373e-08
prompt maxe=1.66893005e-06
```

25-frame C++ frontend Vulkan check:

```text
frontend prompt_len=22
ref (400,) got (400,) mismatch 0 / 400
```

20-second C++ frontend Vulkan check:

```text
frontend prompt_len=97
generated frames=250
wav_samples=480000
code mismatch 0 / 4000
wav mae=4.05714474e-06
wav rmse=6.609014e-06
wav maxe=0.0001267888
```

Dynamic talker prefill check:

```text
source fixed params: s22 and s97
dynamic params after rewrite: identical
shared talker bin sha256: identical
short prompt_len=22: code mismatch 0 / 400
long prompt_len=97: code mismatch 0 / 4000
long f32 wav mae=4.05714474e-06
long f32 wav rmse=6.609014e-06
long f32 wav maxe=0.0001267888
```

The public text path no longer depends on precomputed prompt embeddings or a prompt-length-specific prefill graph. It uses:

- C++ tokenizer and CustomVoice prompt construction
- ncnn text/codec embedding nets
- runtime-generated mask/RoPE
- dynamic talker prefill param generated by `make_qwen3_tts_talker_prefill_dynamic.py`

## Implementation Steps

Done:

1. Export small frontend nets:
   - `talker_text_embed.ncnn.param/bin`
   - `talker_codec_embed.ncnn.param/bin`
2. Add a C++ tokenizer:
   - either port byte-level BPE
   - or generate `tokenizer.txt` and use a simplified loader
3. Add `Qwen3TTSFrontend`:
   - `encode_assistant_text(text)`
   - `language_id(name)`
   - `speaker_id(name)`
   - `build_customvoice_prompt(text, language, speaker)`
4. Refactor runtime:
   - make talker prefill accept `ncnn::Mat prompt`
   - generate mask/RoPE internally
   - remove `prefill_embed/mask/cos/sin` from public CLI path
5. Validation gates:
   - tokenizer ids vs PyTorch for Chinese/English/mixed text
   - C++ prompt embedding vs Python `build_nonstream_prefill()` max abs diff
   - first main-code argmax vs PyTorch
   - 25-frame free-running code parity
   - 250-frame/20-second final wav parity

Next:

- Move the model schema closer to `ncnn_llm` style with model-directory-relative frontend fields.
- Prepare upstreamable `ncnn_llm` integration after the standalone package layout is cleaned up.

## Later Scope

After CustomVoice is complete:

- Base x-vector voice clone:
  - C++ mel frontend
  - speaker encoder ncnn
- ICL voice clone:
  - Mimi encoder ncnn
  - split-RVQ C++ encoder
