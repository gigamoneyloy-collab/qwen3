# Fit With ncnn_llm

This note compares the current Qwen3-TTS ncnn runtime with the local `ncnn_llm` project.

## Current ncnn_llm Style

The local `ncnn_llm` checkout uses:

```text
src/
examples/
export/
assets/
xmake.lua
```

Key conventions observed:

- Build system is `xmake`, not CMake.
- Core runtime code is placed in `src/`.
- User-facing entry points are placed in `examples/`.
- Conversion scripts are placed in `export/`.
- Model packages live under `assets/<model_name>/`.
- Models are loaded from a model directory, usually `assets/<model>/model.json`.
- JSON is parsed with `nlohmann_json`.
- User CLI style is flag-based:

```text
--model <model_dir>
--threads <n>
--vulkan
--vulkan-device <id>
```

The existing LLM/OCR/ASR paths already use ncnn KV-cache autoregressive decode, optional Vulkan, and small example binaries.

## Current Qwen3-TTS-ncnn Shape

The current standalone repo has:

```text
src/qwen3_tts_runtime.{h,cpp}
src/qwen3_tts_example.cpp
src/*_check.cpp
docs/
CMakeLists.txt
```

After cleanup:

- Public default build:
  - `qwen3_tts_runtime`
  - `qwen3_tts`
- Internal validation/debug tools:
  - only built with `-DQWEN3_TTS_BUILD_TOOLS=ON`
- Model package docs:
  - `docs/MODEL_PACKAGE.md`
- Conversion docs:
  - `docs/CONVERSION.md`
- Public CLI:
  - flag style: `--model`, `--frames`, `--out`, `--codes`, `--threads`, `--vulkan`
  - staged compatibility flags: `--text`, `--speaker`, `--language`

## Fit Assessment

The runtime concept is compatible with `ncnn_llm`:

- It is an ncnn-based autoregressive model runtime.
- It uses KV-cache for the talker decode graph.
- It supports optional Vulkan.
- It uses a model JSON and separate converted assets.
- It has a small example CLI.
- It keeps conversion scripts separate from C++ runtime code.

However, it is not yet a clean `ncnn_llm` PR:

- It uses CMake, while `ncnn_llm` uses `xmake.lua`.
- It parses JSON with a small local string parser instead of `nlohmann_json`.
- Current `model.json` points directly to many files; `ncnn_llm` prefers model-directory-relative layout.
- The standalone runtime now has a C++ CustomVoice frontend and dynamic talker prefill, but the code still needs to be reshaped to `ncnn_llm` naming/build conventions.
- Many validation tools are standalone source files and would not belong in the main `ncnn_llm` target.

## Recommended Upstream Integration Shape

If integrating into `ncnn_llm`, use this shape:

```text
ncnn_llm/
  src/
    ncnn_llm_tts.h
    ncnn_llm_tts.cpp
  examples/
    tts_main.cpp
  export/
    qwen3_tts_export_decoder.py
    qwen3_tts_export_talker.py
    qwen3_tts_export_code_predictor.py
    qwen3_tts_export_front_weights.py
  assets/
    qwen3_tts/
      model.json
      ...
```

Add xmake targets:

```lua
target("tts_main")
    set_kind("binary")
    add_includedirs("examples/")
    add_files("examples/tts_main.cpp")
    add_deps("ncnn_llm")
    add_packages("ncnn", "nlohmann_json")
```

Possible C++ API:

```cpp
ncnn_llm_tts tts("./assets/qwen3_tts", true, 4, 0);
tts.generate_to_wav("你好，欢迎使用通义千问语音合成。", "out.wav", 250);
```

Possible CLI:

```bash
xmake run tts_main --model ./assets/qwen3_tts --text "你好" --out out.wav --frames 250 --vulkan --threads 4
```

## Blocking Work Before a Good ncnn_llm PR

The main model-capability gap has been closed in the standalone runtime:

- C++ tokenizer and CustomVoice prompt construction
- ncnn text/codec embedding nets
- runtime-generated mask/RoPE
- dynamic talker prefill graph shared across prompt lengths
- validated Vulkan parity:
  - short prompt_len=22: `0/400` code mismatch
  - long prompt_len=97: `0/4000` code mismatch

Remaining blockers are integration quality rather than core functionality:

- Convert the build from CMake-only to `xmake.lua` targets.
- Replace the local JSON parser with `nlohmann_json`.
- Make all model paths directory-relative.
- Move the public CLI to `examples/`.
- Hide or remove validation-only binaries from the upstream target.
- Decide whether the three Vulkan operator fixes land in ncnn first or are documented as a required minimum ncnn revision.

## Current Recommendation

Use the `ncnn/` integration in the official `QwenLM/Qwen3-TTS` repository as
the primary issue deliverable.

For `ncnn_llm`:

- Treat it as a reference for project shape and runtime style.
- Do not submit the current standalone code as-is.
- Prepare a later integration branch after:
  - converting config parsing to `nlohmann_json`
  - switching to model-directory-relative paths
  - adding an `xmake.lua` target
  - moving the CLI into `examples/tts_main.cpp`
  - slimming the runtime source set to public API plus one example
