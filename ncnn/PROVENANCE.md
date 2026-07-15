# Provenance and attribution

This `ncnn/` integration is submitted to the official
[`QwenLM/Qwen3-TTS`](https://github.com/QwenLM/Qwen3-TTS) repository as an
independent, squashed Git commit. It is not represented as wholly original
work.

The initial runtime, conversion tools, documentation, validation assets, and
Apache-2.0 license were derived from
[`LudovicoYIN/Qwen3-TTS-ncnn`](https://github.com/LudovicoYIN/Qwen3-TTS-ncnn)
at commit `9bc18b5794e7648a7a410abac457d3d9f045f4b8`. Earlier third-party adaptations
are identified in the source comments and existing project documentation.

The official-repository integration adds and maintains a Windows + WSL build
path, installed-ncnn package integration, CLI smoke test, and reproducible
validation evidence. The derived implementation is isolated under `ncnn/`;
the upstream Apache-2.0 license and source comments are preserved.
