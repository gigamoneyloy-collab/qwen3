# Feature Matrix

This matrix compares the public ncnn runtime with the official Qwen3-TTS Python
repository. It is intentionally strict: a feature is marked done only when the
repo has code, reproducible instructions, and at least a smoke validation path.

| Official Qwen3-TTS feature | ncnn repo status | Evidence | Remaining work |
| --- | --- | --- | --- |
| 12Hz 0.6B CustomVoice fixed reference prompt | Implemented | Development parity logs show `0/400` code mismatches for 25 frames and `0/4000` for the 250-frame reference-asset run | Keep regression assets external to avoid committing weights/dumps |
| 12Hz 0.6B CustomVoice C++ dynamic frontend | Partial | Short and longer English `Vivian` prompts produce audible checked-in wavs; prompt matrix covers all 10 explicit languages and all 9 speakers with max error `<=2.38418579e-7`; audio smoke covers English/Chinese/Japanese/Korean end to end | Add longer generation/audio quality regression cases beyond deterministic 25-frame smoke |
| Supported CustomVoice speakers | Prompt-level implemented | Prompt matrix covers all 9 built-in speakers: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee | Add generated-audio smoke tests for every speaker |
| Supported languages | Prompt-level implemented | Prompt matrix covers Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian, plus Eric/Dylan Chinese dialect routing; audio smoke confirms UTF-8 text-file generation for English/Chinese/Japanese/Korean | Implement and validate official `Auto` no-think behavior |
| CustomVoice `instruct` | Not implemented | CLI has no `--instruct`; 0.6B official wrapper disables instruct | Add only if targeting 1.7B CustomVoice and validate prompt construction |
| Batch inference | Not implemented | CLI processes one request at a time | Add batch runner once single-sample parity is stable |
| Streaming / low-latency generation | Not implemented | Runtime decodes requested frames after code generation | Design chunked generation/decode API and measure first-audio latency |
| VoiceDesign | Not implemented | No exported VoiceDesign model path or frontend prompt builder | Export 1.7B VoiceDesign graphs and implement design-instruct prompt path |
| Base voice clone | Not implemented | No speaker encoder/reference audio path | Export speaker encoder/reference codec path and implement `create_voice_clone_prompt` equivalent |
| Reusable voice clone prompt | Not implemented | No serialized voice prompt format | Define portable prompt artifact after Base clone works |
| Official Gradio demo parity | Not implemented | Repo is C++ CLI/runtime only | Build a demo wrapper after core feature parity |
| ncnn_llm upstream PR shape | Partial | Standalone runtime and operator repro docs exist | Reshape names/build/API to upstream conventions after behavior is stable |

## Current Acceptance Bar

The repo is suitable as a reproducible standalone CustomVoice/ncnn workbench
only after the dynamic frontend claims are validated honestly. It is not yet
ready to claim full official Qwen3-TTS parity or to submit as a complete
`ncnn_llm` feature PR.
