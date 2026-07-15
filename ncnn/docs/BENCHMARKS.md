# Benchmarks

Benchmarks are reported separately from model load time whenever possible.
GPU clock and power state have a large effect on Vulkan timings, so treat single
runs without clock information as rough samples.

## Windows RTX 3090

Current dynamic frontend smoke runs:

| Case | Prompt length | Frames | Audio | Wall time | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Short English dynamic frontend | 21 | 25 | 2.0s | about 10.9s | Includes process start, model load, generation, wav write; audible sample checked in |
| Long English dynamic frontend | 46 | 25 | 2.0s | about 15.0s | Includes process start, model load, generation, wav write; `0/400` code mismatch vs PyTorch reference |

An earlier long English text produced a file that sounded silent. Follow-up
parity showed ncnn and PyTorch generated identical codes for that text, and the
PyTorch reference wav was also near-silent for the first 25 frames. That text is
therefore not used as a demo sample.

Loaded-once benchmark with the KV-shared package on the same machine:

```text
load_ms,4147.477
warmup generate_ms=4559.521 wav_ms=610.585 total_ms=5170.106
repeat generate_ms=4380.118 wav_ms=606.416 total_ms=4986.534
repeat generate_ms=4389.545 wav_ms=589.223 total_ms=4978.768
repeat generate_ms=4384.528 wav_ms=587.202 total_ms=4971.730
generate mean_ms=4384.730 median_ms=4384.528 min_ms=4380.118 max_ms=4389.545
wav mean_ms=594.281 median_ms=589.223 min_ms=587.202 max_ms=606.416
total mean_ms=4979.011 median_ms=4978.768 min_ms=4971.730 max_ms=4986.534
```

Earlier hot-clock Windows observation:

```text
PyTorch CUDA fp32 generate: about 3.797s for the 25-frame class run
ncnn exact KV-shared hot generate: about 3.95-4.00s
```

Recent low-clock polluted samples were slower; RTX 3090 was observed around
`810 MHz / 34 W` and even `210 MHz / 28 W` in later runs. Do not compare those
directly against hot-clock PyTorch numbers.

## Linux H100

Historical validation/benchmark summary:

| Case | Frames | Audio | Time |
| --- | ---: | ---: | ---: |
| ncnn CPU | 25 | about 2s | 45.10s |
| ncnn Vulkan | 25 | about 2s | 34.12s |
| PyTorch CPU fp32 | 25 max | about 1.92s | 3.19s |
| PyTorch CUDA fp32 | 25 max | about 1.92s | 2.88s |
| PyTorch CUDA bf16 | 25 max | about 1.92s | 6.49s |
| PyTorch CPU fp32 | 250 max | about 19.92s | 33.71s |
| PyTorch CUDA fp32 | 250 max | about 19.92s | 22.67s |

The current project prioritizes exact FP32 parity first. Further speed work is
tracked in the progress notes and should focus on fused code predictor execution.
