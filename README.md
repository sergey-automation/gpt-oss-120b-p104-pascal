# GPT-OSS-120B on 10× P104-100 Pascal GPUs

Running GPT-OSS-120B Q4_K_M GGUF on inexpensive Pascal mining GPUs using llama.cpp.

Tested configuration:

* 10× NVIDIA P104-100 8GB
* PCIe Gen1 x1
* llama.cpp CUDA backend
* GPT-OSS-120B Q4_K_M GGUF
* Ubuntu 24.04.4 LTS
* CUDA toolkit 12.0
* NVIDIA driver 535.288.01
* Intel Pentium G4620
* 16 GB RAM

The goal of this repository is not to claim optimal performance.

The goal is to document a reproducible real-world setup where a 120B MoE model successfully runs on old Pascal mining GPUs connected through PCIe Gen1 x1.

---

# Repository

Repository name:

```text
gpt-oss-120b-p104-pascal
```

---

# Hardware

## GPUs

```text
10× NVIDIA P104-100
8 GB VRAM each
Pascal GP104
Compute capability 6.1
PCIe Gen1 x1
Power limit: 180 W
```

Detected by:

```text
nvidia-smi --query-gpu=name,pci.bus_id,driver_version,vbios_version,memory.total,pcie.link.gen.current,pcie.link.width.current,power.limit --format=csv
```

Result:

```text
10× NVIDIA P104-100
PCIe Gen1 x1
8192 MiB each
```

## CPU

```text
Intel Pentium G4620
2 cores / 4 threads
3.70 GHz
```

## RAM

```text
16 GB DDR4
```

---

# Software Environment

## OS

```text
Ubuntu 24.04.4 LTS
Kernel 6.17.0-23-generic
```

## CUDA

```text
CUDA toolkit 12.0
Driver 535.288.01
```

## Compiler

```text
GCC 13.3.0
```

## Python

```text
Python 3.12.3
```

## llama.cpp

```text
version: 9100
commit: 2e97c5f96
```

Built with:

```text
GNU 13.3.0 for Linux x86_64
```

---

# Model

Model used:

```text
GPT-OSS-120B
Q4_K_M GGUF
```

GGUF metadata:

```text
Architecture: gpt-oss
Layers: 36
Context length: 131072
Experts per layer: 128
Active experts: 4
Attention heads: 64
KV heads: 8
Embedding size: 2880
Model params: 116.83B
```

Quantization types inside GGUF:

```text
f32
q5_0
q8_0
q4_K
mxfp4
```

GGUF size:

```text
58.45 GiB
```

---

# Important Observations

## Full GPU Offload

llama.cpp successfully offloaded the full model to GPU:

```text
offloaded 37/37 layers to GPU
```

No CPU inference fallback was observed.

---

## mmap=true

The model was loaded with:

```text
mmap = true
```

This allows the GGUF file to remain memory-mapped instead of fully copied into RAM.

System RAM usage remained low despite running a 120B model on a machine with only 16 GB RAM.

---

## CPU_Mapped model buffer

llama.cpp reported:

```text
CPU_Mapped model buffer size = 379 MiB
```

This does not mean CPU inference.

The model remained fully GPU offloaded.

---

## KV Cache Reservation

KV cache memory was reserved in advance.

VRAM usage remained nearly constant during long prompts.

Example:

```text
GPU1:
ctx256    -> 7678 MiB
ctx32768  -> 7704 MiB
```

VRAM growth:

```text
~26 MiB
```

This indicates that llama.cpp preallocated KV cache memory for the configured context/parallel slots.

---

## PARALLEL and Context Splitting

llama.cpp splits the global context between parallel slots.

Observed:

```text
PARALLEL=2 -> n_ctx_seq ≈ 66048
PARALLEL=4 -> n_ctx_seq ≈ 33024
PARALLEL=6 -> n_ctx_seq ≈ 22016
```

---

# Benchmark Configuration

Main benchmark configuration:

```text
CTX_SIZE=131072
PARALLEL=4
CTK=f16
BATCH=2048
UBATCH=1024
split-mode=layer
```

Tensor split:

```text
2/4/4/4/4/4/4/4/3/3
```

The setup successfully started and completed benchmark runs despite:

```text
cannot meet free memory targets on all devices
```

This indicates that llama.cpp "target free memory" is a safety heuristic rather than a hard limit.

---

# Memory Usage

Projected VRAM usage:

```text
70480 MiB used
80255 MiB available
```

Free VRAM after allocation:

```text
GPU0  -> 2008 MiB free
GPU1  ->  421 MiB free
GPU2  ->  422 MiB free
GPU3  ->  422 MiB free
GPU4  ->  421 MiB free
GPU5  ->  422 MiB free
GPU6  ->  422 MiB free
GPU7  ->  421 MiB free
GPU8  -> 2066 MiB free
GPU9  -> 2748 MiB free
```

---

# Benchmark Results

| Prompt tokens | Prefill tok/s | Decode tok/s |
| ------------- | ------------: | -----------: |
| 255           |        114.72 |        28.88 |
| 504           |        214.41 |        30.02 |
| 1017          |        269.97 |        29.54 |
| 2044          |        271.72 |        29.36 |
| 4076          |        282.76 |        28.09 |
| 6142          |        273.66 |        28.18 |
| 8164          |        259.87 |        27.62 |
| 12280         |        234.57 |        26.50 |
| 16373         |        214.83 |        25.49 |
| 24560         |        182.66 |        23.78 |
| 32761         |        158.17 |        21.97 |
| 64423         |        104.89 |        18.14 |

---

# GPU Utilization

Average utilization:

| Stage   | Average GPU Util |
| ------- | ---------------: |
| Prefill |          ~11–12% |
| Decode  |          ~10–11% |

This low utilization does not prove PCIe bandwidth bottleneck.

The system still achieved:

```text
~283 tok/s prefill
~30 tok/s decode
```

on:

```text
10× Pascal GPUs
PCIe Gen1 x1
```

Observed behavior suggests pipeline imbalance and synchronization overhead rather than raw PCIe throughput limitation.

GPU0 was often the busiest during prefill.

GPU7–GPU9 were often the busiest during decode.

---

# Power Consumption

Measured wall power:

| State                   |  Power |
| ----------------------- | -----: |
| Idle                    | ~123 W |
| Before VRAM loading     | ~128 W |
| Model loading into VRAM | ~222 W |
| Prefill stage           | ~680 W |
| Decode stage            | ~580 W |

During model loading, VRAM writes occurred mostly sequentially GPU-by-GPU.

---

# Key Result

This repository demonstrates that:

```text
GPT-OSS-120B Q4_K_M
can run on
10× Pascal P104-100 8GB
through PCIe Gen1 x1
using llama.cpp CUDA backend
```

The setup is not optimized for maximum performance.

The focus is reproducibility and demonstrating that large MoE inference is technically possible on inexpensive older hardware.
