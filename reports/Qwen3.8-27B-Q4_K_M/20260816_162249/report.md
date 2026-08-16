# llama-server parallel-slots context benchmark report

## Test header

- MODEL: `/mnt/models/AI/LLM/Qwen3.8-27B-Q4_K_M.gguf`
- NGL: `999`
- CTX_SIZE: `200000`
- N_GEN: `128`
- BATCH: `4097`
- UBATCH: `32`
- CTK: `f16`
- CTV: `f16`
- SPEC_TYPE: `draft-mtp`
- SPEC_DRAFT_N_MAX: `3`
- SPLIT_MODE: `layer`
- TENSOR_SPLIT: `18/18/17/15`
- PARALLEL: `1`
- TEMPERATURE: `0.15`
- CACHE_PROMPT: `0`
- FLASH_ATTN: `auto`
- THREADS: `auto`
- THREADS_BATCH: `auto`
- REPEATS: `1`
- CUDA_VISIBLE_DEVICES: `0,1,2,3,4,5`
- TURBOPREFILL: `0`
- TurboPrefill status: `TurboPrefill implementation detected; inactive (TURBOPREFILL=0)`
- TurboPrefill version: `TurboPrefill`
- llama.cpp git describe: `b10451-1-gba0d2b391-dirty`
- llama.cpp git commit: `ba0d2b3918c4662d8a1fb2eee21c365265f2901f`
- Server PID: `217879`
- KEEP_SERVER_RUNNING: `1`
- Parallel-slots mode: `active_slots=1..PARALLEL`
- Metrics policy: `server per-request timings only; no combined throughput calculated`
- llama_server_log: `/home/serg/workspace/versions/TurboPrefill_b10451/bench_reports_Qwen3.8-27B-Q4_K_M/20260816_162249/llama_server.log`

## Environment

### TURBOPREFILL

```text
0
```

### RUN_DIR

```text
/home/serg/workspace/versions/TurboPrefill_b10451
```

### CONFIG_PATH

```text
/home/serg/workspace/versions/TurboPrefill_b10451/config_Qwen_3.8.sh
```

### LLAMA_SERVER_BIN

```text
/home/serg/workspace/versions/TurboPrefill_b10451/build/bin/llama-server
```

### LOCAL_LD_LIBRARY_PATH

```text
/home/serg/workspace/versions/TurboPrefill_b10451/build/bin
```

### CUDA_VISIBLE_DEVICES_effective

```text
0,1,2,3,4,5
```

### selected_gpu_count

```text
6
```

### selected_gpu_models

```text
NVIDIA P104-100 x6
```

### llama_server_version

```text
version: 0.1.0-dev (build 1, commit 10bf611)
built with GNU 13.3.0 for Linux x86_64
```

### uname

```text
Linux turboprefill 6.17.0-29-generic #29~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon May 11 10:30:58 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
```

### lscpu

```text
Архитектура:                             x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           39 bits physical, 48 bits virtual
Порядок байт:                            Little Endian
CPU(s):                                  4
On-line CPU(s) list:                     0-3
ID прроизводителя:                       GenuineIntel
Имя модели:                              Intel(R) Core(TM) i5-7500 CPU @ 3.40GHz
Семейство ЦПУ:                           6
Модель:                                  158
Потоков на ядро:                         1
Ядер на сокет:                           4
Сокетов:                                 1
Степпинг:                                9
CPU(s) scaling MHz:                      96%
CPU max MHz:                             3800.0000
CPU min MHz:                             800.0000
BogoMIPS:                                6799.81
Флаги:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault pti ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp md_clear flush_l1d arch_capabilities
L1d cache:                               128 KiB (4 instances)
L1i cache:                               128 KiB (4 instances)
L2 cache:                                1 MiB (4 instances)
L3 cache:                                6 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-3
Vulnerability Gather data sampling:      Vulnerable
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             KVM: Mitigation: VMX unsupported
Vulnerability L1tf:                      Mitigation; PTE Inversion
Vulnerability Mds:                       Mitigation; Clear CPU buffers; SMT disabled
Vulnerability Meltdown:                  Mitigation; PTI
Vulnerability Mmio stale data:           Mitigation; Clear CPU buffers; SMT disabled
Vulnerability Old microcode:             Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Mitigation; IBRS
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; IBRS; IBPB conditional; STIBP disabled; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                     Mitigation; Microcode
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Mitigation; TSX disabled
Vulnerability Vmscape:                   Mitigation; IBPB before exit to userspace
```

### motherboard_vendor

```text
ASUSTeK COMPUTER INC.
```

### motherboard_name

```text
B250 MINING EXPERT
```

### motherboard_version

```text
Rev X.0x
```

### memory_summary

```text
всего        занят        своб      общая  буф/врем.   доступно
Память:         15Gi       959Mi       946Mi       144Ki        13Gi        14Gi
Подкачка:       23Gi       117Mi        23Gi
```

### memory_modules

```text
unavailable: Command '['dmidecode', '--type', '17']' returned non-zero exit status 1.
```

### nvidia_smi

```text
0, NVIDIA P104-100, 00000000:01:00.0, 535.309.01, 8192 MiB, 1, 1
1, NVIDIA P104-100, 00000000:02:00.0, 535.309.01, 8192 MiB, 1, 1
2, NVIDIA P104-100, 00000000:0A:00.0, 535.309.01, 8192 MiB, 1, 1
3, NVIDIA P104-100, 00000000:0C:00.0, 535.309.01, 8192 MiB, 1, 1
4, NVIDIA P104-100, 00000000:0D:00.0, 535.309.01, 8192 MiB, 1, 1
5, NVIDIA P104-100, 00000000:0E:00.0, 535.309.01, 8192 MiB, 1, 1
6, NVIDIA P104-100, 00000000:0F:00.0, 535.309.01, 8192 MiB, 1, 1
7, NVIDIA P104-100, 00000000:10:00.0, 535.309.01, 8192 MiB, 1, 1
8, NVIDIA P104-100, 00000000:11:00.0, 535.309.01, 8192 MiB, 1, 1
9, NVIDIA P104-100, 00000000:12:00.0, 535.309.01, 8192 MiB, 1, 1
10, NVIDIA P104-100, 00000000:13:00.0, 535.309.01, 8192 MiB, 1, 1
11, NVIDIA P104-100, 00000000:14:00.0, 535.309.01, 8192 MiB, 1, 1
```

### nvcc

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0
```

### cmake

```text
cmake version 3.28.3

CMake suite maintained and supported by Kitware (kitware.com/cmake).
```

### git_describe

```text
b10451-1-gba0d2b391-dirty
```

### git_commit

```text
ba0d2b3918c4662d8a1fb2eee21c365265f2901f
```

### git_last_commit

```text
2026-08-16 13:24:17 +0300
Port TurboPrefill v2.1.3 to llama.cpp b10451
```

### git_turboprefill_hint

```text
ba0d2b3918c4662d8a1fb2eee21c365265f2901f Port TurboPrefill v2.1.3 to llama.cpp b10451
```

### model_path

```text
/mnt/models/AI/LLM/Qwen3.8-27B-Q4_K_M.gguf
```

### model_filename

```text
Qwen3.8-27B-Q4_K_M.gguf
```

### model_size_bytes

```text
16810714432
```

### model_size_gib

```text
15.656
```

### gguf_architecture

```text
qwen35
```

### gguf_tensor_count

```text
866
```

### gguf_tensor_types

```text
F32:360, Q4_K:439, Q6_K:67
```

### model_sha256

```text
disabled (MODEL_HASH=0)
```

### TurboPrefill runtime markers

```text
4.00.170.004 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
4.38.229.991 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
4.39.912.804 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
5.18.388.757 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
5.20.289.231 I decode: TurboPrefill requested=0 active=0 n_tokens=623 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
5.26.463.180 I decode: TurboPrefill requested=0 active=0 n_tokens=623 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
5.43.233.695 I decode: TurboPrefill requested=0 active=0 n_tokens=232 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
5.45.625.268 I decode: TurboPrefill requested=0 active=0 n_tokens=232 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
5.58.785.482 I decode: TurboPrefill requested=0 active=0 n_tokens=505 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
6.02.984.321 I decode: TurboPrefill requested=0 active=0 n_tokens=505 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
6.15.857.301 I decode: TurboPrefill requested=0 active=0 n_tokens=1046 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
6.25.039.758 I decode: TurboPrefill requested=0 active=0 n_tokens=1046 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
6.41.300.712 I decode: TurboPrefill requested=0 active=0 n_tokens=2294 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
7.01.343.358 I decode: TurboPrefill requested=0 active=0 n_tokens=2294 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
7.15.538.458 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
7.51.480.629 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
7.53.102.650 I decode: TurboPrefill requested=0 active=0 n_tokens=155 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
7.54.469.576 I decode: TurboPrefill requested=0 active=0 n_tokens=155 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
8.13.562.303 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
8.49.427.223 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
```

## Server command

```bash
/home/serg/workspace/versions/TurboPrefill_b10451/build/bin/llama-server -m /mnt/models/AI/LLM/Qwen3.8-27B-Q4_K_M.gguf --host 0.0.0.0 --port 8081 -lv 4 -ngl 999 -c 200000 --override-kv llama.context_length=int:200000 -b 4097 -ub 32 -np 1 -ctk f16 -ctv f16 -sm layer -ts 18/18/17/15 --flash-attn auto --no-warmup --no-mmproj --spec-type draft-mtp --spec-draft-n-max 3
```

Server PID: `217879`  
Stop command: `kill -INT 217879`

## Summary

| File | Active slots | Request | Prompt tokens | Completion tokens | Prefill tok/s | Prefill time s | Decode tok/s | Decode time s | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ctx_000256.txt | 1 | 1 | 268 | 128 | 57.60 | 4.65 | 13.09 | 9.70 | 18.72 |
| ctx_000512.txt | 1 | 1 | 541 | 128 | 82.46 | 6.56 | 13.76 | 9.23 | 16.97 |
| ctx_001024.txt | 1 | 1 | 1082 | 128 | 91.88 | 11.78 | 10.42 | 12.19 | 25.24 |
| ctx_002048.txt | 1 | 1 | 2330 | 128 | 100.57 | 23.17 | 13.91 | 9.13 | 33.78 |
| ctx_004096.txt | 1 | 1 | 4288 | 128 | 103.68 | 41.36 | 12.80 | 9.93 | 53.20 |
| ctx_008192.txt | 1 | 1 | 8853 | 128 | 101.91 | 86.87 | 13.77 | 9.22 | 102.81 |
| ctx_016384.txt | 1 | 1 | 17670 | 128 | 95.72 | 184.60 | 12.79 | 9.93 | 198.95 |

## GPU load by stage

### ctx_000256.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 75.0 | 100.0 | 67.8 | 83.8 | 7512 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 23.5 | 47.0 | 53.1 | 53.6 | 7120 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 2.5 | 5.0 | 116.8 | 182.1 | 7860 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 54.7 | 54.9 | 7490 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 8.9 | 84 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.7 | 9.8 | 84 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 17.8 | 27.0 | 85.4 | 179.2 | 7512 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 17.0 | 25.0 | 134.7 | 193.3 | 7120 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 24.5 | 72.0 | 54.7 | 79.2 | 7860 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 46.7 | 75.0 | 68.8 | 143.6 | 7490 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 8.9 | 84 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 10.1 | 84 |

### ctx_000512.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 14.3 | 18.0 | 92.7 | 174.2 | 7512 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 13.7 | 26.0 | 53.0 | 53.1 | 7120 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 46.7 | 47.0 | 52.4 | 53.3 | 7860 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 14.0 | 42.0 | 86.8 | 150.8 | 7490 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.8 | 8.9 | 84 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.7 | 9.8 | 84 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 2.2 | 5.0 | 50.6 | 55.5 | 7512 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 18.0 | 22.0 | 85.0 | 182.2 | 7120 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 13.5 | 18.0 | 91.6 | 158.5 | 7860 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 33.5 | 48.0 | 107.7 | 169.7 | 7490 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.8 | 8.9 | 84 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.7 | 9.8 | 84 |

### ctx_001024.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 10.4 | 50.0 | 78.6 | 168.0 | 7512 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 18.8 | 47.0 | 73.8 | 155.8 | 7120 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 23.6 | 47.0 | 79.4 | 183.3 | 7860 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 25.6 | 98.0 | 91.4 | 180.4 | 7490 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 9.1 | 84 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.7 | 9.9 | 84 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 26.0 | 30.0 | 65.6 | 109.0 | 7512 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 14.0 | 28.0 | 70.6 | 154.3 | 7120 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 26.8 | 28.0 | 67.6 | 125.0 | 7860 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 39.3 | 55.0 | 84.2 | 135.7 | 7490 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 9.2 | 84 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.7 | 10.0 | 84 |

### ctx_002048.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 36.1 | 93.0 | 90.4 | 176.4 | 7512 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 29.8 | 80.0 | 75.7 | 179.2 | 7120 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 35.2 | 65.0 | 66.6 | 175.8 | 7860 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 18.1 | 67.0 | 108.4 | 180.8 | 7490 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 9.1 | 84 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.7 | 10.0 | 84 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 21.2 | 27.0 | 51.6 | 54.5 | 7512 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 21.0 | 25.0 | 65.0 | 119.4 | 7120 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 12.2 | 25.0 | 67.6 | 120.0 | 7860 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 42.4 | 53.0 | 119.6 | 213.8 | 7490 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 8.9 | 84 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 9.9 | 84 |

### ctx_004096.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 32.8 | 95.0 | 76.5 | 175.9 | 7512 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 35.1 | 94.0 | 80.5 | 185.9 | 7120 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 29.9 | 88.0 | 90.1 | 180.7 | 7860 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 29.1 | 98.0 | 88.3 | 162.2 | 7490 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 9.2 | 84 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.7 | 10.2 | 84 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 21.8 | 30.0 | 46.4 | 52.0 | 7512 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 20.0 | 29.0 | 59.4 | 109.4 | 7120 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 15.4 | 26.0 | 82.1 | 136.3 | 7860 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 36.8 | 70.0 | 93.0 | 209.7 | 7490 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 8.9 | 84 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 10.0 | 84 |

### ctx_008192.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 26.6 | 72.0 | 91.6 | 179.9 | 7512 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 28.0 | 92.0 | 88.2 | 185.1 | 7120 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 20.6 | 70.0 | 80.9 | 185.3 | 7860 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 27.7 | 98.0 | 70.5 | 187.5 | 7490 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 9.2 | 84 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 10.1 | 84 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 32.4 | 87.0 | 102.3 | 171.3 | 7512 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 24.3 | 31.0 | 52.8 | 71.4 | 7120 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 21.4 | 54.0 | 48.0 | 54.3 | 7860 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 40.0 | 54.0 | 79.5 | 163.2 | 7490 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 8.9 | 84 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 10.0 | 84 |

### ctx_016384.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 30.1 | 89.0 | 77.9 | 174.7 | 7512 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 26.4 | 96.0 | 87.3 | 189.3 | 7120 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 30.2 | 99.0 | 84.5 | 185.3 | 7860 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 25.6 | 98.0 | 88.3 | 195.1 | 7490 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 10.2 | 84 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 10.6 | 84 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 27.9 | 78.0 | 74.3 | 167.2 | 7512 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 16.7 | 29.0 | 92.6 | 182.0 | 7120 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 16.6 | 29.0 | 70.6 | 150.8 | 7860 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 39.1 | 79.0 | 83.0 | 173.1 | 7490 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 9.1 | 84 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 10.5 | 84 |


## CPU / RAM / swap load by stage

### ctx_000256.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 47.0 %, max 49.5 %

RAM used: avg 3613 MiB, max 3794 MiB, avg 22.8 %, max 23.9 %

Swap used: avg 372 MiB, max 372 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 26.0 | 28.2 |
| 1 | 24.9 | 27.8 |
| 2 | 75.5 | 97.4 |
| 3 | 62.0 | 94.9 |

Decode stage:

CPU total: avg 40.2 %, max 41.8 %

RAM used: avg 3742 MiB, max 3841 MiB, avg 23.6 %, max 24.2 %

Swap used: avg 372 MiB, max 372 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 57.0 | 100.0 |
| 1 | 23.5 | 26.3 |
| 2 | 25.2 | 45.0 |
| 3 | 55.0 | 100.0 |

### ctx_000512.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 40.1 %, max 41.0 %

RAM used: avg 4231 MiB, max 4268 MiB, avg 26.7 %, max 26.9 %

Swap used: avg 372 MiB, max 372 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 18.7 | 23.6 |
| 1 | 26.8 | 31.1 |
| 2 | 16.0 | 19.3 |
| 3 | 99.1 | 100.0 |

Decode stage:

CPU total: avg 43.2 %, max 49.2 %

RAM used: avg 4256 MiB, max 4313 MiB, avg 26.8 %, max 27.2 %

Swap used: avg 372 MiB, max 372 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 99.5 | 100.0 |
| 1 | 20.5 | 27.1 |
| 2 | 24.1 | 32.6 |
| 3 | 28.5 | 55.5 |

### ctx_001024.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 38.8 %, max 39.5 %

RAM used: avg 4396 MiB, max 4466 MiB, avg 27.7 %, max 28.1 %

Swap used: avg 372 MiB, max 372 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 33.1 | 95.3 |
| 1 | 16.1 | 25.0 |
| 2 | 23.7 | 44.3 |
| 3 | 81.8 | 100.0 |

Decode stage:

CPU total: avg 43.1 %, max 52.0 %

RAM used: avg 4777 MiB, max 4796 MiB, avg 30.1 %, max 30.2 %

Swap used: avg 372 MiB, max 372 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 22.8 | 42.2 |
| 1 | 85.8 | 100.0 |
| 2 | 20.4 | 27.7 |
| 3 | 43.3 | 100.0 |

### ctx_002048.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 40.1 %, max 53.4 %

RAM used: avg 5114 MiB, max 5248 MiB, avg 32.2 %, max 33.1 %

Swap used: avg 371 MiB, max 371 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 48.8 | 100.0 |
| 1 | 64.0 | 100.0 |
| 2 | 20.9 | 33.5 |
| 3 | 26.6 | 88.5 |

Decode stage:

CPU total: avg 41.0 %, max 41.4 %

RAM used: avg 5230 MiB, max 5305 MiB, avg 32.9 %, max 33.4 %

Swap used: avg 371 MiB, max 371 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 54.1 | 100.0 |
| 1 | 24.8 | 34.3 |
| 2 | 19.2 | 20.6 |
| 3 | 65.8 | 100.0 |

### ctx_004096.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 40.9 %, max 54.9 %

RAM used: avg 5744 MiB, max 5850 MiB, avg 36.2 %, max 36.9 %

Swap used: avg 371 MiB, max 371 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 17.4 | 32.3 |
| 1 | 25.1 | 78.5 |
| 2 | 52.1 | 100.0 |
| 3 | 68.8 | 100.0 |

Decode stage:

CPU total: avg 42.8 %, max 48.6 %

RAM used: avg 5937 MiB, max 6004 MiB, avg 37.4 %, max 37.8 %

Swap used: avg 371 MiB, max 371 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 21.7 | 25.7 |
| 1 | 19.7 | 26.3 |
| 2 | 75.1 | 100.0 |
| 3 | 54.4 | 100.0 |

### ctx_008192.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 40.6 %, max 51.3 %

RAM used: avg 5397 MiB, max 6681 MiB, avg 34.0 %, max 42.1 %

Swap used: avg 370 MiB, max 370 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 41.7 | 100.0 |
| 1 | 58.4 | 100.0 |
| 2 | 41.0 | 100.0 |
| 3 | 21.2 | 34.9 |

Decode stage:

CPU total: avg 42.3 %, max 49.1 %

RAM used: avg 5503 MiB, max 5647 MiB, avg 34.7 %, max 35.6 %

Swap used: avg 370 MiB, max 370 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 32.4 | 59.9 |
| 1 | 17.9 | 25.3 |
| 2 | 100.0 | 100.0 |
| 3 | 18.4 | 23.6 |

### ctx_016384.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 40.9 %, max 53.7 %

RAM used: avg 6441 MiB, max 6682 MiB, avg 40.6 %, max 42.1 %

Swap used: avg 370 MiB, max 370 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 62.0 | 100.0 |
| 1 | 20.7 | 38.3 |
| 2 | 19.1 | 41.4 |
| 3 | 61.5 | 100.0 |

Decode stage:

CPU total: avg 41.9 %, max 48.2 %

RAM used: avg 6754 MiB, max 6883 MiB, avg 42.5 %, max 43.4 %

Swap used: avg 370 MiB, max 370 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 20.2 | 25.0 |
| 1 | 46.0 | 100.0 |
| 2 | 18.8 | 28.2 |
| 3 | 82.5 | 100.0 |

