# llama-server parallel-slots context benchmark report

## Test header

- MODEL: `/mnt/models/AI/LLM/Qwen3.8-27B-Q4_K_M.gguf`
- NGL: `999`
- CTX_SIZE: `260000`
- N_GEN: `128`
- BATCH: `4097`
- UBATCH: `32`
- CTK: `f16`
- CTV: `f16`
- SPEC_TYPE: `draft-mtp`
- SPEC_DRAFT_N_MAX: `3`
- SPLIT_MODE: `layer`
- TENSOR_SPLIT: `11/11/11/11/11/9`
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
- Server PID: `160706`
- KEEP_SERVER_RUNNING: `1`
- Parallel-slots mode: `active_slots=1..PARALLEL`
- Metrics policy: `server per-request timings only; no combined throughput calculated`
- llama_server_log: `/home/serg/workspace/versions/TurboPrefill_b10451/bench_reports_Qwen3.8-27B-Q4_K_M/20260816_143508/llama_server.log`

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
CPU(s) scaling MHz:                      97%
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
Память:         15Gi       6.7Gi       318Mi       279Mi       9.1Gi       8.8Gi
Подкачка:       23Gi       298Mi        23Gi
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
7.21.746.548 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
8.03.195.191 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
8.04.866.972 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
8.44.399.649 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
8.46.297.132 I decode: TurboPrefill requested=0 active=0 n_tokens=623 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
8.52.700.766 I decode: TurboPrefill requested=0 active=0 n_tokens=623 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
9.09.328.136 I decode: TurboPrefill requested=0 active=0 n_tokens=232 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
9.11.455.333 I decode: TurboPrefill requested=0 active=0 n_tokens=232 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
9.24.540.252 I decode: TurboPrefill requested=0 active=0 n_tokens=505 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
9.28.778.083 I decode: TurboPrefill requested=0 active=0 n_tokens=505 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
9.41.682.821 I decode: TurboPrefill requested=0 active=0 n_tokens=1046 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
9.50.920.630 I decode: TurboPrefill requested=0 active=0 n_tokens=1046 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
10.07.146.255 I decode: TurboPrefill requested=0 active=0 n_tokens=2294 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
10.27.681.399 I decode: TurboPrefill requested=0 active=0 n_tokens=2294 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
10.41.990.095 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
11.19.041.235 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
11.20.663.143 I decode: TurboPrefill requested=0 active=0 n_tokens=155 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
11.22.089.758 I decode: TurboPrefill requested=0 active=0 n_tokens=155 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
11.40.915.358 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
12.17.859.528 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
```

## Server command

```bash
/home/serg/workspace/versions/TurboPrefill_b10451/build/bin/llama-server -m /mnt/models/AI/LLM/Qwen3.8-27B-Q4_K_M.gguf --host 0.0.0.0 --port 8081 -lv 4 -ngl 999 -c 260000 --override-kv llama.context_length=int:260000 -b 4097 -ub 32 -np 1 -ctk f16 -ctv f16 -sm layer -ts 11/11/11/11/11/9 --flash-attn auto --no-warmup --no-mmproj --spec-type draft-mtp --spec-draft-n-max 3
```

Server PID: `160706`  
Stop command: `kill -INT 160706`

## Summary

| File | Active slots | Request | Prompt tokens | Completion tokens | Prefill tok/s | Prefill time s | Decode tok/s | Decode time s | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ctx_000256.txt | 1 | 1 | 268 | 128 | 60.97 | 4.40 | 13.16 | 9.65 | 18.39 |
| ctx_000512.txt | 1 | 1 | 541 | 128 | 81.85 | 6.61 | 13.71 | 9.26 | 17.02 |
| ctx_001024.txt | 1 | 1 | 1082 | 128 | 91.42 | 11.84 | 10.46 | 12.15 | 25.25 |
| ctx_002048.txt | 1 | 1 | 2330 | 128 | 98.44 | 23.67 | 13.81 | 9.20 | 34.35 |
| ctx_004096.txt | 1 | 1 | 4288 | 128 | 100.80 | 42.54 | 13.13 | 9.67 | 54.13 |
| ctx_008192.txt | 1 | 1 | 8853 | 128 | 99.24 | 89.21 | 13.67 | 9.29 | 105.19 |
| ctx_016384.txt | 1 | 1 | 17670 | 128 | 93.51 | 188.95 | 12.79 | 9.93 | 203.30 |

## GPU load by stage

### ctx_000256.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 49.5 | 99.0 | 51.5 | 51.7 | 6086 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 52.9 | 53.0 | 4708 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 50.8 | 50.8 | 5924 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 54.3 | 54.3 | 5670 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 53.0 | 53.0 | 5738 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 38.5 | 77.0 | 50.0 | 50.0 | 6274 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 12.5 | 18.0 | 73.3 | 167.5 | 6086 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 9.7 | 16.0 | 68.1 | 146.8 | 4708 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 14.7 | 39.0 | 59.5 | 107.4 | 5924 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 16.8 | 65.0 | 55.0 | 64.5 | 5670 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 17.8 | 27.0 | 57.5 | 91.7 | 5738 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 36.0 | 48.0 | 95.3 | 147.2 | 6274 |

### ctx_000512.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 12.0 | 35.0 | 70.4 | 98.5 | 6086 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 21.3 | 32.0 | 112.8 | 167.2 | 4708 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 11.3 | 34.0 | 51.2 | 51.4 | 5924 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 6.3 | 19.0 | 55.1 | 55.7 | 5670 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 17.0 | 32.0 | 68.6 | 99.0 | 5738 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 18.7 | 31.0 | 73.6 | 119.0 | 6274 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 18.0 | 18.0 | 51.8 | 53.6 | 6086 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 5.8 | 16.0 | 59.0 | 79.3 | 4708 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 17.8 | 18.0 | 81.9 | 176.3 | 5924 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 16.0 | 16.0 | 52.7 | 53.9 | 5670 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 6.2 | 18.0 | 52.1 | 57.6 | 5738 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 19.5 | 25.0 | 97.3 | 157.7 | 6274 |

### ctx_001024.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 13.4 | 35.0 | 51.9 | 52.7 | 6086 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 12.8 | 32.0 | 57.4 | 75.7 | 4708 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 13.6 | 34.0 | 74.2 | 147.7 | 5924 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 11.6 | 32.0 | 76.2 | 161.1 | 5670 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 10.8 | 32.0 | 80.5 | 188.5 | 5738 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 26.6 | 89.0 | 64.2 | 98.4 | 6274 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 10.7 | 20.0 | 57.9 | 117.8 | 6086 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 9.3 | 18.0 | 47.8 | 52.6 | 4708 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 14.7 | 20.0 | 60.7 | 120.8 | 5924 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 18.5 | 53.0 | 69.0 | 171.0 | 5670 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 17.8 | 31.0 | 51.7 | 53.6 | 5738 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 27.3 | 43.0 | 81.8 | 154.9 | 6274 |

### ctx_002048.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 19.8 | 69.0 | 98.2 | 173.0 | 6086 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 15.3 | 64.0 | 61.4 | 145.4 | 4708 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 21.6 | 53.0 | 57.3 | 82.1 | 5924 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 23.9 | 64.0 | 88.6 | 174.0 | 5670 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 23.3 | 68.0 | 60.3 | 128.9 | 5738 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 13.8 | 47.0 | 60.4 | 134.1 | 6274 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 10.2 | 18.0 | 56.7 | 81.4 | 6086 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 11.4 | 16.0 | 76.7 | 172.4 | 4708 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 10.4 | 18.0 | 50.6 | 52.2 | 5924 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 12.8 | 16.0 | 54.5 | 64.0 | 5670 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 16.0 | 18.0 | 65.7 | 142.2 | 5738 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 38.0 | 44.0 | 91.4 | 122.9 | 6274 |

### ctx_004096.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 20.4 | 69.0 | 74.9 | 169.7 | 6086 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 19.3 | 64.0 | 70.3 | 166.6 | 4708 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 22.3 | 72.0 | 76.1 | 161.6 | 5924 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 17.6 | 67.0 | 76.3 | 178.9 | 5670 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 18.7 | 58.0 | 78.2 | 165.8 | 5738 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 23.6 | 98.0 | 63.7 | 164.4 | 6274 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 10.0 | 21.0 | 50.1 | 62.7 | 6086 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 8.3 | 17.0 | 81.2 | 139.2 | 4708 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 24.2 | 81.0 | 69.0 | 105.4 | 5924 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 9.7 | 17.0 | 56.2 | 62.5 | 5670 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 11.3 | 19.0 | 60.7 | 95.8 | 5738 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 20.3 | 44.0 | 102.1 | 170.0 | 6274 |

### ctx_008192.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 22.0 | 100.0 | 76.4 | 175.7 | 6086 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 19.0 | 43.0 | 63.4 | 163.0 | 4708 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 17.2 | 76.0 | 67.7 | 183.1 | 5924 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 14.9 | 54.0 | 85.6 | 182.0 | 5670 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 17.8 | 69.0 | 69.3 | 178.5 | 5738 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 14.8 | 98.0 | 63.1 | 167.2 | 6274 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 20.2 | 41.0 | 49.6 | 53.2 | 6086 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 17.8 | 37.0 | 61.1 | 116.4 | 4708 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 9.9 | 20.0 | 80.8 | 181.4 | 5924 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 4.9 | 17.0 | 64.9 | 124.2 | 5670 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 11.9 | 19.0 | 70.3 | 170.9 | 5738 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 38.5 | 89.0 | 66.9 | 156.7 | 6274 |

### ctx_016384.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 21.2 | 100.0 | 66.8 | 177.0 | 6086 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 16.6 | 63.0 | 73.6 | 179.0 | 4708 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 17.6 | 55.0 | 76.1 | 181.7 | 5924 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 17.9 | 73.0 | 74.5 | 184.1 | 5670 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 19.3 | 78.0 | 74.6 | 183.8 | 5738 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 19.1 | 98.0 | 63.2 | 163.7 | 6274 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 20.5 | 60.0 | 52.1 | 53.4 | 6086 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 16.2 | 43.0 | 58.7 | 74.2 | 4708 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 10.3 | 21.0 | 89.8 | 169.2 | 5924 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 4.3 | 19.0 | 53.6 | 54.9 | 5670 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 9.5 | 19.0 | 52.4 | 53.4 | 5738 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 29.3 | 78.0 | 97.2 | 161.7 | 6274 |


## CPU / RAM / swap load by stage

### ctx_000256.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 37.1 %, max 38.0 %

RAM used: avg 3859 MiB, max 3882 MiB, avg 24.3 %, max 24.5 %

Swap used: avg 380 MiB, max 380 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 17.9 | 23.0 |
| 1 | 99.5 | 100.0 |
| 2 | 12.5 | 15.1 |
| 3 | 18.5 | 19.9 |

Decode stage:

CPU total: avg 40.0 %, max 47.9 %

RAM used: avg 3800 MiB, max 3866 MiB, avg 23.9 %, max 24.4 %

Swap used: avg 380 MiB, max 380 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 17.8 | 25.7 |
| 1 | 100.0 | 100.0 |
| 2 | 24.8 | 45.9 |
| 3 | 17.1 | 22.8 |

### ctx_000512.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 36.7 %, max 36.9 %

RAM used: avg 4287 MiB, max 4305 MiB, avg 27.0 %, max 27.1 %

Swap used: avg 380 MiB, max 380 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 99.8 | 100.0 |
| 1 | 17.0 | 18.8 |
| 2 | 16.4 | 20.6 |
| 3 | 13.2 | 16.3 |

Decode stage:

CPU total: avg 41.4 %, max 53.6 %

RAM used: avg 4331 MiB, max 4404 MiB, avg 27.3 %, max 27.7 %

Swap used: avg 380 MiB, max 380 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 57.6 | 99.5 |
| 1 | 22.6 | 37.9 |
| 2 | 34.8 | 83.6 |
| 3 | 50.6 | 95.3 |

### ctx_001024.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 36.9 %, max 37.5 %

RAM used: avg 4748 MiB, max 4782 MiB, avg 29.9 %, max 30.1 %

Swap used: avg 380 MiB, max 380 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 99.8 | 100.0 |
| 1 | 16.3 | 21.8 |
| 2 | 16.0 | 22.2 |
| 3 | 15.6 | 17.8 |

Decode stage:

CPU total: avg 40.0 %, max 49.8 %

RAM used: avg 4849 MiB, max 4899 MiB, avg 30.5 %, max 30.9 %

Swap used: avg 380 MiB, max 380 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 84.2 | 100.0 |
| 1 | 13.8 | 16.5 |
| 2 | 44.3 | 99.5 |
| 3 | 17.8 | 22.2 |

### ctx_002048.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 37.7 %, max 46.6 %

RAM used: avg 5225 MiB, max 5307 MiB, avg 32.9 %, max 33.4 %

Swap used: avg 380 MiB, max 380 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 19.2 | 65.0 |
| 1 | 15.7 | 22.3 |
| 2 | 16.5 | 22.2 |
| 3 | 99.1 | 100.0 |

Decode stage:

CPU total: avg 37.8 %, max 38.3 %

RAM used: avg 5327 MiB, max 5368 MiB, avg 33.6 %, max 33.8 %

Swap used: avg 380 MiB, max 380 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 19.9 | 23.5 |
| 1 | 15.7 | 19.9 |
| 2 | 15.7 | 18.6 |
| 3 | 100.0 | 100.0 |

### ctx_004096.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 38.5 %, max 51.8 %

RAM used: avg 5759 MiB, max 5907 MiB, avg 36.3 %, max 37.2 %

Swap used: avg 380 MiB, max 380 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 85.0 | 100.0 |
| 1 | 15.9 | 23.9 |
| 2 | 16.1 | 25.8 |
| 3 | 36.7 | 100.0 |

Decode stage:

CPU total: avg 37.9 %, max 39.3 %

RAM used: avg 5901 MiB, max 5998 MiB, avg 37.2 %, max 37.8 %

Swap used: avg 380 MiB, max 380 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 59.7 | 100.0 |
| 1 | 57.9 | 100.0 |
| 2 | 15.8 | 23.7 |
| 3 | 18.0 | 23.8 |

### ctx_008192.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 38.4 %, max 52.7 %

RAM used: avg 5454 MiB, max 6727 MiB, avg 34.4 %, max 42.4 %

Swap used: avg 380 MiB, max 380 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 14.6 | 31.5 |
| 1 | 79.5 | 100.0 |
| 2 | 39.9 | 100.0 |
| 3 | 19.2 | 72.2 |

Decode stage:

CPU total: avg 41.3 %, max 46.7 %

RAM used: avg 5530 MiB, max 5670 MiB, avg 34.8 %, max 35.7 %

Swap used: avg 379 MiB, max 379 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 21.3 | 26.1 |
| 1 | 55.2 | 100.0 |
| 2 | 72.6 | 100.0 |
| 3 | 16.0 | 23.9 |

### ctx_016384.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 40.3 %, max 53.3 %

RAM used: avg 6489 MiB, max 6848 MiB, avg 40.9 %, max 43.1 %

Swap used: avg 379 MiB, max 379 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 55.2 | 100.0 |
| 1 | 40.2 | 100.0 |
| 2 | 44.9 | 100.0 |
| 3 | 20.7 | 38.8 |

Decode stage:

CPU total: avg 41.0 %, max 45.8 %

RAM used: avg 6847 MiB, max 6914 MiB, avg 43.1 %, max 43.6 %

Swap used: avg 379 MiB, max 379 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 73.6 | 100.0 |
| 1 | 34.1 | 99.5 |
| 2 | 34.4 | 99.7 |
| 3 | 22.0 | 28.6 |

