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
- TURBOPREFILL: `1`
- TurboPrefill status: `active (TURBOPREFILL=1)`
- TurboPrefill version: `TurboPrefill:`
- llama.cpp git describe: `b10451-1-gba0d2b391-dirty`
- llama.cpp git commit: `ba0d2b3918c4662d8a1fb2eee21c365265f2901f`
- Server PID: `173210`
- KEEP_SERVER_RUNNING: `1`
- Parallel-slots mode: `active_slots=1..PARALLEL`
- Metrics policy: `server per-request timings only; no combined throughput calculated`
- llama_server_log: `/home/serg/workspace/versions/TurboPrefill_b10451/bench_reports_Qwen3.8-27B-Q4_K_M/20260816_145608/llama_server.log`

## Environment

### TURBOPREFILL

```text
1
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
CPU(s) scaling MHz:                      95%
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
Память:         15Gi       6.8Gi       190Mi       279Mi       9.1Gi       8.7Gi
Подкачка:       23Gi       297Mi        23Gi
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
0.01.052.958 I srv    load_model: TurboPrefill: CUDA Graphs disabled for target and draft contexts
6.15.539.678 I decode: TurboPrefill requested=1 active=1 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=127 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
6.15.601.157 I process_ubatch: TurboPrefill recurrent rs_z=0 first_ubatch=standard turbo_ubatches=126
6.29.237.683 I turboprefill: version=TurboPrefill_b10451_v2.1.3 accumulation_us=4661086 compute_us=8340887 total_us=13001973
6.30.405.143 I decode: TurboPrefill requested=1 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
6.32.044.278 I decode: TurboPrefill requested=1 active=1 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=127 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
6.32.053.527 I process_ubatch: TurboPrefill recurrent rs_z=-1 first_ubatch=turbo turbo_ubatches=127
6.41.863.227 I turboprefill: version=TurboPrefill_b10451_v2.1.3 accumulation_us=1072505 compute_us=8734387 total_us=9806892
6.42.943.795 I decode: TurboPrefill requested=1 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
6.44.848.032 I decode: TurboPrefill requested=1 active=1 n_tokens=623 n_ubatch=32 n_rs_seq=3 turbo_ubatches=19 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
6.44.848.718 I process_ubatch: TurboPrefill recurrent rs_z=-1 first_ubatch=turbo turbo_ubatches=19
6.46.671.675 I turboprefill: version=TurboPrefill_b10451_v2.1.3 accumulation_us=189526 compute_us=1630486 total_us=1820012
6.46.915.276 I decode: TurboPrefill requested=1 active=0 n_tokens=623 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
7.03.322.602 I decode: TurboPrefill requested=1 active=1 n_tokens=232 n_ubatch=32 n_rs_seq=3 turbo_ubatches=7 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
7.03.323.460 I process_ubatch: TurboPrefill recurrent rs_z=0 first_ubatch=standard turbo_ubatches=6
7.04.294.508 I turboprefill: version=TurboPrefill_b10451_v2.1.3 accumulation_us=324893 compute_us=630831 total_us=955724
7.04.603.901 I decode: TurboPrefill requested=1 active=0 n_tokens=232 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
7.17.704.670 I decode: TurboPrefill requested=1 active=1 n_tokens=505 n_ubatch=32 n_rs_seq=3 turbo_ubatches=15 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
7.17.705.382 I process_ubatch: TurboPrefill recurrent rs_z=0 first_ubatch=standard turbo_ubatches=14
7.19.188.419 I turboprefill: version=TurboPrefill_b10451_v2.1.3 accumulation_us=361990 compute_us=1105970 total_us=1467960
```

## Server command

```bash
/home/serg/workspace/versions/TurboPrefill_b10451/build/bin/llama-server -m /mnt/models/AI/LLM/Qwen3.8-27B-Q4_K_M.gguf --host 0.0.0.0 --port 8081 -lv 4 -ngl 999 -c 260000 --override-kv llama.context_length=int:260000 -b 4097 -ub 32 -np 1 -ctk f16 -ctv f16 -sm layer -ts 11/11/11/11/11/9 --flash-attn auto --no-warmup --no-mmproj --spec-type draft-mtp --spec-draft-n-max 3
```

Server PID: `173210`  
Stop command: `kill -INT 173210`

## Summary

| File | Active slots | Request | Prompt tokens | Completion tokens | Prefill tok/s | Prefill time s | Decode tok/s | Decode time s | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ctx_000256.txt | 1 | 1 | 268 | 128 | 75.44 | 3.55 | 13.14 | 9.67 | 17.56 |
| ctx_000512.txt | 1 | 1 | 541 | 128 | 129.97 | 4.16 | 13.31 | 9.54 | 14.85 |
| ctx_001024.txt | 1 | 1 | 1082 | 128 | 179.69 | 6.02 | 10.71 | 11.86 | 19.82 |
| ctx_002048.txt | 1 | 1 | 2330 | 128 | 254.23 | 9.16 | 13.83 | 9.18 | 20.35 |
| ctx_004096.txt | 1 | 1 | 4288 | 128 | 286.60 | 14.96 | 13.27 | 9.57 | 26.74 |
| ctx_008192.txt | 1 | 1 | 8853 | 128 | 317.18 | 27.91 | 14.18 | 8.96 | 43.78 |
| ctx_016384.txt | 1 | 1 | 17670 | 128 | 298.75 | 59.15 | 12.73 | 9.98 | 73.58 |

## GPU load by stage

### ctx_000256.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 82.0 | 82.0 | 51.4 | 51.4 | 6088 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 52.4 | 52.4 | 4710 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 51.1 | 51.1 | 5926 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 54.2 | 54.2 | 5672 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 52.9 | 52.9 | 5740 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 4.0 | 4.0 | 49.7 | 49.7 | 6276 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 12.9 | 18.0 | 83.6 | 174.8 | 6088 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 11.4 | 16.0 | 51.6 | 52.2 | 4710 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 9.0 | 18.0 | 50.5 | 51.6 | 5926 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 8.6 | 16.0 | 54.3 | 61.3 | 5672 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 18.0 | 46.0 | 63.9 | 118.1 | 5740 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 30.3 | 99.0 | 92.9 | 139.1 | 6276 |

### ctx_000512.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 98.0 | 98.0 | 120.9 | 120.9 | 6088 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 89.0 | 89.0 | 135.8 | 135.8 | 4710 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 97.0 | 97.0 | 157.1 | 157.1 | 5926 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 91.0 | 91.0 | 167.6 | 167.6 | 5672 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 90.0 | 90.0 | 178.2 | 178.2 | 5740 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 62.0 | 62.0 | 161.3 | 161.3 | 6276 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 13.0 | 20.0 | 50.6 | 66.0 | 6088 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 13.2 | 18.0 | 61.4 | 107.6 | 4710 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 11.8 | 20.0 | 48.4 | 51.2 | 5926 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 12.0 | 18.0 | 60.8 | 107.3 | 5672 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 14.0 | 21.0 | 66.1 | 122.6 | 5740 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 38.8 | 43.0 | 84.7 | 138.1 | 6276 |

### ctx_001024.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 67.0 | 97.0 | 112.1 | 178.2 | 6088 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 63.0 | 92.0 | 89.4 | 131.7 | 4710 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 47.5 | 95.0 | 86.2 | 126.9 | 5926 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 56.0 | 88.0 | 89.7 | 119.5 | 5672 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 45.0 | 89.0 | 152.6 | 165.7 | 5740 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 29.5 | 59.0 | 93.6 | 136.5 | 6276 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 16.0 | 35.0 | 74.2 | 170.2 | 6088 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 11.8 | 32.0 | 77.2 | 183.0 | 4710 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 12.0 | 18.0 | 74.7 | 176.2 | 5926 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 16.8 | 21.0 | 74.3 | 180.8 | 5672 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 13.0 | 18.0 | 55.4 | 98.1 | 5740 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 32.3 | 43.0 | 91.1 | 169.6 | 6276 |

### ctx_002048.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 57.8 | 98.0 | 102.1 | 170.2 | 6088 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 54.2 | 93.0 | 103.7 | 174.7 | 4710 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 56.8 | 97.0 | 109.3 | 176.5 | 5926 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 51.2 | 89.0 | 112.5 | 181.5 | 5672 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 47.5 | 92.0 | 122.6 | 182.7 | 5740 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 57.2 | 98.0 | 119.5 | 161.3 | 6276 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 11.8 | 18.0 | 74.7 | 176.0 | 6088 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 11.4 | 16.0 | 55.1 | 65.2 | 4710 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 14.4 | 18.0 | 72.9 | 144.0 | 5926 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 7.6 | 16.0 | 96.2 | 177.6 | 5672 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 9.2 | 17.0 | 55.6 | 62.6 | 5740 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 36.8 | 44.0 | 61.0 | 86.7 | 6276 |

### ctx_004096.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 65.3 | 98.0 | 130.1 | 177.4 | 6088 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 59.7 | 92.0 | 133.1 | 177.7 | 4710 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 64.3 | 97.0 | 125.4 | 179.4 | 5926 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 59.5 | 90.0 | 113.3 | 177.9 | 5672 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 60.2 | 92.0 | 120.1 | 181.1 | 5740 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 69.2 | 98.0 | 109.4 | 155.2 | 6276 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 13.0 | 18.0 | 78.1 | 177.4 | 6088 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 11.2 | 17.0 | 79.9 | 170.7 | 4710 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 13.8 | 21.0 | 67.4 | 149.7 | 5926 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 18.3 | 55.0 | 55.5 | 65.2 | 5672 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 12.0 | 19.0 | 63.4 | 131.0 | 5740 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 24.8 | 45.0 | 75.0 | 124.9 | 6276 |

### ctx_008192.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 69.4 | 98.0 | 117.0 | 174.2 | 6088 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 58.5 | 90.0 | 120.8 | 181.5 | 4710 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 65.8 | 97.0 | 132.4 | 185.6 | 5926 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 64.8 | 92.0 | 126.2 | 184.8 | 5672 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 63.5 | 93.0 | 124.2 | 191.6 | 5740 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 58.6 | 100.0 | 82.1 | 167.8 | 6276 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 29.0 | 98.0 | 59.1 | 107.0 | 6088 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 22.0 | 88.0 | 62.9 | 118.0 | 4710 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 29.9 | 97.0 | 81.9 | 152.0 | 5926 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 20.9 | 92.0 | 101.1 | 182.3 | 5672 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 18.3 | 89.0 | 69.1 | 176.3 | 5740 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 42.7 | 98.0 | 93.6 | 157.9 | 6276 |

### ctx_016384.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 61.8 | 99.0 | 122.8 | 174.5 | 6088 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 55.2 | 92.0 | 116.7 | 183.2 | 4710 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 62.7 | 98.0 | 122.8 | 186.5 | 5926 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 57.5 | 93.0 | 123.4 | 190.7 | 5672 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 58.7 | 93.0 | 122.8 | 188.2 | 5740 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 66.1 | 99.0 | 103.7 | 168.0 | 6276 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 21.7 | 47.0 | 74.0 | 172.6 | 6088 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 7.3 | 18.0 | 81.7 | 163.3 | 4710 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 13.7 | 38.0 | 60.8 | 95.6 | 5926 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 12.9 | 40.0 | 69.5 | 150.3 | 5672 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 20.0 | 41.0 | 64.9 | 132.6 | 5740 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 51.4 | 81.0 | 84.8 | 163.9 | 6276 |


## CPU / RAM / swap load by stage

### ctx_000256.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 39.2 %, max 39.2 %

RAM used: avg 5110 MiB, max 5110 MiB, avg 32.2 %, max 32.2 %

Swap used: avg 371 MiB, max 371 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 11.3 | 11.3 |
| 1 | 16.4 | 16.4 |
| 2 | 30.8 | 30.8 |
| 3 | 98.6 | 98.6 |

Decode stage:

CPU total: avg 41.3 %, max 51.3 %

RAM used: avg 5077 MiB, max 5168 MiB, avg 32.0 %, max 32.6 %

Swap used: avg 371 MiB, max 371 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 19.0 | 25.8 |
| 1 | 62.3 | 100.0 |
| 2 | 20.3 | 28.9 |
| 3 | 63.4 | 100.0 |

### ctx_000512.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 40.0 %, max 40.0 %

RAM used: avg 5389 MiB, max 5389 MiB, avg 33.9 %, max 33.9 %

Swap used: avg 371 MiB, max 371 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 19.3 | 19.3 |
| 1 | 18.3 | 18.3 |
| 2 | 23.0 | 23.0 |
| 3 | 99.5 | 99.5 |

Decode stage:

CPU total: avg 40.0 %, max 41.4 %

RAM used: avg 5447 MiB, max 5482 MiB, avg 34.3 %, max 34.5 %

Swap used: avg 371 MiB, max 371 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 18.9 | 23.8 |
| 1 | 20.5 | 27.6 |
| 2 | 20.7 | 25.2 |
| 3 | 99.8 | 100.0 |

### ctx_001024.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 39.5 %, max 39.6 %

RAM used: avg 5394 MiB, max 5466 MiB, avg 34.0 %, max 34.4 %

Swap used: avg 672 MiB, max 681 MiB, avg 2.7 %, max 2.8 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 73.4 | 83.4 |
| 1 | 26.9 | 29.5 |
| 2 | 24.2 | 29.9 |
| 3 | 33.9 | 36.7 |

Decode stage:

CPU total: avg 41.1 %, max 49.9 %

RAM used: avg 5713 MiB, max 5825 MiB, avg 36.0 %, max 36.7 %

Swap used: avg 643 MiB, max 648 MiB, avg 2.6 %, max 2.6 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 86.1 | 100.0 |
| 1 | 25.1 | 56.1 |
| 2 | 30.4 | 65.7 |
| 3 | 22.7 | 28.8 |

### ctx_002048.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 38.5 %, max 42.5 %

RAM used: avg 6222 MiB, max 6251 MiB, avg 39.2 %, max 39.4 %

Swap used: avg 618 MiB, max 635 MiB, avg 2.5 %, max 2.6 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 89.7 | 100.0 |
| 1 | 16.3 | 21.7 |
| 2 | 22.4 | 30.6 |
| 3 | 25.4 | 30.4 |

Decode stage:

CPU total: avg 42.1 %, max 47.4 %

RAM used: avg 6401 MiB, max 6485 MiB, avg 40.3 %, max 40.9 %

Swap used: avg 595 MiB, max 599 MiB, avg 2.4 %, max 2.4 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 27.7 | 55.4 |
| 1 | 50.7 | 97.0 |
| 2 | 34.0 | 61.7 |
| 3 | 55.8 | 99.5 |

### ctx_004096.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 40.6 %, max 44.8 %

RAM used: avg 6847 MiB, max 6974 MiB, avg 43.1 %, max 43.9 %

Swap used: avg 536 MiB, max 581 MiB, avg 2.2 %, max 2.4 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 22.8 | 25.9 |
| 1 | 86.8 | 100.0 |
| 2 | 23.0 | 30.0 |
| 3 | 29.5 | 56.4 |

Decode stage:

CPU total: avg 41.8 %, max 50.7 %

RAM used: avg 7107 MiB, max 7222 MiB, avg 44.8 %, max 45.5 %

Swap used: avg 499 MiB, max 510 MiB, avg 2.0 %, max 2.1 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 41.6 | 95.2 |
| 1 | 62.7 | 100.0 |
| 2 | 19.0 | 25.5 |
| 3 | 43.7 | 76.5 |

### ctx_008192.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 42.3 %, max 48.7 %

RAM used: avg 6837 MiB, max 7905 MiB, avg 43.1 %, max 49.8 %

Swap used: avg 450 MiB, max 474 MiB, avg 1.8 %, max 1.9 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 78.1 | 100.0 |
| 1 | 20.1 | 28.6 |
| 2 | 23.4 | 32.5 |
| 3 | 47.6 | 100.0 |

Decode stage:

CPU total: avg 42.3 %, max 52.8 %

RAM used: avg 6809 MiB, max 7021 MiB, avg 42.9 %, max 44.2 %

Swap used: avg 430 MiB, max 430 MiB, avg 1.8 %, max 1.8 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 82.7 | 100.0 |
| 1 | 19.3 | 32.1 |
| 2 | 24.2 | 32.3 |
| 3 | 42.6 | 100.0 |

### ctx_016384.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 43.3 %, max 53.7 %

RAM used: avg 7848 MiB, max 8182 MiB, avg 49.4 %, max 51.5 %

Swap used: avg 430 MiB, max 430 MiB, avg 1.7 %, max 1.8 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 29.8 | 96.2 |
| 1 | 23.8 | 40.2 |
| 2 | 24.4 | 33.9 |
| 3 | 94.9 | 100.0 |

Decode stage:

CPU total: avg 42.3 %, max 53.2 %

RAM used: avg 8300 MiB, max 8408 MiB, avg 52.3 %, max 53.0 %

Swap used: avg 421 MiB, max 421 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 24.3 | 71.6 |
| 1 | 22.2 | 29.5 |
| 2 | 22.7 | 30.9 |
| 3 | 99.9 | 100.0 |

