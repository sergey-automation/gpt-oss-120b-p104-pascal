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
- TURBOPREFILL: `1`
- TurboPrefill status: `active (TURBOPREFILL=1)`
- TurboPrefill version: `TurboPrefill:`
- llama.cpp git describe: `b10451-1-gba0d2b391-dirty`
- llama.cpp git commit: `ba0d2b3918c4662d8a1fb2eee21c365265f2901f`
- Server PID: `225575`
- KEEP_SERVER_RUNNING: `1`
- Parallel-slots mode: `active_slots=1..PARALLEL`
- Metrics policy: `server per-request timings only; no combined throughput calculated`
- llama_server_log: `/home/serg/workspace/versions/TurboPrefill_b10451/bench_reports_Qwen3.8-27B-Q4_K_M/20260816_163642/llama_server.log`

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
Память:         15Gi       6.7Gi       251Mi       245Mi       9.1Gi       8.9Gi
Подкачка:       23Gi       289Mi        23Gi
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
0.01.001.977 I srv    load_model: TurboPrefill: CUDA Graphs disabled for target and draft contexts
5.39.350.462 I decode: TurboPrefill requested=1 active=1 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=127 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
5.39.428.028 I process_ubatch: TurboPrefill recurrent rs_z=0 first_ubatch=standard turbo_ubatches=126
5.55.083.649 I turboprefill: version=TurboPrefill_b10451_v2.1.3 accumulation_us=3362057 compute_us=11555794 total_us=14917851
5.56.180.235 I decode: TurboPrefill requested=1 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
5.57.825.005 I decode: TurboPrefill requested=1 active=1 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=127 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
5.57.834.276 I process_ubatch: TurboPrefill recurrent rs_z=-1 first_ubatch=turbo turbo_ubatches=127
6.11.984.430 I turboprefill: version=TurboPrefill_b10451_v2.1.3 accumulation_us=1899617 compute_us=12247774 total_us=14147391
6.12.909.057 I decode: TurboPrefill requested=1 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
6.14.823.478 I decode: TurboPrefill requested=1 active=1 n_tokens=623 n_ubatch=32 n_rs_seq=3 turbo_ubatches=19 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
6.14.824.176 I process_ubatch: TurboPrefill recurrent rs_z=-1 first_ubatch=turbo turbo_ubatches=19
6.17.162.027 I turboprefill: version=TurboPrefill_b10451_v2.1.3 accumulation_us=211955 compute_us=2123114 total_us=2335069
6.17.392.430 I decode: TurboPrefill requested=1 active=0 n_tokens=623 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
6.33.852.998 I decode: TurboPrefill requested=1 active=1 n_tokens=232 n_ubatch=32 n_rs_seq=3 turbo_ubatches=7 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
6.33.853.842 I process_ubatch: TurboPrefill recurrent rs_z=0 first_ubatch=standard turbo_ubatches=6
6.34.958.018 I turboprefill: version=TurboPrefill_b10451_v2.1.3 accumulation_us=343197 compute_us=743967 total_us=1087164
6.35.256.390 I decode: TurboPrefill requested=1 active=0 n_tokens=232 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
6.48.738.396 I decode: TurboPrefill requested=1 active=1 n_tokens=505 n_ubatch=32 n_rs_seq=3 turbo_ubatches=15 devices=6 pipeline=1 split_mode=1 embeddings_nextn=1
6.48.742.725 I process_ubatch: TurboPrefill recurrent rs_z=0 first_ubatch=standard turbo_ubatches=14
6.50.697.523 I turboprefill: version=TurboPrefill_b10451_v2.1.3 accumulation_us=399181 compute_us=1454366 total_us=1853547
```

## Server command

```bash
/home/serg/workspace/versions/TurboPrefill_b10451/build/bin/llama-server -m /mnt/models/AI/LLM/Qwen3.8-27B-Q4_K_M.gguf --host 0.0.0.0 --port 8081 -lv 4 -ngl 999 -c 200000 --override-kv llama.context_length=int:200000 -b 4097 -ub 32 -np 1 -ctk f16 -ctv f16 -sm layer -ts 18/18/17/15 --flash-attn auto --no-warmup --no-mmproj --spec-type draft-mtp --spec-draft-n-max 3
```

Server PID: `225575`  
Stop command: `kill -INT 225575`

## Summary

| File | Active slots | Request | Prompt tokens | Completion tokens | Prefill tok/s | Prefill time s | Decode tok/s | Decode time s | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ctx_000256.txt | 1 | 1 | 268 | 128 | 73.04 | 3.67 | 13.00 | 9.77 | 17.78 |
| ctx_000512.txt | 1 | 1 | 541 | 128 | 112.76 | 4.80 | 13.76 | 9.23 | 15.35 |
| ctx_001024.txt | 1 | 1 | 1082 | 128 | 167.51 | 6.46 | 10.36 | 12.26 | 20.45 |
| ctx_002048.txt | 1 | 1 | 2330 | 128 | 206.08 | 11.31 | 13.72 | 9.26 | 22.26 |
| ctx_004096.txt | 1 | 1 | 4288 | 128 | 236.21 | 18.15 | 13.00 | 9.77 | 30.02 |
| ctx_008192.txt | 1 | 1 | 8853 | 128 | 246.46 | 35.92 | 13.88 | 9.15 | 51.85 |
| ctx_016384.txt | 1 | 1 | 17670 | 128 | 237.77 | 74.32 | 12.55 | 10.12 | 88.86 |

## GPU load by stage

### ctx_000256.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 78.0 | 78.0 | 51.9 | 51.9 | 7514 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 53.3 | 53.3 | 7122 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 51.6 | 51.6 | 7862 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 53.0 | 53.0 | 54.6 | 54.6 | 7492 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.8 | 8.8 | 86 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 9.8 | 86 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 9.6 | 24.0 | 70.8 | 142.9 | 7514 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 11.4 | 25.0 | 58.3 | 95.4 | 7122 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 26.0 | 81.0 | 65.5 | 158.9 | 7862 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 32.1 | 99.0 | 91.5 | 161.6 | 7492 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 9.1 | 86 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.7 | 9.8 | 86 |

### ctx_000512.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 54.5 | 98.0 | 99.3 | 146.9 | 7514 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 52.5 | 92.0 | 100.2 | 146.6 | 7122 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 23.5 | 47.0 | 102.7 | 153.4 | 7862 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 110.7 | 166.5 | 7492 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.8 | 8.8 | 86 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 10.0 | 10.2 | 86 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 27.0 | 27.0 | 139.0 | 175.2 | 7514 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 25.0 | 25.0 | 89.6 | 178.7 | 7122 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 25.0 | 25.0 | 48.7 | 50.1 | 7862 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 33.2 | 50.0 | 56.1 | 61.0 | 7492 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.8 | 8.9 | 86 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 9.9 | 86 |

### ctx_001024.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 49.0 | 98.0 | 115.0 | 178.0 | 7514 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 46.0 | 92.0 | 120.3 | 187.1 | 7122 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 45.5 | 91.0 | 104.0 | 156.1 | 7862 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 55.0 | 71.0 | 62.2 | 70.5 | 7492 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 8.9 | 86 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 9.9 | 86 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 18.0 | 27.0 | 76.0 | 168.2 | 7514 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 20.3 | 28.0 | 66.8 | 131.5 | 7122 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 31.9 | 85.0 | 62.1 | 123.8 | 7862 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 32.7 | 52.0 | 87.4 | 176.5 | 7492 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.8 | 8.9 | 86 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 9.9 | 86 |

### ctx_002048.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 71.2 | 98.0 | 119.1 | 181.8 | 7514 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 55.0 | 92.0 | 108.0 | 181.8 | 7122 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 55.4 | 91.0 | 105.5 | 184.1 | 7862 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 60.0 | 68.0 | 74.5 | 121.1 | 7492 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 9.2 | 86 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 9.9 | 86 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 23.0 | 30.0 | 64.1 | 109.2 | 7514 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 20.8 | 25.0 | 102.7 | 181.0 | 7122 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 21.2 | 25.0 | 69.0 | 147.0 | 7862 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 34.0 | 52.0 | 86.7 | 147.8 | 7492 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 9.0 | 86 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 10.0 | 86 |

### ctx_004096.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 70.6 | 99.0 | 126.7 | 174.9 | 7514 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 66.0 | 92.0 | 143.1 | 184.5 | 7122 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 74.5 | 95.0 | 140.9 | 185.2 | 7862 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 56.5 | 74.0 | 115.3 | 187.5 | 7492 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 8.9 | 86 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.7 | 9.9 | 86 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 33.4 | 79.0 | 58.7 | 90.9 | 7514 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 32.4 | 72.0 | 61.8 | 116.0 | 7122 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 18.2 | 29.0 | 46.4 | 52.4 | 7862 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 32.0 | 53.0 | 97.2 | 156.7 | 7492 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 8.9 | 86 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 9.9 | 86 |

### ctx_008192.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 78.9 | 99.0 | 125.2 | 177.2 | 7514 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 66.4 | 98.0 | 130.8 | 186.2 | 7122 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 57.7 | 94.0 | 143.6 | 182.9 | 7862 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 48.7 | 85.0 | 105.4 | 175.9 | 7492 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 9.4 | 86 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 10.0 | 86 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 30.1 | 98.0 | 79.1 | 161.2 | 7514 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 28.9 | 93.0 | 62.6 | 145.7 | 7122 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 8.6 | 21.0 | 62.3 | 149.3 | 7862 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 34.6 | 83.0 | 113.8 | 176.1 | 7492 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.0 | 10.1 | 86 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.7 | 9.9 | 86 |

### ctx_016384.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 71.1 | 99.0 | 130.9 | 182.1 | 7514 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 68.4 | 96.0 | 130.6 | 189.1 | 7122 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 71.3 | 99.0 | 130.1 | 189.1 | 7862 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 71.5 | 99.0 | 118.5 | 189.7 | 7492 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 10.3 | 86 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 10.3 | 86 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 16.9 | 46.0 | 73.8 | 175.4 | 7514 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 18.0 | 58.0 | 71.4 | 163.8 | 7122 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 20.4 | 41.0 | 87.4 | 179.3 | 7862 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 39.9 | 89.0 | 69.0 | 101.5 | 7492 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 8.9 | 8.9 | 86 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 9.8 | 9.9 | 86 |


## CPU / RAM / swap load by stage

### ctx_000256.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 53.0 %, max 53.0 %

RAM used: avg 4638 MiB, max 4638 MiB, avg 29.2 %, max 29.2 %

Swap used: avg 368 MiB, max 368 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 99.1 | 99.1 |
| 1 | 10.4 | 10.4 |
| 2 | 73.1 | 73.1 |
| 3 | 29.0 | 29.0 |

Decode stage:

CPU total: avg 40.5 %, max 41.7 %

RAM used: avg 4633 MiB, max 4733 MiB, avg 29.2 %, max 29.8 %

Swap used: avg 368 MiB, max 368 MiB, avg 1.5 %, max 1.5 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 99.6 | 100.0 |
| 1 | 17.4 | 20.6 |
| 2 | 22.1 | 29.3 |
| 3 | 22.7 | 30.7 |

### ctx_000512.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 39.9 %, max 40.6 %

RAM used: avg 4622 MiB, max 4739 MiB, avg 29.1 %, max 29.9 %

Swap used: avg 579 MiB, max 585 MiB, avg 2.4 %, max 2.4 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 19.5 | 26.1 |
| 1 | 50.0 | 70.0 |
| 2 | 18.6 | 21.0 |
| 3 | 71.5 | 98.6 |

Decode stage:

CPU total: avg 42.3 %, max 46.3 %

RAM used: avg 4863 MiB, max 4903 MiB, avg 30.6 %, max 30.9 %

Swap used: avg 567 MiB, max 567 MiB, avg 2.3 %, max 2.3 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 26.3 | 47.4 |
| 1 | 16.6 | 19.0 |
| 2 | 26.6 | 35.0 |
| 3 | 99.5 | 100.0 |

### ctx_001024.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 38.0 %, max 42.0 %

RAM used: avg 5247 MiB, max 5260 MiB, avg 33.0 %, max 33.1 %

Swap used: avg 554 MiB, max 559 MiB, avg 2.3 %, max 2.3 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 57.4 | 96.3 |
| 1 | 53.3 | 79.5 |
| 2 | 17.0 | 19.3 |
| 3 | 24.4 | 25.4 |

Decode stage:

CPU total: avg 41.7 %, max 44.9 %

RAM used: avg 5430 MiB, max 5504 MiB, avg 34.2 %, max 34.7 %

Swap used: avg 535 MiB, max 540 MiB, avg 2.2 %, max 2.2 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 31.2 | 67.4 |
| 1 | 27.0 | 36.4 |
| 2 | 57.5 | 99.5 |
| 3 | 50.7 | 100.0 |

### ctx_002048.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 39.3 %, max 43.7 %

RAM used: avg 5773 MiB, max 5863 MiB, avg 36.4 %, max 36.9 %

Swap used: avg 500 MiB, max 523 MiB, avg 2.0 %, max 2.1 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 67.5 | 100.0 |
| 1 | 24.9 | 26.6 |
| 2 | 30.4 | 77.2 |
| 3 | 33.9 | 63.9 |

Decode stage:

CPU total: avg 42.9 %, max 50.4 %

RAM used: avg 6011 MiB, max 6042 MiB, avg 37.9 %, max 38.1 %

Swap used: avg 479 MiB, max 479 MiB, avg 2.0 %, max 2.0 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 32.0 | 71.5 |
| 1 | 19.7 | 24.1 |
| 2 | 20.0 | 28.0 |
| 3 | 99.7 | 100.0 |

### ctx_004096.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 41.4 %, max 44.5 %

RAM used: avg 6498 MiB, max 6627 MiB, avg 40.9 %, max 41.7 %

Swap used: avg 443 MiB, max 478 MiB, avg 1.8 %, max 1.9 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 20.7 | 23.5 |
| 1 | 84.9 | 100.0 |
| 2 | 20.0 | 25.5 |
| 3 | 39.7 | 94.4 |

Decode stage:

CPU total: avg 44.5 %, max 49.8 %

RAM used: avg 6749 MiB, max 6782 MiB, avg 42.5 %, max 42.7 %

Swap used: avg 424 MiB, max 424 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 28.4 | 47.8 |
| 1 | 99.9 | 100.0 |
| 2 | 24.4 | 33.3 |
| 3 | 25.0 | 30.2 |

### ctx_008192.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 42.2 %, max 51.6 %

RAM used: avg 6378 MiB, max 7491 MiB, avg 40.2 %, max 47.2 %

Swap used: avg 423 MiB, max 424 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 53.8 | 100.0 |
| 1 | 71.1 | 100.0 |
| 2 | 20.3 | 34.4 |
| 3 | 23.4 | 31.2 |

Decode stage:

CPU total: avg 42.6 %, max 50.3 %

RAM used: avg 6395 MiB, max 6536 MiB, avg 40.3 %, max 41.2 %

Swap used: avg 422 MiB, max 422 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 27.5 | 60.1 |
| 1 | 99.9 | 100.0 |
| 2 | 20.4 | 23.6 |
| 3 | 22.4 | 31.0 |

### ctx_016384.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 43.4 %, max 50.8 %

RAM used: avg 7487 MiB, max 7730 MiB, avg 47.2 %, max 48.7 %

Swap used: avg 422 MiB, max 422 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 26.6 | 73.6 |
| 1 | 75.0 | 100.0 |
| 2 | 47.9 | 100.0 |
| 3 | 23.7 | 48.1 |

Decode stage:

CPU total: avg 43.0 %, max 52.5 %

RAM used: avg 7834 MiB, max 7967 MiB, avg 49.3 %, max 50.2 %

Swap used: avg 422 MiB, max 422 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 21.0 | 28.8 |
| 1 | 75.3 | 100.0 |
| 2 | 52.2 | 100.0 |
| 3 | 23.3 | 33.0 |

