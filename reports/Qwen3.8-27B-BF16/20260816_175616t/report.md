# llama-server parallel-slots context benchmark report

## Test header

- MODEL: `/mnt/models/AI/LLM/Qwen3.8-27B-BF16.gguf`
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
- TENSOR_SPLIT: `10/10/10/10/10/10/10/10/10/10/10/10`
- PARALLEL: `1`
- TEMPERATURE: `0.15`
- CACHE_PROMPT: `0`
- FLASH_ATTN: `auto`
- THREADS: `auto`
- THREADS_BATCH: `auto`
- REPEATS: `1`
- CUDA_VISIBLE_DEVICES: `0,1,2,3,4,5,6,7,8,9,10,11`
- TURBOPREFILL: `1`
- TurboPrefill status: `active (TURBOPREFILL=1)`
- TurboPrefill version: `TurboPrefill:`
- llama.cpp git describe: `b10451-1-gba0d2b391-dirty`
- llama.cpp git commit: `ba0d2b3918c4662d8a1fb2eee21c365265f2901f`
- Server PID: `272746`
- KEEP_SERVER_RUNNING: `1`
- Parallel-slots mode: `active_slots=1..PARALLEL`
- Metrics policy: `server per-request timings only; no combined throughput calculated`
- llama_server_log: `/home/serg/workspace/versions/TurboPrefill_b10451/bench_reports_Qwen3.8-27B-BF16/20260816_175616/llama_server.log`

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
0,1,2,3,4,5,6,7,8,9,10,11
```

### selected_gpu_count

```text
12
```

### selected_gpu_models

```text
NVIDIA P104-100 x12
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
CPU(s) scaling MHz:                      75%
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
Память:         15Gi       7.7Gi       211Mi       315Mi       8.2Gi       7.8Gi
Подкачка:       23Gi       328Mi        23Gi
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
/mnt/models/AI/LLM/Qwen3.8-27B-BF16.gguf
```

### model_filename

```text
Qwen3.8-27B-BF16.gguf
```

### model_size_bytes

```text
54657733952
```

### model_size_gib

```text
50.904
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
BF16:506, F32:360
```

### model_sha256

```text
disabled (MODEL_HASH=0)
```

### TurboPrefill runtime markers

```text
0.02.023.257 I srv    load_model: TurboPrefill: CUDA Graphs disabled for target and draft contexts
18.54.876.318 I decode: TurboPrefill requested=1 active=1 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=127 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
18.54.961.675 I process_ubatch: TurboPrefill recurrent rs_z=0 first_ubatch=standard turbo_ubatches=126
19.25.367.128 I turboprefill: version=TurboPrefill_b10451_v2.1.3 accumulation_us=10406998 compute_us=17567345 total_us=27974343
19.28.695.903 I decode: TurboPrefill requested=1 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
19.32.423.991 I decode: TurboPrefill requested=1 active=1 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=127 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
19.32.433.158 I process_ubatch: TurboPrefill recurrent rs_z=-1 first_ubatch=turbo turbo_ubatches=127
19.51.482.769 I turboprefill: version=TurboPrefill_b10451_v2.1.3 accumulation_us=1813993 compute_us=17232492 total_us=19046485
19.54.538.387 I decode: TurboPrefill requested=1 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
19.58.496.101 I decode: TurboPrefill requested=1 active=1 n_tokens=623 n_ubatch=32 n_rs_seq=3 turbo_ubatches=19 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
19.58.496.839 I process_ubatch: TurboPrefill recurrent rs_z=-1 first_ubatch=turbo turbo_ubatches=19
20.02.484.077 I turboprefill: version=TurboPrefill_b10451_v2.1.3 accumulation_us=207625 compute_us=3776454 total_us=3984079
20.03.726.432 I decode: TurboPrefill requested=1 active=0 n_tokens=623 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
20.31.203.963 I decode: TurboPrefill requested=1 active=1 n_tokens=232 n_ubatch=32 n_rs_seq=3 turbo_ubatches=7 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
20.31.204.805 I process_ubatch: TurboPrefill recurrent rs_z=0 first_ubatch=standard turbo_ubatches=6
20.34.513.951 I turboprefill: version=TurboPrefill_b10451_v2.1.3 accumulation_us=1289145 compute_us=1992805 total_us=3281950
20.35.354.030 I decode: TurboPrefill requested=1 active=0 n_tokens=232 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
20.59.764.522 I decode: TurboPrefill requested=1 active=1 n_tokens=505 n_ubatch=32 n_rs_seq=3 turbo_ubatches=15 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
20.59.765.001 I process_ubatch: TurboPrefill recurrent rs_z=0 first_ubatch=standard turbo_ubatches=14
21.04.336.972 I turboprefill: version=TurboPrefill_b10451_v2.1.3 accumulation_us=1570627 compute_us=2983251 total_us=4553878
```

## Server command

```bash
/home/serg/workspace/versions/TurboPrefill_b10451/build/bin/llama-server -m /mnt/models/AI/LLM/Qwen3.8-27B-BF16.gguf --host 0.0.0.0 --port 8081 -lv 4 -ngl 999 -c 260000 --override-kv llama.context_length=int:260000 -b 4097 -ub 32 -np 1 -ctk f16 -ctv f16 -sm layer -ts 10/10/10/10/10/10/10/10/10/10/10/10 --flash-attn auto --no-warmup --no-mmproj --spec-type draft-mtp --spec-draft-n-max 3
```

Server PID: `272746`  
Stop command: `kill -INT 272746`

## Summary

| File | Active slots | Request | Prompt tokens | Completion tokens | Prefill tok/s | Prefill time s | Decode tok/s | Decode time s | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ctx_000256.txt | 1 | 1 | 268 | 128 | 34.85 | 7.69 | 6.45 | 19.68 | 31.69 |
| ctx_000512.txt | 1 | 1 | 541 | 128 | 56.30 | 9.61 | 6.73 | 18.87 | 29.63 |
| ctx_001024.txt | 1 | 1 | 1082 | 128 | 89.08 | 12.15 | 5.64 | 22.53 | 35.95 |
| ctx_002048.txt | 1 | 1 | 2330 | 128 | 119.17 | 19.55 | 5.84 | 21.73 | 42.78 |
| ctx_004096.txt | 1 | 1 | 4288 | 128 | 139.09 | 30.83 | 6.13 | 20.72 | 53.47 |
| ctx_008192.txt | 1 | 1 | 8853 | 128 | 155.79 | 56.82 | 6.61 | 19.22 | 82.69 |
| ctx_016384.txt | 1 | 1 | 17670 | 128 | 154.31 | 114.51 | 5.41 | 23.45 | 142.41 |

## GPU load by stage

### ctx_000256.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 46.7 | 100.0 | 76.0 | 123.8 | 5638 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 43.0 | 77.0 | 85.6 | 151.1 | 4894 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 33.0 | 99.0 | 82.8 | 144.3 | 6620 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 29.3 | 88.0 | 74.4 | 115.4 | 4894 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 32.3 | 97.0 | 87.0 | 153.6 | 6620 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 29.0 | 87.0 | 60.1 | 87.7 | 4894 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 0.7 | 2.0 | 86.4 | 152.4 | 5638 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 55.2 | 55.3 | 5876 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 50.3 | 50.4 | 5638 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 52.2 | 52.8 | 4894 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 9.3 | 28.0 | 53.4 | 53.5 | 6620 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 33.0 | 99.0 | 53.9 | 54.5 | 7750 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 8.4 | 23.0 | 45.8 | 51.9 | 5638 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 7.9 | 19.0 | 51.6 | 97.8 | 4894 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 3.3 | 23.0 | 49.8 | 93.5 | 6620 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 3.7 | 19.0 | 49.1 | 54.9 | 4894 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 10.5 | 23.0 | 53.1 | 118.5 | 6620 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 6.4 | 17.0 | 61.6 | 113.3 | 4894 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 8.1 | 23.0 | 51.8 | 99.4 | 5638 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 5.3 | 19.0 | 48.4 | 55.3 | 5876 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 23.0 | 58.3 | 174.6 | 5638 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 10.3 | 37.0 | 52.3 | 110.9 | 4894 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 14.3 | 71.0 | 47.9 | 54.3 | 6620 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 22.1 | 97.0 | 87.0 | 181.3 | 7750 |

### ctx_000512.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 7.2 | 29.0 | 51.9 | 52.0 | 5638 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 16.2 | 65.0 | 52.0 | 53.9 | 4894 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 24.5 | 98.0 | 50.9 | 54.4 | 6620 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 22.0 | 88.0 | 77.2 | 149.8 | 4894 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 24.0 | 96.0 | 77.0 | 152.5 | 6620 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 33.5 | 76.0 | 74.2 | 143.6 | 4894 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 30.5 | 99.0 | 93.9 | 150.9 | 5638 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 21.8 | 87.0 | 78.6 | 152.2 | 5876 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 42.5 | 99.0 | 74.2 | 145.1 | 5638 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 29.5 | 88.0 | 74.9 | 145.4 | 4894 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 32.0 | 98.0 | 100.1 | 150.3 | 6620 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 15.8 | 61.0 | 62.8 | 86.7 | 7750 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 5.0 | 20.0 | 61.3 | 140.2 | 5638 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 3.3 | 17.0 | 55.3 | 76.7 | 4894 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 9.6 | 20.0 | 64.1 | 166.5 | 6620 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 7.7 | 17.0 | 63.1 | 140.5 | 4894 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 3.7 | 23.0 | 57.4 | 109.2 | 6620 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 10.3 | 19.0 | 45.1 | 57.5 | 4894 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 11.1 | 20.0 | 68.0 | 180.1 | 5638 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 7.3 | 17.0 | 66.8 | 151.6 | 5876 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 8.9 | 21.0 | 55.6 | 104.0 | 5638 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 9.6 | 19.0 | 52.3 | 63.7 | 4894 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 9.7 | 21.0 | 67.6 | 177.2 | 6620 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 28.0 | 60.0 | 83.7 | 176.9 | 7750 |

### ctx_001024.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 38.8 | 97.0 | 86.1 | 140.8 | 5638 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 34.6 | 87.0 | 89.5 | 155.0 | 4894 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 38.2 | 98.0 | 88.0 | 153.6 | 6620 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 29.6 | 76.0 | 90.8 | 147.5 | 4894 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 39.0 | 98.0 | 92.3 | 154.2 | 6620 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 45.0 | 86.0 | 79.0 | 141.4 | 4894 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 54.2 | 98.0 | 98.6 | 148.7 | 5638 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 38.8 | 87.0 | 87.0 | 149.6 | 5876 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 48.2 | 98.0 | 105.5 | 149.1 | 5638 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 39.6 | 86.0 | 74.6 | 148.6 | 4894 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 43.4 | 100.0 | 84.1 | 155.1 | 6620 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 16.8 | 47.0 | 76.2 | 155.1 | 7750 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 10.5 | 20.0 | 61.1 | 141.0 | 5638 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 8.5 | 19.0 | 56.6 | 121.8 | 4894 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 6.9 | 20.0 | 66.7 | 171.5 | 6620 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 4.6 | 17.0 | 60.2 | 105.0 | 4894 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 8.9 | 20.0 | 74.1 | 178.0 | 6620 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 4.8 | 19.0 | 49.7 | 57.9 | 4894 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 5.3 | 20.0 | 61.2 | 123.8 | 5638 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 6.5 | 27.0 | 74.4 | 175.2 | 5876 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 7.7 | 45.0 | 59.8 | 145.8 | 5638 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 36.0 | 52.6 | 53.4 | 4894 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 4.6 | 21.0 | 51.9 | 55.3 | 6620 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 23.0 | 60.0 | 69.1 | 152.0 | 7750 |

### ctx_002048.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 41.7 | 99.0 | 81.1 | 145.1 | 5638 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 34.9 | 87.0 | 80.0 | 149.2 | 4894 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 43.1 | 98.0 | 94.9 | 157.1 | 6620 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 36.3 | 88.0 | 89.2 | 157.4 | 4894 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 51.0 | 99.0 | 94.3 | 156.3 | 6620 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 44.0 | 88.0 | 80.0 | 144.8 | 4894 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 45.6 | 99.0 | 101.9 | 151.8 | 5638 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 49.3 | 87.0 | 88.4 | 157.6 | 5876 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 52.3 | 99.0 | 93.6 | 149.2 | 5638 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 37.8 | 89.0 | 98.9 | 147.4 | 4894 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 43.8 | 99.0 | 94.2 | 152.2 | 6620 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 33.0 | 99.0 | 79.7 | 133.2 | 7750 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 6.9 | 20.0 | 76.9 | 151.7 | 5638 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 19.0 | 54.3 | 124.1 | 4894 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 7.9 | 21.0 | 56.9 | 151.4 | 6620 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 7.2 | 18.0 | 50.8 | 53.1 | 4894 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 9.6 | 21.0 | 52.8 | 77.2 | 6620 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 5.9 | 19.0 | 73.6 | 170.0 | 4894 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 7.0 | 21.0 | 58.2 | 102.5 | 5638 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 7.6 | 18.0 | 50.3 | 56.3 | 5876 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 8.5 | 21.0 | 56.6 | 146.6 | 5638 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 7.4 | 18.0 | 52.0 | 75.9 | 4894 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 8.2 | 21.0 | 54.9 | 82.9 | 6620 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 25.8 | 60.0 | 86.6 | 189.6 | 7750 |

### ctx_004096.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 63.9 | 100.0 | 104.7 | 145.8 | 5638 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 43.3 | 88.0 | 97.7 | 157.8 | 4894 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 48.9 | 98.0 | 111.7 | 155.8 | 6620 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 45.9 | 88.0 | 94.3 | 157.7 | 4894 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 54.0 | 99.0 | 103.8 | 164.2 | 6620 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 42.4 | 88.0 | 91.6 | 146.2 | 4894 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 49.4 | 99.0 | 97.2 | 151.5 | 5638 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 40.4 | 87.0 | 89.9 | 158.6 | 5876 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 52.4 | 99.0 | 95.6 | 153.1 | 5638 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 43.2 | 88.0 | 87.3 | 155.9 | 4894 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 52.9 | 100.0 | 96.2 | 151.7 | 6620 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 43.6 | 99.0 | 85.2 | 145.2 | 7750 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 7.6 | 21.0 | 63.2 | 127.7 | 5638 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 20.0 | 45.9 | 53.5 | 4894 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 9.9 | 21.0 | 54.2 | 83.0 | 6620 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 6.9 | 18.0 | 65.0 | 161.8 | 4894 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 8.5 | 21.0 | 60.3 | 98.8 | 6620 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 7.0 | 19.0 | 70.7 | 162.1 | 4894 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 11.0 | 45.0 | 63.9 | 159.1 | 5638 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 8.6 | 27.0 | 54.7 | 57.2 | 5876 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 10.4 | 45.0 | 54.2 | 87.2 | 5638 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 9.7 | 36.0 | 65.0 | 164.5 | 4894 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 6.4 | 21.0 | 56.4 | 95.8 | 6620 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 17.8 | 60.0 | 72.1 | 159.2 | 7750 |

### ctx_008192.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 63.2 | 99.0 | 105.0 | 145.7 | 5638 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 55.0 | 87.0 | 99.2 | 152.7 | 4894 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 63.2 | 99.0 | 112.6 | 157.9 | 6620 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 51.6 | 88.0 | 110.0 | 168.3 | 4894 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 61.9 | 100.0 | 115.2 | 162.9 | 6620 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 50.5 | 88.0 | 98.3 | 147.2 | 4894 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 57.2 | 100.0 | 106.8 | 152.1 | 5638 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 48.1 | 88.0 | 96.8 | 153.1 | 5876 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 56.6 | 99.0 | 102.5 | 155.4 | 5638 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 47.2 | 88.0 | 98.5 | 160.5 | 4894 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 55.4 | 100.0 | 106.0 | 156.2 | 6620 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 47.3 | 99.0 | 90.7 | 154.2 | 7750 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 3.8 | 24.0 | 53.6 | 95.1 | 5638 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 4.7 | 20.0 | 50.6 | 54.5 | 4894 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 24.0 | 50.9 | 60.3 | 6620 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 4.7 | 20.0 | 58.4 | 118.7 | 4894 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 3.8 | 24.0 | 64.8 | 130.9 | 6620 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 3.5 | 17.0 | 52.5 | 62.0 | 4894 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 7.3 | 23.0 | 53.1 | 71.0 | 5638 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 10.4 | 48.0 | 60.1 | 141.1 | 5876 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 19.1 | 98.0 | 61.7 | 122.1 | 5638 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 18.4 | 86.0 | 65.4 | 153.4 | 4894 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 15.7 | 99.0 | 74.1 | 155.6 | 6620 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 22.8 | 99.0 | 73.1 | 174.2 | 7750 |

### ctx_016384.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 60.4 | 99.0 | 104.8 | 149.0 | 5638 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 50.3 | 87.0 | 104.0 | 155.5 | 4894 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 63.1 | 100.0 | 110.5 | 162.1 | 6620 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 54.0 | 91.0 | 107.5 | 163.7 | 4894 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 59.8 | 100.0 | 118.4 | 161.1 | 6620 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 49.9 | 88.0 | 95.2 | 144.8 | 4894 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 59.2 | 100.0 | 106.6 | 152.1 | 5638 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 50.2 | 88.0 | 101.2 | 155.1 | 5876 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 59.4 | 99.0 | 106.9 | 157.2 | 5638 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 51.6 | 88.0 | 103.0 | 161.4 | 4894 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 63.1 | 100.0 | 107.5 | 155.8 | 6620 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 53.2 | 99.0 | 90.6 | 156.4 | 7750 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 10.0 | 42.0 | 60.0 | 142.2 | 5638 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 9.7 | 36.0 | 58.6 | 114.2 | 4894 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 9.7 | 36.0 | 55.0 | 89.0 | 6620 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 0.9 | 7.0 | 65.3 | 175.6 | 4894 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 6.0 | 22.0 | 64.1 | 177.9 | 6620 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 4.3 | 18.0 | 50.6 | 102.7 | 4894 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 8.2 | 21.0 | 68.5 | 183.4 | 5638 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 2.8 | 19.0 | 57.0 | 72.2 | 5876 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 13.4 | 45.0 | 55.8 | 113.3 | 5638 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 10.1 | 36.0 | 61.5 | 130.9 | 4894 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 9.4 | 30.0 | 61.0 | 165.6 | 6620 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 14.0 | 42.0 | 79.0 | 159.2 | 7750 |


## CPU / RAM / swap load by stage

### ctx_000256.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 36.9 %, max 37.5 %

RAM used: avg 6976 MiB, max 7087 MiB, avg 43.9 %, max 44.6 %

Swap used: avg 409 MiB, max 409 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 23.4 | 32.4 |
| 1 | 13.3 | 19.2 |
| 2 | 99.5 | 100.0 |
| 3 | 11.3 | 15.0 |

Decode stage:

CPU total: avg 40.6 %, max 50.8 %

RAM used: avg 6975 MiB, max 7084 MiB, avg 43.9 %, max 44.6 %

Swap used: avg 409 MiB, max 409 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 89.8 | 100.0 |
| 1 | 17.1 | 22.2 |
| 2 | 34.5 | 100.0 |
| 3 | 20.7 | 27.7 |

### ctx_000512.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 39.0 %, max 45.4 %

RAM used: avg 7387 MiB, max 7404 MiB, avg 46.5 %, max 46.6 %

Swap used: avg 408 MiB, max 408 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 14.8 | 17.0 |
| 1 | 98.5 | 100.0 |
| 2 | 15.5 | 21.2 |
| 3 | 26.7 | 57.1 |

Decode stage:

CPU total: avg 40.3 %, max 50.3 %

RAM used: avg 7436 MiB, max 7518 MiB, avg 46.8 %, max 47.4 %

Swap used: avg 408 MiB, max 408 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 33.3 | 100.0 |
| 1 | 91.5 | 100.0 |
| 2 | 17.1 | 23.7 |
| 3 | 18.9 | 22.1 |

### ctx_001024.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 36.9 %, max 37.8 %

RAM used: avg 7893 MiB, max 7926 MiB, avg 49.7 %, max 49.9 %

Swap used: avg 408 MiB, max 408 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 14.1 | 17.5 |
| 1 | 99.7 | 100.0 |
| 2 | 15.6 | 18.5 |
| 3 | 17.7 | 22.8 |

Decode stage:

CPU total: avg 39.8 %, max 45.0 %

RAM used: avg 7956 MiB, max 8055 MiB, avg 50.1 %, max 50.7 %

Swap used: avg 408 MiB, max 408 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 19.1 | 27.6 |
| 1 | 49.7 | 100.0 |
| 2 | 44.5 | 100.0 |
| 3 | 45.7 | 100.0 |

### ctx_002048.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 36.8 %, max 49.7 %

RAM used: avg 8311 MiB, max 8435 MiB, avg 52.3 %, max 53.1 %

Swap used: avg 407 MiB, max 407 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 15.3 | 27.6 |
| 1 | 21.7 | 58.3 |
| 2 | 94.0 | 100.0 |
| 3 | 15.9 | 36.1 |

Decode stage:

CPU total: avg 40.4 %, max 51.4 %

RAM used: avg 8461 MiB, max 8534 MiB, avg 53.3 %, max 53.8 %

Swap used: avg 405 MiB, max 407 MiB, avg 1.6 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 24.8 | 63.8 |
| 1 | 58.5 | 100.0 |
| 2 | 58.9 | 100.0 |
| 3 | 18.8 | 25.9 |

### ctx_004096.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 37.7 %, max 49.1 %

RAM used: avg 8776 MiB, max 8911 MiB, avg 55.3 %, max 56.1 %

Swap used: avg 403 MiB, max 403 MiB, avg 1.6 %, max 1.6 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 46.3 | 100.0 |
| 1 | 14.3 | 22.1 |
| 2 | 73.0 | 100.0 |
| 3 | 16.8 | 30.6 |

Decode stage:

CPU total: avg 40.4 %, max 52.3 %

RAM used: avg 8997 MiB, max 9062 MiB, avg 56.7 %, max 57.1 %

Swap used: avg 403 MiB, max 403 MiB, avg 1.6 %, max 1.6 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 17.6 | 21.0 |
| 1 | 18.8 | 25.4 |
| 2 | 35.2 | 100.0 |
| 3 | 89.5 | 100.0 |

### ctx_008192.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 38.2 %, max 52.4 %

RAM used: avg 8673 MiB, max 9688 MiB, avg 54.6 %, max 61.0 %

Swap used: avg 403 MiB, max 403 MiB, avg 1.6 %, max 1.6 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 17.1 | 26.3 |
| 1 | 53.8 | 100.0 |
| 2 | 15.0 | 23.8 |
| 3 | 66.7 | 100.0 |

Decode stage:

CPU total: avg 39.7 %, max 51.3 %

RAM used: avg 8929 MiB, max 9055 MiB, avg 56.2 %, max 57.0 %

Swap used: avg 403 MiB, max 403 MiB, avg 1.6 %, max 1.6 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 18.1 | 23.5 |
| 1 | 21.6 | 69.8 |
| 2 | 18.8 | 23.1 |
| 3 | 100.0 | 100.0 |

### ctx_016384.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 38.2 %, max 52.4 %

RAM used: avg 10122 MiB, max 10578 MiB, avg 63.8 %, max 66.6 %

Swap used: avg 403 MiB, max 403 MiB, avg 1.6 %, max 1.6 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 16.2 | 33.2 |
| 1 | 63.5 | 100.0 |
| 2 | 16.0 | 36.2 |
| 3 | 56.8 | 100.0 |

Decode stage:

CPU total: avg 40.0 %, max 50.6 %

RAM used: avg 10467 MiB, max 10564 MiB, avg 65.9 %, max 66.5 %

Swap used: avg 403 MiB, max 403 MiB, avg 1.6 %, max 1.6 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 17.9 | 23.2 |
| 1 | 22.3 | 65.9 |
| 2 | 19.7 | 27.4 |
| 3 | 100.0 | 100.0 |

