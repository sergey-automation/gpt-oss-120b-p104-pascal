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
- TURBOPREFILL: `0`
- TurboPrefill status: `TurboPrefill implementation detected; inactive (TURBOPREFILL=0)`
- TurboPrefill version: `TurboPrefill`
- llama.cpp git describe: `b10451-1-gba0d2b391-dirty`
- llama.cpp git commit: `ba0d2b3918c4662d8a1fb2eee21c365265f2901f`
- Server PID: `240646`
- KEEP_SERVER_RUNNING: `1`
- Parallel-slots mode: `active_slots=1..PARALLEL`
- Metrics policy: `server per-request timings only; no combined throughput calculated`
- llama_server_log: `/home/serg/workspace/versions/TurboPrefill_b10451/bench_reports_Qwen3.8-27B-BF16/20260816_170605/llama_server.log`

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
Память:         15Gi       7.8Gi       324Mi       1.0Gi       8.7Gi       7.7Gi
Подкачка:       23Gi       343Mi        23Gi
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
18.59.216.901 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
21.35.693.497 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
21.39.441.124 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
24.08.879.810 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
24.12.844.493 I decode: TurboPrefill requested=0 active=0 n_tokens=623 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
24.35.929.862 I decode: TurboPrefill requested=0 active=0 n_tokens=623 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
25.03.923.585 I decode: TurboPrefill requested=0 active=0 n_tokens=232 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
25.11.966.734 I decode: TurboPrefill requested=0 active=0 n_tokens=232 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
25.35.081.161 I decode: TurboPrefill requested=0 active=0 n_tokens=505 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
25.52.296.335 I decode: TurboPrefill requested=0 active=0 n_tokens=505 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
26.17.152.343 I decode: TurboPrefill requested=0 active=0 n_tokens=1046 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
26.53.782.139 I decode: TurboPrefill requested=0 active=0 n_tokens=1046 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
27.23.257.966 I decode: TurboPrefill requested=0 active=0 n_tokens=2294 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
28.45.118.849 I decode: TurboPrefill requested=0 active=0 n_tokens=2294 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
29.14.323.945 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
31.39.782.062 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
31.43.499.912 I decode: TurboPrefill requested=0 active=0 n_tokens=155 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
31.49.063.224 I decode: TurboPrefill requested=0 active=0 n_tokens=155 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
32.19.875.344 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=3 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
34.45.243.898 I decode: TurboPrefill requested=0 active=0 n_tokens=4097 n_ubatch=32 n_rs_seq=0 turbo_ubatches=0 devices=12 pipeline=1 split_mode=1 embeddings_nextn=1
```

## Server command

```bash
/home/serg/workspace/versions/TurboPrefill_b10451/build/bin/llama-server -m /mnt/models/AI/LLM/Qwen3.8-27B-BF16.gguf --host 0.0.0.0 --port 8081 -lv 4 -ngl 999 -c 260000 --override-kv llama.context_length=int:260000 -b 4097 -ub 32 -np 1 -ctk f16 -ctv f16 -sm layer -ts 10/10/10/10/10/10/10/10/10/10/10/10 --flash-attn auto --no-warmup --no-mmproj --spec-type draft-mtp --spec-draft-n-max 3
```

Server PID: `240646`  
Stop command: `kill -INT 240646`

## Summary

| File | Active slots | Request | Prompt tokens | Completion tokens | Prefill tok/s | Prefill time s | Decode tok/s | Decode time s | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ctx_000256.txt | 1 | 1 | 268 | 128 | 23.13 | 11.59 | 6.90 | 18.39 | 34.30 |
| ctx_000512.txt | 1 | 1 | 541 | 128 | 25.77 | 21.00 | 6.42 | 19.79 | 41.94 |
| ctx_001024.txt | 1 | 1 | 1082 | 128 | 26.41 | 40.97 | 5.37 | 23.65 | 65.89 |
| ctx_002048.txt | 1 | 1 | 2330 | 128 | 26.65 | 87.42 | 5.85 | 21.69 | 110.61 |
| ctx_004096.txt | 1 | 1 | 4288 | 128 | 27.07 | 158.41 | 6.21 | 20.45 | 180.78 |
| ctx_008192.txt | 1 | 1 | 8853 | 128 | 26.86 | 329.63 | 6.79 | 18.71 | 355.00 |
| ctx_016384.txt | 1 | 1 | 17670 | 128 | 26.40 | 669.35 | 5.25 | 24.19 | 697.97 |

## GPU load by stage

### ctx_000256.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 10.6 | 53.0 | 63.1 | 112.9 | 5634 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 21.6 | 55.0 | 51.3 | 53.9 | 4890 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 3.6 | 18.0 | 66.0 | 121.6 | 6616 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 52.2 | 55.5 | 4890 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 53.1 | 54.4 | 6616 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 9.6 | 48.0 | 49.8 | 51.3 | 4890 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 60.4 | 91.3 | 5634 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 62.9 | 98.2 | 5872 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 14.0 | 70.0 | 49.8 | 50.8 | 5634 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 12.2 | 61.0 | 51.7 | 53.3 | 4890 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 7.0 | 35.0 | 72.4 | 153.4 | 6616 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 22.2 | 99.0 | 54.1 | 54.4 | 7746 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 15.4 | 45.0 | 51.6 | 53.8 | 5634 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 10.4 | 36.0 | 56.0 | 91.8 | 4890 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 11.6 | 36.0 | 61.8 | 159.1 | 6616 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 3.5 | 17.0 | 56.0 | 76.5 | 4890 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 9.8 | 20.0 | 63.5 | 168.7 | 6616 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 6.8 | 17.0 | 51.9 | 65.4 | 4890 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 8.4 | 20.0 | 65.9 | 181.0 | 5634 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 2.0 | 17.0 | 53.8 | 57.7 | 5872 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 6.5 | 20.0 | 57.4 | 121.7 | 5634 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 3.9 | 17.0 | 62.0 | 125.3 | 4890 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 7.1 | 21.0 | 67.0 | 184.0 | 6616 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 16.2 | 59.0 | 77.6 | 189.6 | 7746 |

### ctx_000512.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 10.9 | 79.0 | 62.1 | 142.1 | 5634 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 3.8 | 29.0 | 63.6 | 146.8 | 4890 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 11.3 | 66.0 | 63.0 | 149.9 | 6616 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 7.4 | 51.0 | 56.1 | 60.9 | 4890 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 14.6 | 71.0 | 64.7 | 147.0 | 6616 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 3.4 | 20.0 | 61.0 | 142.5 | 4890 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 10.2 | 47.0 | 53.9 | 54.8 | 5634 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 11.3 | 72.0 | 66.0 | 153.6 | 5872 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 17.4 | 85.0 | 61.3 | 145.8 | 5634 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 6.8 | 53.0 | 63.3 | 140.9 | 4890 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 3.8 | 24.0 | 63.0 | 141.3 | 6616 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 2.1 | 19.0 | 64.7 | 148.4 | 7746 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 3.2 | 19.0 | 58.3 | 117.1 | 5634 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 3.4 | 17.0 | 57.2 | 78.1 | 4890 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 8.4 | 23.0 | 51.9 | 116.7 | 6616 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 11.5 | 36.0 | 47.3 | 55.4 | 4890 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 9.4 | 36.0 | 45.6 | 54.3 | 6616 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 7.8 | 17.0 | 52.7 | 64.6 | 4890 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 10.1 | 23.0 | 54.5 | 124.5 | 5634 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 3.4 | 17.0 | 57.1 | 82.1 | 5872 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 9.9 | 23.0 | 52.3 | 114.4 | 5634 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 4.6 | 19.0 | 52.7 | 93.4 | 4890 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 8.9 | 23.0 | 59.8 | 134.3 | 6616 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 16.8 | 57.0 | 74.4 | 164.4 | 7746 |

### ctx_001024.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 10.6 | 85.0 | 61.6 | 142.4 | 5634 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 9.0 | 75.0 | 59.7 | 152.2 | 4890 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 7.6 | 67.0 | 62.1 | 145.1 | 6616 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 3.2 | 60.0 | 64.8 | 145.7 | 4890 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 9.3 | 71.0 | 59.1 | 151.1 | 6616 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 11.1 | 74.0 | 57.3 | 147.5 | 4890 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 19.9 | 98.0 | 62.8 | 141.2 | 5634 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 7.1 | 60.0 | 69.4 | 148.9 | 5872 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 11.9 | 72.0 | 61.6 | 146.7 | 5634 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 3.0 | 57.0 | 58.5 | 148.6 | 4890 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 8.5 | 80.0 | 61.6 | 146.1 | 6616 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 9.5 | 99.0 | 62.3 | 147.4 | 7746 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 8.6 | 23.0 | 48.0 | 91.4 | 5634 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 6.1 | 19.0 | 49.6 | 92.0 | 4890 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 4.4 | 23.0 | 49.5 | 118.1 | 6616 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 9.0 | 19.0 | 46.3 | 53.9 | 4890 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 7.7 | 21.0 | 49.6 | 62.9 | 6616 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 9.7 | 19.0 | 51.6 | 117.5 | 4890 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 11.4 | 23.0 | 52.8 | 126.8 | 5634 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 9.1 | 19.0 | 49.2 | 58.8 | 5872 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 9.1 | 21.0 | 62.1 | 176.1 | 5634 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 10.5 | 18.0 | 53.3 | 97.7 | 4890 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 8.7 | 21.0 | 60.3 | 131.5 | 6616 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 14.7 | 51.0 | 90.1 | 170.2 | 7746 |

### ctx_002048.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 8.7 | 71.0 | 58.9 | 148.0 | 5634 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 6.2 | 60.0 | 63.0 | 155.3 | 4890 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 8.5 | 70.0 | 58.7 | 150.9 | 6616 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 8.2 | 60.0 | 60.0 | 150.1 | 4890 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 10.1 | 74.0 | 62.1 | 152.1 | 6616 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 12.0 | 73.0 | 58.3 | 137.6 | 4890 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 14.3 | 99.0 | 64.0 | 153.7 | 5634 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 4.5 | 59.0 | 66.0 | 155.8 | 5872 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 10.4 | 72.0 | 55.4 | 139.9 | 5634 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 10.7 | 59.0 | 55.4 | 127.8 | 4890 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 14.2 | 100.0 | 62.6 | 151.0 | 6616 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 8.2 | 99.0 | 60.4 | 137.1 | 7746 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 20.0 | 52.1 | 56.2 | 5634 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 3.5 | 19.0 | 54.2 | 79.1 | 4890 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 7.2 | 21.0 | 63.1 | 142.7 | 6616 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 6.5 | 18.0 | 60.6 | 152.8 | 4890 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 7.2 | 21.0 | 57.5 | 150.7 | 6616 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 10.8 | 36.0 | 46.7 | 51.7 | 4890 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 12.2 | 45.0 | 52.9 | 53.7 | 5634 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 8.2 | 18.0 | 50.5 | 56.8 | 5872 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 6.0 | 21.0 | 63.6 | 153.4 | 5634 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 3.3 | 19.0 | 61.6 | 119.8 | 4890 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 7.6 | 21.0 | 56.6 | 130.8 | 6616 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 13.3 | 60.0 | 82.6 | 164.9 | 7746 |

### ctx_004096.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 9.9 | 79.0 | 62.4 | 149.4 | 5634 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 9.9 | 76.0 | 59.3 | 154.2 | 4890 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 11.4 | 100.0 | 61.5 | 154.6 | 6616 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 7.4 | 81.0 | 64.5 | 156.3 | 4890 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 8.0 | 98.0 | 67.2 | 159.5 | 6616 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 6.2 | 71.0 | 56.2 | 138.1 | 4890 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 8.7 | 99.0 | 62.2 | 152.1 | 5634 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 9.4 | 60.0 | 62.8 | 154.9 | 5872 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 12.1 | 99.0 | 62.0 | 147.1 | 5634 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 8.0 | 88.0 | 57.7 | 149.1 | 4890 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 11.4 | 99.0 | 64.1 | 154.8 | 6616 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 6.0 | 99.0 | 59.7 | 151.8 | 7746 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 5.4 | 20.0 | 64.7 | 177.6 | 5634 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 3.5 | 17.0 | 56.9 | 94.4 | 4890 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 9.7 | 21.0 | 62.3 | 153.3 | 6616 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 6.8 | 18.0 | 67.1 | 174.1 | 4890 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 9.7 | 21.0 | 69.2 | 172.9 | 6616 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 3.4 | 17.0 | 52.0 | 55.5 | 4890 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 4.2 | 21.0 | 57.2 | 87.5 | 5634 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 8.2 | 17.0 | 71.4 | 169.3 | 5872 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 7.7 | 21.0 | 59.6 | 137.5 | 5634 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 7.0 | 17.0 | 61.3 | 140.9 | 4890 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 11.3 | 21.0 | 57.2 | 107.2 | 6616 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 29.7 | 60.0 | 73.3 | 172.5 | 7746 |

### ctx_008192.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 10.6 | 98.0 | 60.7 | 148.5 | 5634 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 9.5 | 82.0 | 61.5 | 153.3 | 4890 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 11.1 | 99.0 | 58.6 | 153.1 | 6616 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 9.1 | 90.0 | 64.8 | 169.3 | 4890 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 10.5 | 99.0 | 64.6 | 159.1 | 6616 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 8.4 | 72.0 | 57.5 | 146.3 | 4890 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 7.9 | 73.0 | 64.9 | 153.4 | 5634 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 7.6 | 82.0 | 62.0 | 159.6 | 5872 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 9.3 | 96.0 | 59.6 | 149.7 | 5634 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 9.0 | 69.0 | 58.4 | 150.9 | 4890 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 11.2 | 100.0 | 64.8 | 156.3 | 6616 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 8.4 | 99.0 | 60.8 | 150.6 | 7746 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 5.1 | 21.0 | 51.0 | 54.1 | 5634 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 4.8 | 17.0 | 55.0 | 78.0 | 4890 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 6.8 | 21.0 | 62.8 | 180.7 | 6616 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 5.0 | 23.0 | 60.9 | 147.4 | 4890 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 11.9 | 70.0 | 55.7 | 101.1 | 6616 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 8.8 | 50.0 | 55.8 | 92.4 | 4890 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 11.6 | 30.0 | 68.4 | 177.6 | 5634 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 7.6 | 20.0 | 61.7 | 117.6 | 5872 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 9.2 | 23.0 | 61.4 | 179.3 | 5634 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 5.8 | 18.0 | 58.1 | 110.0 | 4890 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 5.4 | 21.0 | 61.5 | 146.0 | 6616 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 21.8 | 92.0 | 86.5 | 198.8 | 7746 |

### ctx_016384.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 10.7 | 98.0 | 59.9 | 149.4 | 5634 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 9.4 | 89.0 | 62.3 | 154.2 | 4890 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 9.8 | 100.0 | 62.1 | 154.7 | 6616 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 8.0 | 88.0 | 61.9 | 157.9 | 4890 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 10.0 | 100.0 | 62.2 | 158.3 | 6616 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 9.4 | 89.0 | 58.7 | 147.4 | 4890 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 10.7 | 100.0 | 62.7 | 186.3 | 5634 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 9.1 | 89.0 | 64.7 | 159.2 | 5872 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 10.5 | 100.0 | 59.8 | 150.4 | 5634 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 9.7 | 88.0 | 61.7 | 151.2 | 4890 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 9.3 | 99.0 | 64.0 | 156.9 | 6616 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 8.0 | 99.0 | 58.3 | 151.3 | 7746 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 9.3 | 22.0 | 75.9 | 182.1 | 5634 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 3.8 | 18.0 | 58.6 | 126.3 | 4890 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 12.1 | 64.0 | 45.4 | 53.6 | 6616 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 10.9 | 63.0 | 63.2 | 183.1 | 4890 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 8.9 | 37.0 | 62.1 | 133.9 | 6616 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 5.2 | 20.0 | 58.3 | 173.3 | 4890 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 3.7 | 24.0 | 47.0 | 55.8 | 5634 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 6.9 | 19.0 | 58.1 | 149.5 | 5872 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 9.0 | 24.0 | 55.8 | 123.5 | 5634 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 18.0 | 56.4 | 120.1 | 4890 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 9.8 | 25.0 | 46.7 | 53.6 | 6616 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 27.7 | 86.0 | 67.1 | 145.8 | 7746 |


## CPU / RAM / swap load by stage

### ctx_000256.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 39.7 %, max 50.9 %

RAM used: avg 4740 MiB, max 4768 MiB, avg 29.9 %, max 30.0 %

Swap used: avg 407 MiB, max 407 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 19.0 | 26.5 |
| 1 | 14.0 | 16.1 |
| 2 | 25.9 | 63.8 |
| 3 | 99.8 | 100.0 |

Decode stage:

CPU total: avg 39.6 %, max 50.0 %

RAM used: avg 4773 MiB, max 4825 MiB, avg 30.1 %, max 30.4 %

Swap used: avg 407 MiB, max 407 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 40.5 | 100.0 |
| 1 | 20.7 | 48.8 |
| 2 | 20.3 | 70.0 |
| 3 | 76.6 | 100.0 |

### ctx_000512.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 37.8 %, max 52.2 %

RAM used: avg 5214 MiB, max 5290 MiB, avg 32.8 %, max 33.3 %

Swap used: avg 407 MiB, max 407 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 99.2 | 100.0 |
| 1 | 13.8 | 18.0 |
| 2 | 21.9 | 66.0 |
| 3 | 16.1 | 29.2 |

Decode stage:

CPU total: avg 40.6 %, max 52.7 %

RAM used: avg 5334 MiB, max 5391 MiB, avg 33.6 %, max 34.0 %

Swap used: avg 406 MiB, max 407 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 72.6 | 100.0 |
| 1 | 46.3 | 100.0 |
| 2 | 25.2 | 71.6 |
| 3 | 17.8 | 22.6 |

### ctx_001024.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 37.1 %, max 46.5 %

RAM used: avg 5602 MiB, max 5787 MiB, avg 35.3 %, max 36.5 %

Swap used: avg 406 MiB, max 406 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 55.5 | 100.0 |
| 1 | 17.4 | 25.7 |
| 2 | 59.4 | 100.0 |
| 3 | 15.8 | 27.8 |

Decode stage:

CPU total: avg 40.6 %, max 45.2 %

RAM used: avg 5797 MiB, max 5895 MiB, avg 36.5 %, max 37.1 %

Swap used: avg 406 MiB, max 406 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 36.9 | 100.0 |
| 1 | 58.9 | 100.0 |
| 2 | 46.9 | 100.0 |
| 3 | 19.4 | 25.0 |

### ctx_002048.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 37.4 %, max 51.3 %

RAM used: avg 6054 MiB, max 6295 MiB, avg 38.1 %, max 39.7 %

Swap used: avg 406 MiB, max 406 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 17.0 | 27.0 |
| 1 | 54.0 | 100.0 |
| 2 | 62.9 | 100.0 |
| 3 | 15.7 | 26.4 |

Decode stage:

CPU total: avg 39.9 %, max 49.7 %

RAM used: avg 6326 MiB, max 6385 MiB, avg 39.8 %, max 40.2 %

Swap used: avg 406 MiB, max 406 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 18.9 | 29.0 |
| 1 | 100.0 | 100.0 |
| 2 | 20.9 | 72.1 |
| 3 | 19.4 | 28.2 |

### ctx_004096.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 37.7 %, max 52.9 %

RAM used: avg 6663 MiB, max 6925 MiB, avg 42.0 %, max 43.6 %

Swap used: avg 406 MiB, max 406 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 16.2 | 30.0 |
| 1 | 47.9 | 100.0 |
| 2 | 71.4 | 100.0 |
| 3 | 15.1 | 38.0 |

Decode stage:

CPU total: avg 40.2 %, max 50.1 %

RAM used: avg 6948 MiB, max 7017 MiB, avg 43.8 %, max 44.2 %

Swap used: avg 406 MiB, max 406 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 18.3 | 25.9 |
| 1 | 22.9 | 58.7 |
| 2 | 73.6 | 100.0 |
| 3 | 45.5 | 100.0 |

### ctx_008192.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 37.6 %, max 52.6 %

RAM used: avg 6342 MiB, max 7725 MiB, avg 39.9 %, max 48.7 %

Swap used: avg 405 MiB, max 405 MiB, avg 1.6 %, max 1.6 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 19.1 | 100.0 |
| 1 | 48.2 | 100.0 |
| 2 | 30.4 | 100.0 |
| 3 | 52.6 | 100.0 |

Decode stage:

CPU total: avg 39.3 %, max 52.1 %

RAM used: avg 6589 MiB, max 6727 MiB, avg 41.5 %, max 42.4 %

Swap used: avg 405 MiB, max 405 MiB, avg 1.6 %, max 1.6 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 17.3 | 24.5 |
| 1 | 21.4 | 69.3 |
| 2 | 100.0 | 100.0 |
| 3 | 18.2 | 24.8 |

### ctx_016384.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 37.7 %, max 52.8 %

RAM used: avg 7431 MiB, max 7776 MiB, avg 46.8 %, max 49.0 %

Swap used: avg 404 MiB, max 405 MiB, avg 1.6 %, max 1.6 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 53.6 | 100.0 |
| 1 | 16.0 | 33.3 |
| 2 | 55.0 | 100.0 |
| 3 | 25.8 | 100.0 |

Decode stage:

CPU total: avg 39.8 %, max 49.7 %

RAM used: avg 7842 MiB, max 7914 MiB, avg 49.4 %, max 49.8 %

Swap used: avg 404 MiB, max 404 MiB, avg 1.6 %, max 1.6 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 23.5 | 67.5 |
| 1 | 19.7 | 27.0 |
| 2 | 100.0 | 100.0 |
| 3 | 15.9 | 20.7 |

