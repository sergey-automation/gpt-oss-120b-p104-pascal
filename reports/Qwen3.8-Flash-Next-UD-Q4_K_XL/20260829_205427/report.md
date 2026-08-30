# llama-server parallel-slots context benchmark report

## Test header

- MODEL: `/home/serg/workspace/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`
- NGL: `999`
- CTX_SIZE: `260000`
- N_GEN: `128`
- BATCH: `4097`
- UBATCH: `64`
- CTK: `f16`
- CTV: `f16`
- SPEC_TYPE: `none`
- SPEC_DRAFT_N_MAX: `3`
- SPLIT_MODE: `layer`
- TENSOR_SPLIT: `1/1/1/1/1/1/1/1/1/1/1/1/1/1/1`
- PARALLEL: `1`
- TEMPERATURE: `0.15`
- CACHE_PROMPT: `0`
- FLASH_ATTN: `auto`
- THREADS: `auto`
- THREADS_BATCH: `auto`
- REPEATS: `1`
- CUDA_VISIBLE_DEVICES: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14`
- TURBOPREFILL: `0`
- TurboPrefill status: `original llama.cpp / no TurboPrefill marker detected`
- TurboPrefill version: `not found`
- llama.cpp git describe: `b9190-1588-geaf937655-dirty`
- llama.cpp git commit: `eaf93765572e794b8e3754fe45adbe12d381e997`
- Server PID: `48827`
- KEEP_SERVER_RUNNING: `1`
- Parallel-slots mode: `active_slots=1..PARALLEL`
- Metrics policy: `server per-request timings only; no combined throughput calculated`
- llama_server_log: `/home/serg/workspace/versions/llama.cpp_qwen4exp_latest/bench_reports_Qwen3.8-Flash-Next-UD-Q4_K_XL/20260829_205427/llama_server.log`

## Environment

### TURBOPREFILL

```text
0
```

### RUN_DIR

```text
/home/serg/workspace/versions/llama.cpp_qwen4exp_latest
```

### CONFIG_PATH

```text
/home/serg/workspace/versions/llama.cpp_qwen4exp_latest/config_Qwen_3.8-Flash-Next.sh
```

### LLAMA_SERVER_BIN

```text
/home/serg/workspace/versions/llama.cpp_qwen4exp_latest/build/bin/llama-server
```

### LOCAL_LD_LIBRARY_PATH

```text
/home/serg/workspace/versions/llama.cpp_qwen4exp_latest/build/bin
```

### CUDA_VISIBLE_DEVICES_effective

```text
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
```

### selected_gpu_count

```text
15
```

### selected_gpu_models

```text
NVIDIA P104-100 x15
```

### llama_server_version

```text
version: 0.3.0-dev (build 10721, commit eaf937655)
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
Память:         15Gi       1.9Gi       335Mi       126Mi        13Gi        13Gi
Подкачка:       23Gi       270Mi        23Gi
```

### memory_modules

```text
unavailable: Command '['dmidecode', '--type', '17']' returned non-zero exit status 1.
```

### nvidia_smi

```text
0, NVIDIA P104-100, 00000000:01:00.0, 535.309.01, 8192 MiB, 1, 1
1, NVIDIA P104-100, 00000000:02:00.0, 535.309.01, 8192 MiB, 1, 1
2, NVIDIA P104-100, 00000000:06:00.0, 535.309.01, 8192 MiB, 1, 1
3, NVIDIA P104-100, 00000000:07:00.0, 535.309.01, 8192 MiB, 1, 1
4, NVIDIA P104-100, 00000000:08:00.0, 535.309.01, 8192 MiB, 1, 1
5, NVIDIA P104-100, 00000000:0A:00.0, 535.309.01, 8192 MiB, 1, 1
6, NVIDIA P104-100, 00000000:0C:00.0, 535.309.01, 8192 MiB, 1, 1
7, NVIDIA P104-100, 00000000:0D:00.0, 535.309.01, 8192 MiB, 1, 1
8, NVIDIA P104-100, 00000000:0E:00.0, 535.309.01, 8192 MiB, 1, 1
9, NVIDIA P104-100, 00000000:0F:00.0, 535.309.01, 8192 MiB, 1, 1
10, NVIDIA P104-100, 00000000:10:00.0, 535.309.01, 8192 MiB, 1, 1
11, NVIDIA P104-100, 00000000:11:00.0, 535.309.01, 8192 MiB, 1, 1
12, NVIDIA P104-100, 00000000:12:00.0, 535.309.01, 8192 MiB, 1, 1
13, NVIDIA P104-100, 00000000:13:00.0, 535.309.01, 8192 MiB, 1, 1
14, NVIDIA P104-100, 00000000:14:00.0, 535.309.01, 8192 MiB, 1, 1
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
b9190-1588-geaf937655-dirty
```

### git_commit

```text
eaf93765572e794b8e3754fe45adbe12d381e997
```

### git_last_commit

```text
2026-08-27 21:28:11 +0200
exclude from webgpu test
```

### git_turboprefill_hint

```text
none
```

### model_path

```text
/home/serg/workspace/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
```

### model_filename

```text
Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
```

### model_size_bytes

```text
10946624
```

### model_size_gib

```text
0.010
```

### gguf_architecture

```text
qwen4exp
```

### gguf_tensor_count

```text
0
```

### gguf_tensor_types

```text

```

### model_sha256

```text
disabled (MODEL_HASH=0)
```

### TurboPrefill runtime markers

```text
none
```

## Server command

```bash
/home/serg/workspace/versions/llama.cpp_qwen4exp_latest/build/bin/llama-server -m /home/serg/workspace/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --host 0.0.0.0 --port 8081 -lv 4 -ngl 999 -c 260000 --override-kv llama.context_length=int:260000 -b 4097 -ub 64 -np 1 -ctk f16 -ctv f16 -sm layer -ts 1/1/1/1/1/1/1/1/1/1/1/1/1/1/1 -ot 'per_layer_token_embd\.0\.weight=CUDA0,per_layer_token_embd\.1\.weight=CUDA1,per_layer_token_embd\.2\.weight=CUDA2,per_layer_token_embd\.3\.weight=CUDA3,per_layer_token_embd\.4\.weight=CUDA6,per_layer_token_embd\.5\.weight=CUDA4,per_layer_token_embd\.6\.weight=CUDA5,per_layer_token_embd\.7\.weight=CUDA12,per_layer_token_embd\.8\.weight=CUDA7,per_layer_token_embd\.9\.weight=CUDA8,per_layer_token_embd\.10\.weight=CUDA14,per_layer_token_embd\.11\.weight=CUDA9' --flash-attn auto --no-warmup --no-mmproj --spec-type none
```

Server PID: `48827`  
Stop command: `kill -INT 48827`

## Summary

| File | Active slots | Request | Prompt tokens | Completion tokens | Prefill tok/s | Prefill time s | Decode tok/s | Decode time s | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ctx_000256.txt | 1 | 1 | 268 | 128 | 36.95 | 7.25 | 11.86 | 10.71 | 20.32 |
| ctx_000512.txt | 1 | 1 | 541 | 128 | 42.26 | 12.80 | 11.72 | 10.84 | 24.47 |
| ctx_001024.txt | 1 | 1 | 1082 | 128 | 43.45 | 24.90 | 11.46 | 11.08 | 36.89 |
| ctx_002048.txt | 1 | 1 | 2330 | 128 | 47.45 | 49.11 | 11.12 | 11.42 | 61.56 |
| ctx_004096.txt | 1 | 1 | 4288 | 128 | 48.52 | 88.38 | 10.34 | 12.28 | 101.87 |
| ctx_008192.txt | 1 | 1 | 8853 | 128 | 48.00 | 184.46 | 9.41 | 13.50 | 201.70 |
| ctx_016384.txt | 1 | 1 | 17670 | 128 | 43.37 | 407.44 | 7.73 | 16.43 | 426.34 |

## GPU load by stage

### ctx_000256.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 8.3 | 13.0 | 51.2 | 51.3 | 7898 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 2.7 | 8.0 | 52.2 | 52.7 | 5144 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 4.3 | 13.0 | 48.9 | 54.0 | 5828 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 9.0 | 14.0 | 54.6 | 69.7 | 7414 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 21.0 | 50.0 | 62.1 | 96.0 | 5826 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 14.7 | 37.0 | 48.6 | 50.6 | 5828 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 62.6 | 86.2 | 4892 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 4.3 | 13.0 | 48.4 | 52.1 | 7412 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 2.3 | 7.0 | 48.7 | 50.3 | 5826 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 2.3 | 7.0 | 49.3 | 52.6 | 6076 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 2.3 | 7.0 | 53.7 | 54.3 | 5826 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 2.3 | 7.0 | 49.5 | 49.6 | 7412 |
| 12 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 50.0 | 52.0 | 4892 |
| 13 | NVIDIA P104-100 | Gen1 x1 | 1.0 | 3.0 | 50.8 | 52.5 | 5826 |
| 14 | NVIDIA P104-100 | Gen1 x1 | 11.7 | 28.0 | 43.3 | 43.3 | 5390 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 5.8 | 7.0 | 46.9 | 54.2 | 7898 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 4.3 | 6.0 | 46.0 | 52.7 | 5144 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 4.7 | 6.0 | 54.3 | 64.0 | 5828 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 5.8 | 7.0 | 55.5 | 71.4 | 7414 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 4.7 | 6.0 | 55.9 | 65.3 | 5826 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 4.3 | 6.0 | 51.9 | 55.1 | 5828 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 3.8 | 5.0 | 55.3 | 59.7 | 4892 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 5.0 | 6.0 | 53.9 | 60.5 | 7412 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 4.2 | 5.0 | 51.9 | 57.6 | 5826 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 4.2 | 5.0 | 54.6 | 61.5 | 6076 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 6.3 | 13.0 | 49.1 | 54.4 | 5826 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 14.7 | 57.0 | 53.1 | 66.9 | 7412 |
| 12 | NVIDIA P104-100 | Gen1 x1 | 13.5 | 56.0 | 55.2 | 66.4 | 4892 |
| 13 | NVIDIA P104-100 | Gen1 x1 | 10.3 | 37.0 | 57.1 | 72.8 | 5826 |
| 14 | NVIDIA P104-100 | Gen1 x1 | 11.7 | 14.0 | 45.3 | 51.1 | 5390 |

### ctx_000512.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 9.4 | 19.0 | 51.2 | 57.2 | 7900 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 52.0 | 52.7 | 5144 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 2.6 | 12.0 | 51.5 | 51.7 | 5828 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 13.4 | 67.0 | 52.1 | 52.3 | 7414 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 0.0 | 0.0 | 53.3 | 54.1 | 5826 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 3.2 | 15.0 | 73.1 | 163.7 | 5828 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 0.2 | 1.0 | 52.0 | 54.3 | 4892 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 5.2 | 26.0 | 52.0 | 52.0 | 7412 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 8.2 | 40.0 | 50.5 | 50.5 | 5826 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 10.0 | 49.0 | 52.7 | 52.9 | 6076 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 7.4 | 35.0 | 57.5 | 76.9 | 5826 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 14.2 | 39.0 | 49.6 | 49.7 | 7412 |
| 12 | NVIDIA P104-100 | Gen1 x1 | 15.2 | 51.0 | 51.9 | 52.0 | 4892 |
| 13 | NVIDIA P104-100 | Gen1 x1 | 27.2 | 58.0 | 52.5 | 52.6 | 5826 |
| 14 | NVIDIA P104-100 | Gen1 x1 | 6.8 | 21.0 | 43.8 | 44.0 | 5390 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 10.7 | 29.0 | 58.4 | 73.3 | 7900 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 8.7 | 27.0 | 57.9 | 65.7 | 5144 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 6.2 | 10.0 | 51.8 | 60.4 | 5828 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 6.0 | 7.0 | 50.1 | 58.9 | 7414 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 5.2 | 6.0 | 47.7 | 52.6 | 5826 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 5.2 | 6.0 | 43.8 | 50.6 | 5828 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 4.0 | 5.0 | 50.7 | 59.0 | 4892 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 5.7 | 7.0 | 50.2 | 62.6 | 7412 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 4.3 | 5.0 | 48.5 | 57.2 | 5826 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 4.3 | 5.0 | 50.9 | 60.4 | 6076 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 4.2 | 5.0 | 56.3 | 63.5 | 5826 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 5.7 | 7.0 | 46.7 | 66.2 | 7412 |
| 12 | NVIDIA P104-100 | Gen1 x1 | 4.3 | 5.0 | 48.3 | 58.3 | 4892 |
| 13 | NVIDIA P104-100 | Gen1 x1 | 4.0 | 5.0 | 49.7 | 60.1 | 5826 |
| 14 | NVIDIA P104-100 | Gen1 x1 | 14.3 | 20.0 | 45.0 | 49.6 | 5390 |

### ctx_001024.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 3.5 | 15.0 | 60.6 | 152.3 | 7900 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 9.3 | 55.0 | 67.8 | 183.8 | 5144 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 8.2 | 45.0 | 51.6 | 52.0 | 5828 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 12.6 | 73.0 | 53.2 | 65.9 | 7414 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 10.5 | 54.0 | 52.5 | 53.4 | 5826 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 2.2 | 24.0 | 62.6 | 179.2 | 5828 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 4.6 | 45.0 | 56.9 | 89.2 | 4892 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 4.4 | 48.0 | 52.2 | 52.8 | 7412 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 8.7 | 60.0 | 61.3 | 169.5 | 5826 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 8.3 | 52.0 | 52.4 | 53.0 | 6076 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 55.0 | 54.5 | 70.8 | 5826 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 11.9 | 59.0 | 49.2 | 49.7 | 7412 |
| 12 | NVIDIA P104-100 | Gen1 x1 | 15.2 | 56.0 | 53.4 | 67.2 | 4892 |
| 13 | NVIDIA P104-100 | Gen1 x1 | 5.6 | 48.0 | 60.5 | 144.5 | 5826 |
| 14 | NVIDIA P104-100 | Gen1 x1 | 1.5 | 16.0 | 44.2 | 47.5 | 5390 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 6.8 | 7.0 | 52.2 | 56.2 | 7900 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 4.8 | 5.0 | 53.7 | 56.6 | 5144 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 6.0 | 47.3 | 52.0 | 5828 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 6.8 | 7.0 | 48.3 | 52.6 | 7414 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 6.0 | 6.0 | 49.2 | 53.4 | 5826 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 6.0 | 6.0 | 52.7 | 60.9 | 5828 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 5.0 | 5.0 | 51.0 | 54.1 | 4892 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 7.0 | 7.0 | 54.4 | 71.3 | 7412 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 4.5 | 6.0 | 51.4 | 61.0 | 5826 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 4.7 | 5.0 | 50.8 | 53.2 | 6076 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 4.0 | 5.0 | 55.5 | 58.3 | 5826 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 5.8 | 7.0 | 51.0 | 55.4 | 7412 |
| 12 | NVIDIA P104-100 | Gen1 x1 | 5.0 | 5.0 | 49.9 | 53.2 | 4892 |
| 13 | NVIDIA P104-100 | Gen1 x1 | 5.0 | 5.0 | 52.3 | 53.5 | 5826 |
| 14 | NVIDIA P104-100 | Gen1 x1 | 13.3 | 14.0 | 47.4 | 57.6 | 5390 |

### ctx_002048.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 9.9 | 42.0 | 52.8 | 78.8 | 7900 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 4.6 | 52.0 | 56.7 | 136.2 | 5144 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 5.9 | 44.0 | 64.0 | 182.7 | 5828 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 1.1 | 24.0 | 57.9 | 182.0 | 7414 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 7.0 | 59.0 | 52.8 | 53.4 | 5826 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 9.2 | 59.0 | 50.3 | 50.8 | 5828 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 9.7 | 56.0 | 54.4 | 66.9 | 4892 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 9.8 | 69.0 | 61.7 | 173.6 | 7412 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 6.5 | 58.0 | 63.8 | 159.1 | 5826 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 8.5 | 60.0 | 53.0 | 58.0 | 6076 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 9.0 | 48.0 | 61.3 | 139.6 | 5826 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 12.3 | 67.0 | 49.1 | 52.9 | 7412 |
| 12 | NVIDIA P104-100 | Gen1 x1 | 6.7 | 45.0 | 55.8 | 132.7 | 4892 |
| 13 | NVIDIA P104-100 | Gen1 x1 | 3.7 | 42.0 | 67.5 | 184.0 | 5826 |
| 14 | NVIDIA P104-100 | Gen1 x1 | 0.8 | 11.0 | 43.9 | 44.1 | 5390 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 6.3 | 7.0 | 44.8 | 50.7 | 7900 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 4.5 | 5.0 | 52.8 | 56.6 | 5144 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 5.7 | 7.0 | 44.0 | 52.0 | 5828 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 7.2 | 8.0 | 45.2 | 52.6 | 7414 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 6.0 | 46.2 | 53.3 | 5826 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 5.3 | 6.0 | 44.2 | 50.9 | 5828 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 5.3 | 8.0 | 47.9 | 54.7 | 4892 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 6.5 | 10.0 | 46.2 | 53.0 | 7412 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 5.0 | 6.0 | 43.5 | 50.9 | 5826 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 6.3 | 8.0 | 45.5 | 53.3 | 6076 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 4.8 | 6.0 | 45.9 | 54.3 | 5826 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 7.0 | 10.0 | 50.4 | 51.3 | 7412 |
| 12 | NVIDIA P104-100 | Gen1 x1 | 4.5 | 5.0 | 46.9 | 47.4 | 4892 |
| 13 | NVIDIA P104-100 | Gen1 x1 | 5.8 | 8.0 | 48.2 | 48.9 | 5826 |
| 14 | NVIDIA P104-100 | Gen1 x1 | 11.3 | 14.0 | 45.6 | 47.2 | 5390 |

### ctx_004096.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 12.6 | 68.0 | 50.5 | 53.6 | 7900 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 10.1 | 59.0 | 61.6 | 178.2 | 5144 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 9.7 | 64.0 | 62.7 | 177.3 | 5828 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 9.2 | 76.0 | 62.1 | 186.5 | 7414 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 5.3 | 61.0 | 52.8 | 70.5 | 5826 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 8.7 | 64.0 | 53.5 | 94.9 | 5828 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 6.3 | 60.0 | 58.4 | 179.1 | 4892 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 12.3 | 70.0 | 55.3 | 172.3 | 7412 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 7.0 | 60.0 | 57.3 | 162.1 | 5826 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 6.5 | 57.0 | 56.9 | 136.4 | 6076 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 10.2 | 60.0 | 56.4 | 113.7 | 5826 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 10.7 | 68.0 | 53.0 | 159.7 | 7412 |
| 12 | NVIDIA P104-100 | Gen1 x1 | 4.7 | 56.0 | 59.0 | 182.0 | 4892 |
| 13 | NVIDIA P104-100 | Gen1 x1 | 3.2 | 58.0 | 57.9 | 166.4 | 5826 |
| 14 | NVIDIA P104-100 | Gen1 x1 | 1.5 | 29.0 | 44.8 | 77.0 | 5390 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 6.5 | 8.0 | 44.5 | 50.0 | 7900 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 4.0 | 5.0 | 46.3 | 52.3 | 5144 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 6.0 | 46.2 | 52.0 | 5828 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 6.5 | 8.0 | 48.5 | 57.1 | 7414 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 4.7 | 6.0 | 56.5 | 61.1 | 5826 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 5.3 | 6.0 | 44.9 | 51.1 | 5828 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 5.0 | 5.0 | 49.1 | 54.3 | 4892 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 6.0 | 7.0 | 49.5 | 58.5 | 7412 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 6.0 | 46.6 | 51.7 | 5826 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 5.0 | 6.0 | 48.5 | 54.3 | 6076 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 5.3 | 6.0 | 49.1 | 55.2 | 5826 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 6.0 | 7.0 | 45.5 | 52.1 | 7412 |
| 12 | NVIDIA P104-100 | Gen1 x1 | 4.3 | 5.0 | 46.9 | 54.5 | 4892 |
| 13 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 6.0 | 55.9 | 67.3 | 5826 |
| 14 | NVIDIA P104-100 | Gen1 x1 | 11.5 | 14.0 | 48.5 | 60.6 | 5390 |

### ctx_008192.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 9.1 | 71.0 | 58.1 | 176.1 | 7900 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 7.1 | 58.0 | 60.6 | 182.7 | 5144 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 7.5 | 64.0 | 55.1 | 180.6 | 5828 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 9.4 | 82.0 | 56.0 | 181.1 | 7414 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 9.8 | 72.0 | 56.2 | 172.5 | 5826 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 12.4 | 69.0 | 57.0 | 174.1 | 5828 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 8.9 | 65.0 | 58.1 | 172.4 | 4892 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 9.8 | 65.0 | 56.9 | 168.6 | 7412 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 8.1 | 65.0 | 57.9 | 162.5 | 5826 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 8.5 | 66.0 | 55.4 | 174.3 | 6076 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 6.7 | 68.0 | 60.0 | 181.7 | 5826 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 5.8 | 67.0 | 54.7 | 163.0 | 7412 |
| 12 | NVIDIA P104-100 | Gen1 x1 | 6.7 | 58.0 | 55.9 | 185.6 | 4892 |
| 13 | NVIDIA P104-100 | Gen1 x1 | 8.0 | 66.0 | 55.3 | 176.9 | 5826 |
| 14 | NVIDIA P104-100 | Gen1 x1 | 5.1 | 35.0 | 45.6 | 102.1 | 5390 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 5.2 | 9.0 | 44.6 | 50.6 | 7900 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 2.6 | 5.0 | 53.0 | 55.6 | 5144 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 4.9 | 8.0 | 45.7 | 52.0 | 5828 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 9.0 | 47.2 | 52.9 | 7414 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 5.2 | 8.0 | 48.0 | 53.3 | 5826 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 4.6 | 8.0 | 45.8 | 51.1 | 5828 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 9.4 | 54.0 | 53.7 | 62.1 | 4892 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 6.5 | 13.0 | 49.5 | 58.0 | 7412 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 4.4 | 7.0 | 48.1 | 61.4 | 5826 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 3.9 | 7.0 | 56.6 | 72.0 | 6076 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 9.9 | 44.0 | 50.1 | 54.5 | 5826 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 8.5 | 31.0 | 47.8 | 52.8 | 7412 |
| 12 | NVIDIA P104-100 | Gen1 x1 | 4.2 | 8.0 | 62.2 | 161.4 | 4892 |
| 13 | NVIDIA P104-100 | Gen1 x1 | 4.5 | 7.0 | 49.5 | 55.3 | 5826 |
| 14 | NVIDIA P104-100 | Gen1 x1 | 9.4 | 16.0 | 47.2 | 55.3 | 5390 |

### ctx_016384.txt | active_slots=1 | request=1

Prefill stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 8.3 | 82.0 | 56.6 | 162.4 | 7910 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 6.1 | 60.0 | 56.2 | 174.2 | 5144 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 8.6 | 71.0 | 58.2 | 179.4 | 5838 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 10.8 | 87.0 | 57.4 | 191.3 | 7424 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 7.7 | 75.0 | 56.5 | 185.6 | 5836 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 7.3 | 75.0 | 57.3 | 167.8 | 5838 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 60.0 | 56.7 | 171.1 | 4892 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 7.8 | 80.0 | 59.6 | 195.6 | 7422 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 6.9 | 78.0 | 53.6 | 175.3 | 5836 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 8.3 | 72.0 | 56.3 | 161.5 | 6088 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 10.3 | 75.0 | 57.7 | 183.8 | 5836 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 8.4 | 78.0 | 57.2 | 179.9 | 7422 |
| 12 | NVIDIA P104-100 | Gen1 x1 | 6.2 | 58.0 | 55.4 | 185.2 | 4892 |
| 13 | NVIDIA P104-100 | Gen1 x1 | 7.0 | 76.0 | 57.0 | 186.7 | 5836 |
| 14 | NVIDIA P104-100 | Gen1 x1 | 3.8 | 45.0 | 49.3 | 146.9 | 5400 |

Decode stage:

| GPU | name | PCIe | avg util % | max util % | avg W | max W | max VRAM MiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA P104-100 | Gen1 x1 | 5.9 | 11.0 | 44.7 | 50.6 | 7910 |
| 1 | NVIDIA P104-100 | Gen1 x1 | 6.4 | 28.0 | 45.0 | 52.5 | 5144 |
| 2 | NVIDIA P104-100 | Gen1 x1 | 6.6 | 18.0 | 44.5 | 54.0 | 5838 |
| 3 | NVIDIA P104-100 | Gen1 x1 | 9.5 | 27.0 | 45.2 | 52.9 | 7424 |
| 4 | NVIDIA P104-100 | Gen1 x1 | 6.6 | 15.0 | 44.8 | 53.3 | 5836 |
| 5 | NVIDIA P104-100 | Gen1 x1 | 4.9 | 9.0 | 42.7 | 51.0 | 5838 |
| 6 | NVIDIA P104-100 | Gen1 x1 | 6.2 | 27.0 | 46.4 | 53.9 | 4892 |
| 7 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 10.0 | 45.4 | 53.7 | 7422 |
| 8 | NVIDIA P104-100 | Gen1 x1 | 6.0 | 9.0 | 44.6 | 50.9 | 5836 |
| 9 | NVIDIA P104-100 | Gen1 x1 | 4.8 | 9.0 | 47.3 | 55.9 | 6088 |
| 10 | NVIDIA P104-100 | Gen1 x1 | 4.0 | 7.0 | 48.1 | 57.4 | 5836 |
| 11 | NVIDIA P104-100 | Gen1 x1 | 4.6 | 7.0 | 44.3 | 54.6 | 7422 |
| 12 | NVIDIA P104-100 | Gen1 x1 | 2.4 | 5.0 | 54.5 | 67.3 | 4892 |
| 13 | NVIDIA P104-100 | Gen1 x1 | 5.5 | 9.0 | 46.1 | 52.6 | 5836 |
| 14 | NVIDIA P104-100 | Gen1 x1 | 11.0 | 17.0 | 46.5 | 53.8 | 5400 |


## CPU / RAM / swap load by stage

### ctx_000256.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 28.3 %, max 29.0 %

RAM used: avg 4354 MiB, max 4374 MiB, avg 27.4 %, max 27.6 %

Swap used: avg 406 MiB, max 406 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 9.0 | 11.6 |
| 1 | 6.7 | 11.1 |
| 2 | 75.1 | 99.5 |
| 3 | 22.7 | 62.0 |

Decode stage:

CPU total: avg 31.6 %, max 44.4 %

RAM used: avg 4489 MiB, max 4528 MiB, avg 28.3 %, max 28.5 %

Swap used: avg 406 MiB, max 406 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 65.8 | 100.0 |
| 1 | 14.9 | 60.5 |
| 2 | 5.7 | 10.7 |
| 3 | 39.9 | 97.7 |

### ctx_000512.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 28.0 %, max 29.1 %

RAM used: avg 4699 MiB, max 4767 MiB, avg 29.6 %, max 30.0 %

Swap used: avg 406 MiB, max 406 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 81.0 | 92.6 |
| 1 | 4.7 | 11.7 |
| 2 | 8.2 | 12.0 |
| 3 | 18.0 | 58.4 |

Decode stage:

CPU total: avg 31.8 %, max 38.2 %

RAM used: avg 4752 MiB, max 4863 MiB, avg 29.9 %, max 30.6 %

Swap used: avg 406 MiB, max 406 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 71.5 | 100.0 |
| 1 | 13.0 | 51.3 |
| 2 | 8.0 | 11.3 |
| 3 | 34.8 | 96.8 |

### ctx_001024.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 29.5 %, max 43.8 %

RAM used: avg 4997 MiB, max 5079 MiB, avg 31.5 %, max 32.0 %

Swap used: avg 406 MiB, max 406 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 49.7 | 93.1 |
| 1 | 27.7 | 93.9 |
| 2 | 26.8 | 93.0 |
| 3 | 13.9 | 96.8 |

Decode stage:

CPU total: avg 29.1 %, max 30.6 %

RAM used: avg 5109 MiB, max 5155 MiB, avg 32.2 %, max 32.5 %

Swap used: avg 406 MiB, max 406 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 97.6 | 100.0 |
| 1 | 6.3 | 11.6 |
| 2 | 5.2 | 9.3 |
| 3 | 7.2 | 16.4 |

### ctx_002048.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 30.1 %, max 42.3 %

RAM used: avg 5313 MiB, max 5445 MiB, avg 33.5 %, max 34.3 %

Swap used: avg 406 MiB, max 406 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 53.9 | 100.0 |
| 1 | 15.6 | 92.6 |
| 2 | 13.6 | 79.2 |
| 3 | 37.4 | 97.2 |

Decode stage:

CPU total: avg 31.8 %, max 42.8 %

RAM used: avg 5512 MiB, max 5566 MiB, avg 34.7 %, max 35.1 %

Swap used: avg 406 MiB, max 406 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 73.1 | 100.0 |
| 1 | 12.8 | 53.3 |
| 2 | 5.4 | 8.4 |
| 3 | 35.3 | 98.6 |

### ctx_004096.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 31.3 %, max 44.0 %

RAM used: avg 5726 MiB, max 5883 MiB, avg 36.1 %, max 37.1 %

Swap used: avg 406 MiB, max 406 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 14.6 | 95.9 |
| 1 | 57.7 | 100.0 |
| 2 | 11.6 | 95.4 |
| 3 | 41.0 | 100.0 |

Decode stage:

CPU total: avg 29.1 %, max 31.8 %

RAM used: avg 5945 MiB, max 5977 MiB, avg 37.4 %, max 37.6 %

Swap used: avg 406 MiB, max 406 MiB, avg 1.7 %, max 1.7 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 9.3 | 16.4 |
| 1 | 55.3 | 100.0 |
| 2 | 5.4 | 7.9 |
| 3 | 46.2 | 97.7 |

### ctx_008192.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 32.0 %, max 47.8 %

RAM used: avg 5583 MiB, max 6294 MiB, avg 35.2 %, max 39.6 %

Swap used: avg 405 MiB, max 405 MiB, avg 1.6 %, max 1.6 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 19.7 | 100.0 |
| 1 | 52.7 | 100.0 |
| 2 | 10.0 | 100.0 |
| 3 | 45.5 | 100.0 |

Decode stage:

CPU total: avg 31.8 %, max 43.5 %

RAM used: avg 5789 MiB, max 5867 MiB, avg 36.5 %, max 37.0 %

Swap used: avg 405 MiB, max 405 MiB, avg 1.6 %, max 1.6 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 53.4 | 100.0 |
| 1 | 58.6 | 100.0 |
| 2 | 7.3 | 14.4 |
| 3 | 7.5 | 13.4 |

### ctx_016384.txt | active_slots=1 | request=1

Prefill stage:

CPU total: avg 31.3 %, max 47.0 %

RAM used: avg 6204 MiB, max 6412 MiB, avg 39.1 %, max 40.4 %

Swap used: avg 391 MiB, max 391 MiB, avg 1.6 %, max 1.6 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 23.7 | 100.0 |
| 1 | 46.5 | 100.0 |
| 2 | 42.3 | 100.0 |
| 3 | 12.6 | 100.0 |

Decode stage:

CPU total: avg 31.3 %, max 42.0 %

RAM used: avg 6406 MiB, max 6464 MiB, avg 40.4 %, max 40.7 %

Swap used: avg 391 MiB, max 391 MiB, avg 1.6 %, max 1.6 %

| Logical CPU | avg util % | max util % |
|---:|---:|---:|
| 0 | 7.8 | 12.1 |
| 1 | 66.0 | 100.0 |
| 2 | 29.8 | 99.1 |
| 3 | 21.3 | 98.1 |

