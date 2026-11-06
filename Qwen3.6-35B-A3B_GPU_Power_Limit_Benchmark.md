# GPU Power Limit vs. Qwen3.6-35B-A3B Throughput: 6 × NVIDIA P104-100

**Benchmark date:** October 18, 2026  
**Data source:** `Qwen3.6_35B_A3B_MTP_full_report_2026-10-19.md`, sections 47–48.  
**Comparison:** identical model configuration at **90 W** and **215 W power limits per GPU**.

## Summary

| Metric | Average gain at 90 → 215 W |
|---|---:|
| Decode — 10 context points | **+4.55%** |
| Decode — excluding 7,141 tokens | **+5.14%** |
| Prefill — 9 comparable points | **+8.39%** |
| Increase in configured power limit | **+138.9%** |

The reported percentage averages are **from the original report** and represent the arithmetic mean of per-point relative gains. They are not the improvement in arithmetic mean throughput or end-to-end throughput for one continuous request.

## Test configuration

| Parameter | Value |
|---|---|
| OS | Ubuntu 24.04 |
| GPUs | 6 × NVIDIA P104-100 (GP104, 8 GB) |
| GPU selection | `CUDA_VISIBLE_DEVICES=6,7,8,9,10,11` |
| Model split | `-sm layer` |
| Target model | `Qwen3.6-35B-A3B-abliterated-v4`, `Q8_0` (34.66B parameters) |
| Target filename | `Qwen3.6-35B-A3B-abliterated-v4-Q8_0-nomtp.gguf` |
| MTP draft model | `Q8_0`, `n_max=3` |
| Inference backend | `llama.cpp` build `11429` with TurboPrefill |
| Commit | `d81235049384534c167caea52b85a694f6103d14` |
| Power limits | 90 W and 215 W **per GPU** |
| Context length | 7,141–251,282 tokens |

Target model, MTP quantization, `n_max`, and benchmark context points were kept unchanged. The six active GPUs had their power limits raised from 90 W to 215 W.

## Decode throughput

| Context (tokens) | 90 W (tok/s) | 215 W (tok/s) | Gain |
|---:|---:|---:|---:|
| 7,141 | 48.448 | 48.074 | -0.8% |
| 22,400 | 45.020 | 45.794 | +1.7% |
| 37,659 | 41.750 | 43.433 | +4.0% |
| 68,176 | 36.440 | 38.230 | +4.9% |
| 98,694 | 32.230 | 33.298 | +3.3% |
| 129,211 | 28.860 | 30.788 | +6.7% |
| 159,729 | 26.080 | 28.040 | +7.5% |
| 190,247 | 23.690 | 25.062 | +5.8% |
| 220,764 | 21.990 | 23.689 | +7.7% |
| 251,282 | 20.280 | 21.201 | +4.5% |

**Original report result:** +4.55% average over 10 points, or +5.14% when excluding 7,141 tokens. The small negative difference at 7,141 tokens is consistent with measurement variability; no repeated-run uncertainty estimate is available.

## Prefill throughput

| Context (tokens) | 90 W (tok/s) | 215 W (tok/s) | Gain |
|---:|---:|---:|---:|
| 22,400 | 539.130 | 595.250 | +10.4% |
| 37,659 | 359.420 | 396.440 | +10.3% |
| 68,176 | 210.570 | 228.360 | +8.4% |
| 98,694 | 132.810 | 142.130 | +7.0% |
| 129,211 | 96.320 | 102.850 | +6.8% |
| 159,729 | 75.480 | 80.760 | +7.0% |
| 190,247 | 59.910 | 66.010 | +10.2% |
| 220,764 | 51.330 | 55.230 | +7.6% |
| 251,282 | 44.960 | 48.450 | +7.8% |

**Original report result:** +8.39% average over the 9 comparable points.

**Why the 7,141-token point was excluded:** the second run resumed from a checkpoint and actually processed only **4 tokens**, making its reported prefill rate unsuitable for comparison with the original run.

### Benchmark methodology caveat

The long-context sweep ran sequentially with state/checkpoint reuse. Therefore, prefill throughput here measures **processing a newly added prompt segment with an existing context**, not a cold prefill of the full nominal context. From approximately 68k tokens onward, each point involved about 30.4k newly processed tokens.

## Calculation

Relative throughput gain per context point:

```text
Gain (%) = (throughput_215W / throughput_90W - 1) × 100
```

For example, decode at 251,282 tokens:

```text
(21.201 / 20.280 - 1) × 100 = +4.54%
```

Comparison basis: **one context point, one inference mode, one six-GPU configuration**. All throughput values are tokens per second, not tokens per watt.

## Limitations and conclusion

1. This benchmark tests the effect of a **configured GPU power limit**, not measured actual board power. A 215 W limit does not imply a sustained 215 W draw.
2. Actual power samples (`nvidia-smi`, `system_samples.csv`) from both runs are required to calculate tokens per joule.
3. No repeated-run confidence intervals are included here.
4. Findings are specific to **Qwen3.6-35B-A3B, Q8 MTP n_max=3**, the sequential long-context workload and this `llama.cpp`/TurboPrefill build. They should not be generalized to other models, GPUs, MTP modes, or real-world agent workloads without additional measurements.
5. Raising the configured limit from 90 W to 215 W yielded only about **5% faster decode** and **8% faster prefill** on average. These measurements alone do not establish which subsystem limits scaling.

## Data provenance

This document is derived from sections **47 (“Повтор long-grid при PL=215 W”)** and **48 (“Prefill: 90 W vs 215 W”)** of `Qwen3.6_35B_A3B_MTP_full_report_2026-10-19.md` (report dated October 19, 2026). Original throughput measurements were preserved.
