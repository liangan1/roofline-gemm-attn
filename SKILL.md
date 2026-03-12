---
name: roofline-gemm-attn
description: 'Roofline analysis for GEMM/Attention. Use when: classify an operator as compute bound vs L1 bound vs LSC bound vs memory bound; estimate FLOPs/bytes from shapes; interpret VTune/oneAPI profiler bytes/time for Intel GPU (SYCL/XPU). Keywords: roofline, arithmetic intensity, operational intensity, GEMM, attention, flash attention, L1, LSC, bandwidth bound, compute bound.'
argument-hint: 'Examples: /roofline-gemm-attn gemm M=4096 N=4096 K=4096 dtype=bf16 peak_tflops=... bw_mem_gbs=... | /roofline-gemm-attn attn B=1 H=32 S=4096 D=128 dtype=bf16 ...'
user-invocable: true
---

# Roofline (GEMM / Attention)

## Goal
Given an operator + shape (+ optional profiler counters), produce a high-level roofline conclusion:
- `compute bound`
- `L1 bound`
- `LSC bound`
- `memory bound` (HBM/DRAM)

This skill is designed for Intel GPU (SYCL/XPU) workflows, but it works for any device if you provide peak FLOPs and bandwidth roofs.

## When to Use
- You have a GEMM / Attention-like kernel and want a quick bottleneck label.
- You want to decide whether to optimize math/ISA (compute) or data movement (L1/LSC/DRAM).
- You have profiler outputs (time + bytes at one or more hierarchy levels) and want to translate them into a roofline statement.

## Important Assumptions (read first)
- **To distinguish `L1` vs `LSC` vs `memory` bound**, you need *per-level traffic* (bytes) or a defensible model of reuse.
  - If you only provide shape, the script can reliably classify **compute-bound vs global-memory-bound** using minimal DRAM bytes.
  - `L1/LSC bound` requires either (a) profiler bytes per level, or (b) a kernel tiling/reuse model.
- `LSC` here is treated as the cache/load-store path *between L1 and DRAM* (often approximated as L2 / LSC BW roof). Exact mapping depends on platform.

## Procedure
1. Collect inputs:
   - Operator: `gemm` or `attn`
   - Shape + dtype
   - Peak compute: `peak_tflops` (for that dtype)
   - Bandwidth roofs (GB/s): at least `bw_mem_gbs`; optionally `bw_l1_gbs`, `bw_lsc_gbs`
   - Optional measured data:
     - Kernel time `time_ms`
     - Measured bytes at levels: `bytes_l1`, `bytes_lsc`, `bytes_mem`
2. Run the calculator script:
   - Script: [scripts/roofline.py](./scripts/roofline.py)
3. Use the result:
   - If `compute bound`: focus on math pipeline, instruction selection, occupancy, vectorization, or tensor cores/XMX utilization.
   - If `memory bound`: focus on DRAM traffic, fusion, layout, prefetching, and reducing reloads.
   - If `L1/LSC bound`: focus on cache line utilization, coalescing, reuse (tiling), and avoiding thrash.

## Script Usage
### Hardware presets (from your sheet)
- Raw multi-platform sheet paste (for traceability): [references/hardware_sheet_raw_paste.txt](./references/hardware_sheet_raw_paste.txt)
- Script-consumable presets (extensible): [references/hardware_presets.json](./references/hardware_presets.json)
- `hardware_presets.json` is the standard format. Each preset preserves **all provided HW fields** under `fields`, using the original field names from the sheet.
- The script consumes the `roofline` subset (`bw_mem_gbs`, `bw_lsc_gbs`, `bw_l1_gbs`, `peak_tflops_by_dtype`, `frequency_mhz`).
- Cache-level roofs (`bw_l1_gbs`, `bw_lsc_gbs`) are read-only roofs.
- If a preset does not explicitly provide `bw_l1_gbs` / `bw_lsc_gbs`, the script can derive them from the preserved HW fields when `Frequency (MHz)` and the relevant per-clock fields are available.
- Unit convention for derived cache roofs:
  - the raw sheet formulas are written as `bytes_per_clock * frequency / 1e6`
  - in the script, `frequency` is stored as `MHz`, so the equivalent implementation for `GB/s` is `bytes_per_clock * frequency_mhz / 1e3`
- If a preset provides `peak_tflops_by_dtype`, you can omit `--peak-tflops` for those dtypes.
- Old flat preset formats are not supported.
- You can select a platform at runtime via `--preset <name>`.
- For example, `bmg580` now includes `Frequency (MHz)=2850` and `FP16/BF16 theoretical Peak FLOPS = 117`, so `--preset bmg580 --dtype bf16` can fill both compute and bandwidth roofs automatically.

### GEMM (theoretical)
- Minimal DRAM traffic model (reads A,B once; writes C once):
  - FLOPs: `2*M*N*K`
  - Bytes_mem: `(M*K + K*N + M*N) * bytes_per_elem`

Example:
- `python3 .github/skills/roofline-gemm-attn/scripts/roofline.py gemm --m 4096 --n 4096 --k 4096 --dtype bf16 --peak-tflops 200 --bw-mem-gbs 3000`

Example (BMG580 preset provides BF16 peak and bandwidth roofs):
- `python3 .github/skills/roofline-gemm-attn/scripts/roofline.py gemm --m 4096 --n 4096 --k 4096 --dtype bf16 --preset bmg580`

### Attention (theoretical, fused)
- Approx model (fused, no materialized SxS):
  - FLOPs: `B*H*(4*S*S*D)`  (QK^T and PV)
  - Bytes_mem: read Q,K,V and write O: `(3 + 1)*B*H*S*D*bytes_per_elem`

Example:
- `python3 .github/skills/roofline-gemm-attn/scripts/roofline.py attn --b 1 --h 32 --s 4096 --d 128 --dtype bf16 --peak-tflops 200 --bw-mem-gbs 3000`

### With profiler numbers (recommended for L1/LSC attribution)
Example:
- `python3 .github/skills/roofline-gemm-attn/scripts/roofline.py gemm --m 4096 --n 4096 --k 4096 --dtype bf16 --peak-tflops 200 --bw-l1-gbs 20000 --bw-lsc-gbs 8000 --bw-mem-gbs 3000 --time-ms 1.23 --bytes-l1 1.0e12 --bytes-lsc 4.0e11 --bytes-mem 2.0e11`

## Output Interpretation
The script prints:
- Estimated FLOPs
- Bytes and arithmetic intensity (AI) per provided level
- Roofline thresholds (e.g., `peak_tflops / bw_mem_gbs`)
- A bound label and the reasoning

## Notes for agents
- If the user asks “L1/LSC bound?” but provides no per-level bytes/time, first answer compute-vs-DRAM using shape; then ask for the minimum extra info: `time_ms` and either (a) bytes counters or (b) the kernel’s tiling/reuse assumptions.
