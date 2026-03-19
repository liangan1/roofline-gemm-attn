---
name: roofline-gemm-attn
description: 'Roofline analysis for GEMM/Attention. Use when: classify an operator as compute bound vs L1 bound vs LLC bound vs memory bound; estimate FLOPs/bytes from shapes; interpret VTune/oneAPI profiler bytes/time for Intel GPU (SYCL/XPU). Keywords: roofline, arithmetic intensity, operational intensity, GEMM, attention, flash attention, L1, LLC, bandwidth bound, compute bound.'
argument-hint: 'Examples: /roofline-gemm-attn gemm M=4096 N=4096 K=4096 dtype=bf16 peak_tflops=... bw_mem_gbs=... | /roofline-gemm-attn attn B=1 H=32 S=4096 D=128 dtype=bf16 ...'
user-invocable: true
---

# Roofline (GEMM / Attention)

## Goal
Given an operator + shape (+ optional profiler counters), produce a high-level roofline conclusion:
- `compute bound`
- `L1 bound`
- `LLC bound`
- `memory bound` (HBM/DRAM)

This skill is designed for Intel GPU (SYCL/XPU) workflows, but it works for any device if you provide peak FLOPs and bandwidth roofs.

## When to Use
- You have a GEMM / Attention-like kernel and want a quick bottleneck label.
- You want to decide whether to optimize math/ISA (compute) or data movement (L1/LLC/DRAM).
- You have profiler outputs (time + bytes at one or more hierarchy levels) and want to translate them into a roofline statement.

## Important Assumptions (read first)
- **To distinguish `L1` vs `LLC` vs `memory` bound**, you need *per-level traffic* (bytes) or a defensible model of reuse.
  - If you only provide shape, the script can reliably classify **compute-bound vs global-memory-bound** using minimal DRAM bytes.
  - `L1/LLC bound` requires either (a) profiler bytes per level, or (b) a kernel tiling/reuse model.
- `LLC` here is treated as the outer cache path between `L1` and DRAM (often approximated as the L2/LLC bandwidth roof). Exact mapping depends on platform.

## Procedure
1. Collect inputs:
   - Operator: `gemm` or `attn`
   - Shape + dtype
   - Peak compute: `peak_tflops` (for that dtype)
   - Bandwidth roofs (GB/s): at least `bw_mem_gbs`; optionally `bw_l1_gbs`, `bw_llc_gbs`
   - Optional measured data:
     - Kernel time `time_ms`
     - Measured bytes at levels: `bytes_l1`, `bytes_llc`, `bytes_mem`
2. Run the calculator script:
   - Script: [scripts/roofline.py](./scripts/roofline.py)
3. Use the result:
   - If `compute bound`: focus on math pipeline, instruction selection, occupancy, vectorization, or tensor cores/XMX utilization.
   - If `memory bound`: focus on DRAM traffic, fusion, layout, prefetching, and reducing reloads.
  - If `L1/LLC bound`: focus on cache line utilization, coalescing, reuse (tiling), and avoiding thrash.

## Script Usage
### Hardware presets (from your sheet)
- Raw multi-platform sheet paste (for traceability): [references/hardware_sheet_raw_paste.txt](./references/hardware_sheet_raw_paste.txt)
- Script-consumable presets (extensible): [references/hardware_presets.json](./references/hardware_presets.json)
- Algorithm design reference: [references/gemm_tiling_design.md](./references/gemm_tiling_design.md)
- L1-bound judgement and tuning reference: [references/l1_bound_judgement_and_tuning.md](./references/l1_bound_judgement_and_tuning.md)
- LLC-bound judgement and tuning reference: [references/llc_bound_judgement_and_tuning.md](./references/llc_bound_judgement_and_tuning.md)
- DRAM-bound judgement and tuning reference: `references/dram_bound_judgement_and_tuning.md`
- `hardware_presets.json` is the standard format. Each preset preserves **all provided HW fields** under `fields`, using the original field names from the sheet.
- The script consumes the `roofline` subset (`bw_mem_gbs`, `bw_llc_gbs`, `bw_l1_gbs`, `peak_tflops_by_dtype`, `frequency_mhz`).
- User-provided `peak_tflops` or `frequency_mhz` should take precedence over preset defaults. Only when both are absent should the preset default peak be used.
- Cache-level roofs (`bw_l1_gbs`, `bw_llc_gbs`) are read-only roofs.
- If a preset does not explicitly provide `bw_l1_gbs` / `bw_llc_gbs`, the script can derive them from the preserved HW fields when `Frequency (MHz)` and the relevant per-clock fields are available.
- Unit convention for derived cache roofs:
  - the raw sheet formulas are written as `bytes_per_clock * frequency / 1e6`
  - in the script, `frequency` is stored as `MHz`, so the equivalent implementation for `GB/s` is `bytes_per_clock * frequency_mhz / 1e3`
- If a preset provides `peak_tflops_by_dtype`, you can omit `--peak-tflops` for those dtypes.
- If the user provides `frequency_mhz` but not `peak_tflops`, compute roof should be scaled from the preset's default frequency/peak when available.
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

### With profiler numbers (recommended for L1/LLC attribution)
Example:
- `python3 .github/skills/roofline-gemm-attn/scripts/roofline.py gemm --m 4096 --n 4096 --k 4096 --dtype bf16 --peak-tflops 200 --bw-l1-gbs 20000 --bw-llc-gbs 8000 --bw-mem-gbs 3000 --time-ms 1.23 --bytes-l1 1.0e12 --bytes-llc 4.0e11 --bytes-mem 2.0e11`

## Output Interpretation
The script prints:
- Estimated FLOPs
- Bytes and arithmetic intensity (AI) per provided level
- Roofline thresholds (e.g., `peak_tflops / bw_mem_gbs`)
- A bound label and the reasoning

## Notes for agents
- If the user asks “L1/LLC bound?” but provides no per-level bytes/time, first answer compute-vs-DRAM using shape; then ask for the minimum extra info: `time_ms` and either (a) bytes counters or (b) the kernel’s tiling/reuse assumptions.
- When explaining *why* a roof is active, use the GEMM hierarchy in [references/gemm_tiling_design.md](./references/gemm_tiling_design.md): `Global -> Block`, `Block -> Tile`, `Tile -> MMA Atom`.
- For Intel pre-Xe4 platforms, prefer wording that reflects L1-oriented staging/prefetch rather than CUDA-specific `cp.async` assumptions.
- When the user asks whether a GEMM configuration is `L1 bound`, first try to obtain the tiledMMA shape; if it is not given, ask for the tiledMMA config or the relevant source code.
- When judging whether a tile is `L1 bound`, use the subgroup-level screening procedure in [references/l1_bound_judgement_and_tuning.md](./references/l1_bound_judgement_and_tuning.md).
- Do not confuse subgroup-level AI with the platform balance point: first compute subgroup-level AI from tiledMMA shape and subgroup layout, then compare it against `Peak/BW_L1`.
- If the tile appears statically `L1 bound`, provide 2-4 tuning suggestions ranked by low accumulator proxy `M_sg * N_tile`, then mention that profiler `bytes_l1` is still needed for confirmation.
- When the user asks whether a GEMM or attention configuration is `LLC bound`, first try to obtain the block / work-group tile shape; if it is not given, ask for the block tile config or the relevant source code.
- When judging whether a tile is `LLC bound`, use the block-level screening procedure in [references/llc_bound_judgement_and_tuning.md](./references/llc_bound_judgement_and_tuning.md).
- Do not confuse block-level `LLC` client traffic with DRAM-fill traffic: first compute block-level AI from client-side `LLC` bytes, then separately discuss cache hits or DRAM refill reduction.
- For fused kernels with loop-carried reuse, report both `first-step` and `steady-state` `LLC` judgement when the answer could differ.
- If the tile appears statically `LLC bound`, provide 2-4 block-shape or scheduling tuning suggestions, then mention that profiler `bytes_llc` is still needed for confirmation.
- When the user asks whether a GEMM or attention configuration is `DRAM bound`, first try to obtain the full operator shape and whether the kernel is plain or fused.
- When judging whether an operator is `DRAM bound`, use the operator-level screening procedure in `references/dram_bound_judgement_and_tuning.md`.
- Do not confuse `DRAM` bytes with client-side `LLC` bytes: `DRAM bound` should be judged from memory-side refill/writeback traffic, not from `LLC -> client` traffic.
- If the user only provides shape, first compute a best-case lower-bound memory model and say explicitly that it is a lower-bound screen rather than a final measurement.
- If the operator appears structurally `DRAM bound`, prioritize memory-traffic reduction suggestions such as fusion, layout changes, and improved inter-block reuse; mention that profiler `bytes_mem` is preferred for confirmation.

## Unified Tuning Checklist For Agents

When the user asks how to retune a kernel after a `L1` / `LLC` / `DRAM` diagnosis, answer in this order:

1. State the active level being targeted: `L1`, `LLC`, or `DRAM`.
2. State the primary shape or semantic unit:
  - `L1`: tiledMMA / subgroup tile
  - `LLC`: block / work-group tile
  - `DRAM`: full operator plus schedule / reuse model
3. State the primary knob:
  - `L1`: adjust tiledMMA shape, especially `M_sg` then `N_tile`
  - `LLC`: adjust block tiling and block scheduling / locality
  - `DRAM`: reduce true memory traffic first
4. State the main tradeoff explicitly:
  - `L1`: larger tiledMMA shapes may improve `AI_{L1,sg}` but increase GRF pressure and may reduce occupancy
  - `LLC`: larger block tiles may improve `AI_{LLC,wg}` but increase outer-cache working set, may reduce hit rate, and may reduce resident blocks
  - `DRAM`: subgroup or block tiling may help indirectly, but true progress requires lower `bytes_mem`
5. Prefer the smallest change that crosses the relevant roofline threshold while preserving occupancy / residency.
6. End by stating what profiler counter would confirm the hypothesis:
  - `L1`: `bytes_l1`
  - `LLC`: `bytes_llc`
  - `DRAM`: `bytes_mem`

Do not give one-sided advice such as “just increase tile size” without also stating the likely occupancy, register, or cache-residency cost.
