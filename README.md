# roofline-gemm-attn

This is a user guide for the `roofline-gemm-attn` skill.

## Quick Install In VS Code

To use this skill in another local VS Code project, copy the entire folder:

```text
roofline-gemm-attn/
```

into the target workspace at:

```text
.github/skills/
```

so the final path becomes:

```text
.github/skills/roofline-gemm-attn/
```

Then reopen or reload the workspace in VS Code so Copilot / Agent can rescan the skill directory.

Important:
- copy the whole folder, not just `SKILL.md`
- keep the folder name `roofline-gemm-attn` unchanged
- keep the referenced `scripts/` and `references/` files together with the skill

Use this skill when you want to:
- classify a GEMM or Attention kernel as `compute bound`, `L1 bound`, `LLC bound`, or `memory bound`
- estimate arithmetic intensity from shape and dtype
- reason about tiled GEMM design using `Global -> Block -> Tile -> MMA Atom`
- judge whether a tiledMMA configuration is statically `L1 bound`
- judge whether a work-group / block configuration is statically `LLC bound`
- get tuning suggestions when a tile is too L1-heavy

## Files

- `SKILL.md`: skill definition
- `scripts/roofline.py`: roofline calculator
- `references/hardware_presets.json`: hardware presets
- `references/hardware_sheet_raw_paste.txt`: raw hardware notes
- `references/gemm_tiling_design.md`: three-level tiling notes
- `references/l1_bound_judgement_and_tuning.md`: subgroup-level L1-bound logic and tuning rules
- `references/llc_bound_judgement_and_tuning.md`: block-level LLC-bound logic and tuning rules
- `references/dram_bound_judgement_and_tuning.md`: operator-level DRAM-bound logic and tuning rules

## What You Should Provide

### For full-operator roofline
- operator type: `gemm` or `attn`
- shape
- dtype
- hardware preset, or explicit roofs
- optional profiler data:
  - `time_ms`
  - `bytes_l1`
  - `bytes_llc`
  - `bytes_mem`

### For L1-bound tile analysis
- tiledMMA shape `(M_tile, N_tile, K_tile)`
- subgroup layout, especially how subgroups split along `M`
- dtype
- target platform preset

If tiledMMA shape is unknown, provide the kernel source or config snippet that defines it.

### For LLC-bound tile analysis
- work-group / block tile shape `(M_wg, N_wg, K_wg)`
- dtype
- target platform preset
- whether the kernel is a plain GEMM or a fused kernel with loop-carried reuse
- if fused, which operands are loaded once versus streamed each outer-loop step

If block tile shape is unknown, provide the kernel source or config snippet that defines it.

### For DRAM-bound analysis
- full operator shape
- dtype
- target platform preset
- whether the kernel is plain or fused
- whether profiler `bytes_mem` is available
- if not, the schedule or reuse assumptions needed to estimate memory-side traffic

## Basic Usage

### GEMM roofline

```bash
python3 .github/skills/roofline-gemm-attn/scripts/roofline.py gemm --m 4096 --n 4096 --k 4096 --dtype bf16 --preset bmg580
```

### Attention roofline

```bash
python3 .github/skills/roofline-gemm-attn/scripts/roofline.py attn --b 1 --h 32 --s 4096 --d 128 --dtype bf16 --preset bmg580
```

### Override compute roof directly

```bash
python3 .github/skills/roofline-gemm-attn/scripts/roofline.py gemm --m 4096 --n 4096 --k 4096 --dtype bf16 --peak-tflops 100 --bw-mem-gbs 456
```

### Override frequency

```bash
python3 .github/skills/roofline-gemm-attn/scripts/roofline.py gemm --m 4096 --n 4096 --k 4096 --dtype bf16 --preset bmg580 --frequency-mhz 2000
```

## Priority Rules

For compute roof selection, the skill uses this priority:

1. user-provided `peak_tflops`
2. user-provided `frequency_mhz`, which scales preset peak when possible
3. preset default `peak_tflops_by_dtype`

Preset values are defaults, not hard overrides.

## How L1-Bound Judgement Works

When judging whether a tile is `L1 bound`, the skill does not use the final block output shape directly.

Instead, it uses:
- tiledMMA shape
- subgroup layout
- dtype
- platform compute/L1 balance point

The workflow is:

1. derive the subgroup-level tile
2. compute subgroup FLOPs
3. compute subgroup L1 read bytes from `A` and `B`
4. compute subgroup-level arithmetic intensity `AI_L1_sg`
5. compare `AI_L1_sg` against the platform balance point `Peak / BW_L1`

If output tile shape and tiledMMA shape differ, that means second-level tiling (`Block -> Tile`), but the L1-bound judgement still uses the tiledMMA shape.

## How LLC-Bound Judgement Works

When judging whether a tile is `LLC bound`, the skill does not use subgroup tile shape as the primary unit.

Instead, it uses:
- work-group / block tile shape
- dtype
- platform compute/LLC balance point
- the kernel's actual outer-loop dataflow

The workflow is:

1. derive the block-level work unit
2. compute block FLOPs for the relevant outer-loop step
3. compute client-side `LLC` bytes for that step
4. compute block-level arithmetic intensity `AI_LLC_wg`
5. compare `AI_LLC_wg` against the platform balance point `Peak / BW_LLC`

For fused kernels, the first step and steady state may differ, so the skill may report both.

## How DRAM-Bound Judgement Works

When judging whether an operator is `DRAM bound`, the skill uses the full operator and memory-side traffic as the primary unit.

Instead, it uses:
- full operator shape
- dtype
- platform compute/memory balance point
- either measured `bytes_mem` or a lower-bound / schedule-aware memory model

The workflow is:

1. compute operator FLOPs
2. compute or estimate memory-side bytes `Bytes_mem`
3. compute operator arithmetic intensity `AI_mem`
4. compare `AI_mem` against the platform balance point `Peak / BW_mem`

If only shape is available, the skill first uses a best-case lower-bound memory model and says so explicitly.

## Tuning Guidance When A Tile Is L1 Bound

Use this order:

1. increase `M_sg` first
2. then choose the smallest `N_tile` that crosses the L1 threshold
3. do not expect `K_tile` alone to fix static L1-bound behavior

Candidate tiles should be ranked by a simple accumulator-pressure proxy:

$$
M_{sg} \times N_{tile}
$$

Among tiles that are no longer L1 bound, smaller is preferred unless stronger MMA or layout constraints dominate.

## Tuning Guidance When A Tile Is LLC Bound

Use this order:

1. increase block-level reuse first
2. improve block scheduling and locality across neighboring work-groups
3. use `K_wg` as a secondary knob for pipeline and residency tuning, not as the first fix for static LLC pressure

## Tuning Guidance When An Operator Is DRAM Bound

Use this order:

1. reduce true memory traffic first
2. improve inter-block reuse and traversal locality
3. treat block and subgroup tiling as secondary levers unless they materially change memory traffic

## Unified Tuning Checklist

When tuning after a roofline diagnosis, use this order:

1. identify the active level: `L1`, `LLC`, or `DRAM`
2. choose the matching control unit:
  - `L1`: tiledMMA / subgroup tile
  - `LLC`: block / work-group tile
  - `DRAM`: full operator and schedule / reuse model
3. state the main tradeoff before suggesting a larger tile:
  - larger tiledMMA shapes may improve `L1` arithmetic intensity but increase GRF pressure and may reduce occupancy
  - larger block tiles may improve `LLC` arithmetic intensity but increase outer-cache working set, may reduce hit rate, and may reduce resident blocks
  - `DRAM` improvements should be justified by lower memory-side bytes, not just by larger tiles
4. prefer the smallest change that crosses the relevant roofline threshold while preserving occupancy or cache residency
5. confirm the hypothesis with the matching profiler bytes counter whenever possible

## Output Interpretation

The calculator prints:
- estimated FLOPs
- estimated bytes
- arithmetic intensity
- roof ceilings
- a high-level bound label

For `L1` / `LLC` / `memory` attribution, profiler bytes are still preferred.

## Notes

- Cache-level roofs are modeled as read-only roofs.
- Intel pre-Xe4 style reasoning is described in terms of L1-oriented staging and prefetch, not CUDA-style `cp.async` assumptions.
- Static tile-level judgement is a screening tool, not a final proof. Profiler validation is still recommended.