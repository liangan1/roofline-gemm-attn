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
- classify a GEMM or Attention kernel as `compute bound`, `L1 bound`, `LSC bound`, or `memory bound`
- estimate arithmetic intensity from shape and dtype
- reason about tiled GEMM design using `Global -> Block -> Tile -> MMA Atom`
- judge whether a tiledMMA configuration is statically `L1 bound`
- get tuning suggestions when a tile is too L1-heavy

## Files

- `SKILL.md`: skill definition
- `scripts/roofline.py`: roofline calculator
- `references/hardware_presets.json`: hardware presets
- `references/hardware_sheet_raw_paste.txt`: raw hardware notes
- `references/gemm_tiling_design.md`: three-level tiling notes
- `references/l1_bound_judgement_and_tuning.md`: subgroup-level L1-bound logic and tuning rules

## What You Should Provide

### For full-operator roofline
- operator type: `gemm` or `attn`
- shape
- dtype
- hardware preset, or explicit roofs
- optional profiler data:
  - `time_ms`
  - `bytes_l1`
  - `bytes_lsc`
  - `bytes_mem`

### For L1-bound tile analysis
- tiledMMA shape `(M_tile, N_tile, K_tile)`
- subgroup layout, especially how subgroups split along `M`
- dtype
- target platform preset

If tiledMMA shape is unknown, provide the kernel source or config snippet that defines it.

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

## Output Interpretation

The calculator prints:
- estimated FLOPs
- estimated bytes
- arithmetic intensity
- roof ceilings
- a high-level bound label

For `L1` / `LSC` / `memory` attribution, profiler bytes are still preferred.

## Notes

- Cache-level roofs are modeled as read-only roofs.
- Intel pre-Xe4 style reasoning is described in terms of L1-oriented staging and prefetch, not CUDA-style `cp.async` assumptions.
- Static tile-level judgement is a screening tool, not a final proof. Profiler validation is still recommended.