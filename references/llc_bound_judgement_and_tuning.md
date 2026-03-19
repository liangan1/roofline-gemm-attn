# LLC-Bound Judgement And Tuning

This note is for questions like:
- Is this GEMM configuration `LLC bound`?
- From the block/work-group view, is this tile too outer-cache heavy?
- If it is `LLC bound`, which tiling or scheduling knobs should I try first?

It extends roofline analysis with a block-level screening model.

## What To Ask The User First

If the user asks whether a GEMM or attention configuration is `LLC bound`, first try to obtain:
- work-group / block tile shape `(M_wg, N_wg, K_wg)`
- dtype
- target platform / preset
- whether the kernel is a plain GEMM or a fused kernel with loop-carried reuse
- which operands are reused across the outer loop or across neighboring blocks

If the block tile is not explicitly available, ask the user to provide one of:
- the block tile configuration directly
- source code that defines the work-group tile shape
- a kernel config snippet showing CTA / work-group shape and scheduling layout

## Most Important Distinctions

Do **not** confuse these quantities:

1. Platform balance point:

$$
AI_{compute/LLC} = \frac{Peak\ Compute}{BW_{LLC}}
$$

2. Block-level arithmetic intensity:

$$
AI_{LLC,wg}
$$

3. DRAM-fill traffic:

- cache miss refills from memory into outer cache
- affected by inter-block reuse and cache hit rate

For `LLC bound` judgement in this skill, use **client-side LLC traffic**:
- bytes transferred between `LLC` and the kernel clients (work-groups / L1 path/SMEM)
- not reduced DRAM-fill bytes caused by cache hits

The workflow is:
- first compute or estimate `AI_{LLC,wg}` from the block tile and actual dataflow
- then compare it against `AI_{compute/LLC}`
- only after that discuss DRAM-fill reduction or cache-hit effects

## Why LLC Is A Block-Level Question

In the three-level tiling view:
- `Global -> Block` mainly determines `memory` and `LLC` pressure
- `Block -> Tile` mainly determines how much outer-cache traffic is converted into local reuse
- `Tile -> MMA Atom` mainly determines `L1 -> GRF -> XMX` pressure

So:
- `L1 bound` is usually screened at subgroup / tiledMMA level
- `LLC bound` is usually screened at block / work-group level

## Plain GEMM Block-Level Model

Assume a block tile:

$$
(M_{wg}, N_{wg}, K_{wg})
$$

with read-only cache-level traffic.

### FLOPs per block

$$
FLOPs_{wg} = 2 M_{wg} N_{wg} K_{wg}
$$

### Client-side LLC read bytes per block

For a plain GEMM block, the conservative one-pass model is:

$$
Bytes_{LLC,wg} = (M_{wg}K_{wg} + K_{wg}N_{wg}) \times bytes\_per\_elem
$$

### Block-level AI

$$
AI_{LLC,wg} = \frac{2 M_{wg} N_{wg} K_{wg}}{(M_{wg}K_{wg} + K_{wg}N_{wg}) \times bytes\_per\_elem}
$$

For `bf16/fp16` (`bytes_per_elem = 2`), this simplifies to:

$$
AI_{LLC,wg} = \frac{M_{wg}N_{wg}}{M_{wg} + N_{wg}}
$$

Note that in this simple model, `K_wg` cancels out.

## Fused / Loop-Carried Kernels

For fused attention-style kernels or any mainloop with persistent operands, do **not** blindly use the plain GEMM formula.

Instead, compute:

$$
AI_{LLC,wg} = \frac{FLOPs\ per\ outer\ loop\ step}{LLC\ client\ bytes\ per\ outer\ loop\ step}
$$

where `LLC client bytes` should reflect the actual dataflow.

Typical examples:
- If `Q` is loaded once and reused across many `K` blocks, do not charge full `Q` bytes to every steady-state step.
- If `K` and `V` stream every step, they should remain in the per-step byte count.
- If the first step has extra setup traffic, analyze `first-step` and `steady-state` separately.

This distinction is important for flash attention.

## Judgement Rule

- If `AI_{LLC,wg} >> AI_{compute/LLC}`: not statically `LLC bound`
- If `AI_{LLC,wg} ~= AI_{compute/LLC}`: borderline / `LLC`-sensitive
- If `AI_{LLC,wg} < AI_{compute/LLC}`: statically `LLC bound`

## Structural Shortcut

For plain `bf16/fp16` GEMM:

$$
AI_{LLC,wg} = \frac{M_{wg}N_{wg}}{M_{wg} + N_{wg}}
$$

As `N_{wg} \to \infty`:

$$
AI_{LLC,wg} \to M_{wg}
$$

and symmetrically as `M_{wg} \to \infty`:

$$
AI_{LLC,wg} \to N_{wg}
$$

So a necessary condition to escape static `LLC bound` behavior is:

$$
\max(M_{wg}, N_{wg}) > AI_{compute/LLC}
$$

If both block dimensions are small, increasing `K_wg` alone will not fundamentally fix the tile.

## What Usually Causes LLC Pressure

- block tiles are too small, so outer-cache reuse is weak
- the dominant operand is not reused enough across the block
- scheduling destroys cross-block locality
- persistent / paged layouts create extra outer-cache client traffic
- too many work-groups contend for shared outer-cache bandwidth

## Tuning Suggestions When LLC Bound

### 1. Increase block-level reuse first

Usually by increasing the block dimension associated with the bandwidth-dominant operand.

Examples:
- increase `M_wg` if reusing `B/K/V` across more rows is the main opportunity
- increase `N_wg` if reusing `A/Q/P` across more columns is the main opportunity

### 2. Keep `K_wg` as a secondary knob

In the static plain-GEMM model, `K_wg` cancels out of `AI_{LLC,wg}`.

It still matters for:
- prefetch distance
- pipeline efficiency
- register pressure
- cache residency windows

But it is not the first knob for escaping static `LLC bound` behavior.

### 3. Improve block scheduling

This is much more important for `LLC` than for `L1`.

Look for:
- neighboring blocks that reuse the same operand tiles
- traversal orders that preserve locality in the dominant operand
- persistent scheduling strategies that reduce cache thrash

### 4. Separate first-step and steady-state analysis

For fused kernels, a tile may look `LLC`-heavy in the first step but not in steady state.

If a persistent operand is loaded once and reused, report both:
- `AI_{LLC,first}`
- `AI_{LLC,steady}`

## Tuning Tradeoffs

Changing block tiling can improve `LLC` pressure, but it also changes cache behavior and residency.

### 1. Larger block tiles usually improve block-level reuse

If you increase `M_{wg}` or `N_{wg}`, you often improve:
- block-level arithmetic intensity `AI_{LLC,wg}`
- reuse of the bandwidth-dominant operand inside the block

This is the main reason larger block tiles can help in `LLC bound` cases.

### 2. Larger block tiles also increase the outer-cache working set

This can affect:
- cache residency
- reuse distance
- capacity pressure
- conflict behavior

So a larger tile may improve static `AI_{LLC,wg}` while still reducing actual `LLC` hit rate if the working set no longer fits comfortably.

### 3. Larger block tiles can reduce resident blocks

If each work-group consumes more resources, the device may keep fewer blocks resident at once.

This can reduce:
- occupancy
- scheduling flexibility
- latency hiding

So the tradeoff is not just tile reuse versus `LLC` bandwidth, but also tile reuse versus residency.

### 4. `LLC` improvement is not only a tile-shape question

Two block shapes with similar static `AI_{LLC,wg}` can behave differently if:
- traversal order is different
- neighboring blocks reuse different operands
- persistent scheduling changes reuse before eviction

So block scheduling should be considered together with block shape.

### 5. Practical tuning rule

Among candidate block tilings that are no longer statically `LLC bound`, prefer the one that preserves better cache residency and resident-block count unless there is clear evidence that a larger working set still improves net performance.

## Suggested Answer Structure

1. State assumptions: block tile shape, dtype, platform preset, and whether the kernel is plain or fused.
2. State the relevant outer-loop step being analyzed.
3. Compute block-level FLOPs.
4. Compute client-side `LLC` bytes for that step.
5. Compute `AI_{LLC,wg}`.
6. Compute or state `AI_{compute/LLC}`.
7. Compare the two.
8. Classify the tile.
9. If `LLC bound`, provide 2-4 block or scheduling tuning suggestions.
10. End by noting that profiler `bytes_llc` is still required for confirmation.
