# DRAM-Bound Judgement And Tuning

This note is for questions like:
- Is this GEMM configuration `DRAM bound`?
- Is the kernel fundamentally limited by memory bandwidth rather than `LLC` or `L1`?
- If it is `DRAM bound`, which knobs should I try first?

It extends roofline analysis with an operator-level and schedule-aware memory model.

## What To Ask The User First

If the user asks whether a GEMM or attention configuration is `DRAM bound`, first try to obtain:
- full operator shape
- dtype
- target platform / preset
- whether the kernel is plain or fused
- whether neighboring blocks can reuse operands through outer cache
- whether the kernel has persistent operands or loop-carried reuse

If the user has profiler data, ask for:
- `time_ms`
- `bytes_mem`

If `bytes_mem` is unavailable, ask for the minimum extra structure needed to estimate it:
- block shape and traversal order
- which operands are streamed, reused, or materialized
- whether the kernel fuses intermediate tensors

## Most Important Distinctions

Do **not** confuse these quantities:

1. Platform balance point:

$$
AI_{compute/mem} = \frac{Peak\ Compute}{BW_{mem}}
$$

2. Client-side outer-cache traffic:

- bytes between `LLC` and kernel clients
- relevant for `LLC bound`

3. DRAM traffic:

- bytes between memory and the outer cache hierarchy
- relevant for `memory bound`

For `DRAM bound` judgement in this skill, use **memory-side bytes**:
- cache refills from DRAM into outer cache
- writebacks / evictions to memory
- not the larger client-side `LLC` traffic seen by work-groups

The workflow is:
- first estimate or measure `bytes_mem`
- then compute `AI_mem`
- then compare it against `AI_{compute/mem}`

## Why DRAM Is An Operator-Level Question

In the three-level tiling view:
- `Global -> Block` determines how much inter-block reuse can be converted into cache hits
- `Block -> Tile` determines how much block-side reuse reduces `LLC` pressure
- `Tile -> MMA Atom` determines `L1` pressure

So:
- `L1 bound` is usually screened at subgroup level
- `LLC bound` is usually screened at block level
- `DRAM bound` is usually screened at operator level, then refined with schedule-aware reuse assumptions

`DRAM bound` is less about the tile in isolation and more about whether the overall schedule successfully avoids refetching the same data from memory.

## Plain GEMM Minimal-Memory Model

For a plain GEMM:

$$
C = A_{M \times K} B_{K \times N}
$$

the minimal memory traffic lower bound is:

$$
Bytes_{mem,min} = (MK + KN + MN) \times bytes\_per\_elem
$$

and the operator FLOPs are:

$$
FLOPs = 2MNK
$$

So the minimal-memory arithmetic intensity is:

$$
AI_{mem,min} = \frac{2MNK}{(MK + KN + MN) \times bytes\_per\_elem}
$$

This is a **best-case lower bound on DRAM bytes**, not a guarantee.

## What The Minimal-Memory Model Means

- If `AI_{mem,min} < AI_{compute/mem}`, the operator is structurally vulnerable to `DRAM bound` behavior.
- If `AI_{mem,min} >> AI_{compute/mem}`, the operator is not intrinsically `DRAM bound`; if profiling still shows `memory bound`, then schedule, cache, fusion, or layout problems are likely inflating actual memory traffic.

This is the right first screening tool when the user only provides shape.

## Schedule-Aware Memory Model

For real tiled kernels, actual DRAM traffic may be larger than `Bytes_{mem,min}`.

Common reasons:
- neighboring blocks do not hit in outer cache
- traversal order destroys operand reuse
- persistent scheduling is absent or ineffective
- paged or varlen layouts reduce locality
- intermediate tensors are materialized instead of fused

In those cases:

$$
AI_{mem,actual} = \frac{FLOPs}{Bytes_{mem,actual}}
$$

and:

$$
Bytes_{mem,actual} \ge Bytes_{mem,min}
$$

This is the quantity that should be compared with the memory roof when profiler `bytes_mem` is available.

## Fused / Attention-Style Kernels

For fused kernels, do **not** blindly reuse the plain GEMM formula.

Instead, build the memory-side byte model from actual operator semantics.

Examples:
- fused attention usually avoids materializing the `S \times S` score matrix, which greatly reduces DRAM bytes
- if `Q`, `K`, `V` are each read once and `O` is written once, the first-order fused attention lower bound is:

$$
Bytes_{mem,min} \approx (Q + K + V + O)\ bytes
$$

- if cached or paged KV changes reuse, account for the actual memory-side fetch pattern rather than the client-side loop traffic

For fused kernels, it is often useful to report both:
- ideal lower bound `AI_{mem,min}`
- measured or schedule-aware `AI_{mem,actual}`

## Judgement Rule

- If `AI_{mem} >> AI_{compute/mem}`: not `DRAM bound`
- If `AI_{mem} ~= AI_{compute/mem}`: borderline / memory-sensitive
- If `AI_{mem} < AI_{compute/mem}`: `DRAM bound`

When only shape is available, say explicitly whether you are using:
- `AI_{mem,min}` (best-case lower-bound model)
- or `AI_{mem,actual}` (measured / schedule-aware model)

## Structural Interpretation

If the operator is already `DRAM bound` under the minimal-memory model, improving only subgroup or block tiling is unlikely to remove the bottleneck completely.

Typical fixes then require:
- fusion
- reducing tensor materialization
- changing data layout
- reducing precision or compression
- exploiting cross-operator reuse

If the operator is **not** `DRAM bound` under the minimal-memory model but becomes `DRAM bound` in practice, then the main problem is usually:
- weak outer-cache locality
- poor block scheduling
- insufficient persistence
- extra reloads caused by capacity/conflict misses

## Tuning Suggestions When DRAM Bound

### 1. Reduce true memory traffic first

Usually higher leverage than changing MMA tile shape.

Examples:
- fuse adjacent operators
- avoid materializing intermediates
- reduce redundant reads or writes
- improve tensor layout to reduce wasted line fetches

### 2. Improve inter-block reuse

Look for:
- traversal orders that keep the dominant operand hot in outer cache
- persistent scheduling that reuses tiles before eviction
- block shapes that increase reuse distance efficiency

### 3. Use block and subgroup tiling as secondary levers

These can reduce `LLC` and `L1` pressure and sometimes indirectly help memory traffic, but they are not the first explanation for structural `DRAM bound` behavior.

### 4. Separate lower-bound and actual diagnosis

When possible, report both:
- ideal lower-bound memory bytes
- measured `bytes_mem`

The gap between them often tells you whether the real issue is structural arithmetic intensity or poor locality.

## Tuning Tradeoffs

Changing subgroup or block tiling can influence `DRAM` pressure indirectly, but `DRAM bound` optimization is usually governed by memory traffic first.

### 1. Tiling can help memory traffic indirectly

Better block tiling or scheduling can:
- improve outer-cache hit rate
- reduce refetch from memory
- shorten reuse distance

So tiling is still relevant to `DRAM bound` behavior, just usually not as the first-order explanation.

### 2. Larger block tiles are not automatically better for DRAM

A larger block tile may:
- improve inter-block locality
- or increase working-set pressure and reduce cache effectiveness

So the sign is schedule- and capacity-dependent.

The key question is whether the change reduces actual `bytes_mem`, not whether the tile is simply larger.

### 3. Larger tiledMMA shapes are even more indirect for DRAM

Changing tiledMMA shape mainly targets `L1` behavior.

It may still affect end-to-end memory performance through:
- occupancy changes
- pipeline efficiency changes
- block residency changes

But these are secondary paths compared with true memory-traffic reduction.

### 4. If a workload is structurally `DRAM bound`, bandwidth reduction dominates

In that case, the highest-leverage changes are usually:
- fusion
- avoiding materialization
- layout changes
- compression / lower precision
- schedule changes that measurably reduce `bytes_mem`

Not simply larger tiles at lower levels.

### 5. Practical tuning rule

When evaluating a change for a `DRAM bound` workload, ask first whether it reduces memory-side bytes. If the answer is unclear, treat subgroup and block tiling as secondary hypotheses until profiler `bytes_mem` confirms a real reduction.

## Suggested Answer Structure

1. State assumptions: operator shape, dtype, preset, plain vs fused.
2. Compute the relevant operator FLOPs.
3. Compute `Bytes_{mem,min}` or state measured `bytes_mem`.
4. Compute `AI_{mem}`.
5. Compute or state `AI_{compute/mem}`.
6. Compare the two.
7. Classify the operator.
8. Explain whether the conclusion is structural or schedule-dependent.
9. If `DRAM bound`, provide 2-4 memory-traffic reduction suggestions.
10. End by noting that profiler `bytes_mem` is preferred for confirmation.