# L1-Bound Judgement And Tuning

This note is for questions like:
- Is this GEMM configuration L1 bound?
- From the subgroup view, is this tiledMMA shape too L1-heavy?
- If it is L1 bound, what tile sizes should I try first?

It extends roofline analysis with a subgroup-level screening model.

## What To Ask The User First

If the user asks whether a GEMM configuration is `L1 bound`, first try to obtain:
- tiledMMA shape `(M_tile, N_tile, K_tile)`
- subgroup layout (especially how many subgroups split along `M`)
- dtype
- target platform / preset

If tiledMMA shape is not explicitly available, ask the user to provide one of:
- the tiledMMA configuration directly
- source code that defines the tiledMMA / MMA atom shapes
- a kernel config snippet showing tile shapes and subgroup layout

If the output tile shape differs from the tiledMMA shape, treat that as second-level tiling (`Block -> Tile`) and still perform L1-bound judgement from the tiledMMA shape plus subgroup layout.

## Most Important Distinction

Do **not** confuse these two quantities:

1. Platform balance point:

$$
AI_{compute/L1} = \frac{Peak\ Compute}{BW_{L1}}
$$

2. Subgroup-level arithmetic intensity:

$$
AI_{L1,sg}
$$

The workflow is:
- first compute `AI_{L1,sg}` from tiledMMA shape, subgroup layout, and dtype
- then compare it against `AI_{compute/L1}`

## Subgroup-Level Model

Assume a tiledMMA shape:

$$
(M_{tile}, N_{tile}, K_{tile})
$$

and subgroup split `S_M` along `M`.

Then:

$$
M_{sg} = M_{tile} / S_M
$$

### FLOPs per subgroup

$$
FLOPs_{sg} = 2 M_{sg} N_{tile} K_{tile}
$$

### Read bytes per subgroup

For read-only cache-level roofline:

$$
Bytes_{L1,sg} = (M_{sg}K_{tile} + K_{tile}N_{tile}) \times bytes\_per\_elem
$$

### Subgroup AI

$$
AI_{L1,sg} = \frac{2 M_{sg} N_{tile} K_{tile}}{(M_{sg}K_{tile} + K_{tile}N_{tile}) \times bytes\_per\_elem}
$$

For `bf16/fp16` (`bytes_per_elem = 2`), this simplifies to:

$$
AI_{L1,sg} = \frac{M_{sg}N_{tile}}{M_{sg} + N_{tile}}
$$

## Judgement Rule

- If `AI_{L1,sg} >> AI_{compute/L1}`: not statically L1 bound
- If `AI_{L1,sg} ~= AI_{compute/L1}`: borderline / L1-sensitive
- If `AI_{L1,sg} < AI_{compute/L1}`: statically L1 bound

## Structural Shortcut

For `bf16/fp16`, as `N_tile -> infinity`:

$$
AI_{L1,sg} \to M_{sg}
$$

So a necessary condition to escape L1-bound behavior is:

$$
M_{sg} > AI_{compute/L1}
$$

If `M_{sg}` itself is too small, increasing `N_tile` alone cannot fundamentally fix the tile.

## Register-Pressure Proxy

When comparing candidate tiles, use a simple accumulator proxy:

$$
R_{acc} \propto M_{sg} \times N_{tile}
$$

Among candidates that are no longer L1 bound, prefer smaller `M_{sg} * N_{tile}` unless there are stronger MMA atom constraints.

## Tuning Suggestions When L1 Bound

### 1. Increase `M_{sg}` first

This is usually the highest-leverage knob.

Ways:
- increase `M_tile`
- reduce the number of subgroups split across `M`

### 2. Then choose the smallest `N_tile` that crosses the threshold

For fixed `M_{sg}` and target `B = AI_{compute/L1}`:

$$
\frac{M_{sg}N}{M_{sg}+N} > B
$$

which gives:

$$
N > \frac{B M_{sg}}{M_{sg} - B}
$$

This only works if `M_{sg} > B`.

### 3. Do not expect `K_tile` alone to fix L1-bound behavior

In the simplified subgroup read-only model, `K_tile` cancels out of `AI_{L1,sg}`.

`K_tile` still matters for scheduling, prefetch, and GRF pressure, but it is not the first knob for escaping static L1-bound behavior.

## Suggested Answer Structure

1. State assumptions: tiledMMA shape, subgroup split, dtype, platform preset.
2. Derive subgroup tile.
3. Compute subgroup FLOPs and subgroup read bytes.
4. Compute `AI_{L1,sg}`.
5. Compute or state `AI_{compute/L1}`.
6. Compare the two.
7. Classify the tile.
8. If L1 bound, provide 2-4 candidate sizes ranked by low `M_{sg} * N_{tile}`.
9. End by noting that profiler `bytes_l1` is still required for confirmation.