# GEMM Tiling Design Notes

This note captures algorithm-design knowledge that complements roofline analysis.
It is intended for reasoning about *why* a GEMM is compute/L1/LSC/memory bound,
not just *which* roof is active.

## Hierarchical View

For an `M x N x K` GEMM, the computation is typically decomposed into three levels:

1. `Global -> Block`
2. `Block -> Tile`
3. `Tile -> MMA Atom`

Each level is coupled to specific GPU hardware behavior.

## 1. Global -> Block

Goal:
- Partition the full GEMM into independent blocks that can run in parallel across SMs / XeCores.

Hardware mapping:
- CUDA: CTA / thread block scheduled to an SM.
- Intel SYCL/XPU: workgroup scheduled to a XeCore-grouping / execution context.

Key consequence:
- Different SMs / XeCores share outer cache resources (`LSC` / `L2`), so block scheduling affects cache reuse and contention.

Performance implications:
- A schedule that improves producer-consumer locality across neighboring blocks may reduce GMEM traffic.
- A schedule that destroys cross-block reuse can push the kernel toward `LSC` or `memory` bound behavior.
- For skinny-M or skinny-N GEMMs, block shape and block traversal order can materially change outer-cache reuse of `A` or `B`.

Questions to ask:
- Which operand is reused across blocks: `A`, `B`, or both?
- Are neighboring blocks likely to hit in shared outer cache?
- Does the work distribution create load imbalance or under-filled XeCore occupancy?

## 2. Block -> Tile

Goal:
- Subdivide a block into tiles that maximize data reuse while minimizing expensive global-memory traffic.

Hardware mapping:
- CUDA (especially A100+): shared memory plus `cp.async` often enables explicit async GMEM -> SMEM staging.
- Intel platforms before Xe4 generally do not expose a CUDA-like `cp.async` engine, so implementations more often rely on `prefetch`-style movement and L1-oriented staging.

Key difference vs CUDA:
- On CUDA, the classic optimization story is often “move data to SMEM early, overlap copy and compute.”
- On Intel pre-Xe4, the optimization story is often closer to “shape tiles and prefetch behavior so L1 residency and reuse are good enough.”

Performance implications:
- If tile reuse is weak, the kernel becomes more sensitive to `LSC` / `memory` bandwidth.
- If tile sizes are too large, register pressure and occupancy can collapse.
- If tile sizes are too small, math engines are starved and per-tile overhead dominates.

Questions to ask:
- Is the chosen tile shape increasing reuse of the bandwidth-dominant operand?
- Is the implementation limited by L1 capacity, prefetch distance, or outer-cache refill pressure?
- Would changing tile shape shift pressure from DRAM to LSC, or from LSC to L1?

## 3. Tile -> MMA Atom

Goal:
- Pack enough MMA atoms into each tile so tensor engines (`Tensor Core` / `XMX`) stay busy.

Hardware mapping:
- CUDA: MMA atoms typically consume data staged in SMEM and move it into registers for tensor-core execution.
- Intel XPU: MMA atoms consume data staged from L1 into the register file (`GRF` / `RGF`) before `XMX` execution.

Key consequence:
- On Intel, the `L1 -> GRF -> XMX` path is central to sustaining compute throughput.
- Insufficient MMA-atom density per tile can make the kernel under-utilize XMX even if outer-memory traffic is well controlled.

Performance implications:
- Too little work per tile causes instruction overhead and data movement to dominate.
- Too aggressive an atom packing strategy can increase GRF pressure and reduce occupancy.
- The best-performing tile is usually the one that balances XMX issue rate, GRF capacity, and L1 refill behavior.

Questions to ask:
- Does each tile contain enough MMA atoms to amortize data movement?
- Is GRF pressure limiting occupancy or forcing spills?
- Is the kernel actually compute limited, or is XMX underfed by the L1/GRF path?

## How This Interacts with Roofline

Use the hierarchy to interpret roofline outcomes:

- `memory bound`
  - Usually points to weak global reuse, poor block scheduling, or a bandwidth-dominant operand overwhelming outer-cache reuse.
- `LSC bound`
  - Usually points to inter-block or inter-tile reuse not being converted into sufficient locality, or to outer-cache contention.
- `L1 bound`
  - Often means the implementation has already reduced DRAM pressure, but the `Tile -> MMA Atom` feed path is still bandwidth-limited.
- `compute bound`
  - Usually means hierarchy is working well enough that math throughput is now the active ceiling.

## Practical Interpretation for Skinny Shapes

For shapes like `M << N, K` or `N << M, K`:

- One operand often dominates traffic.
- Block scheduling matters more because outer-cache reuse is asymmetric.
- Tile shape should be biased toward reusing the dominant operand.
- A kernel can look mathematically large but still have low operational intensity because reuse is structurally weak.

## Intel-specific Summary

- `Global -> Block`: watch outer-cache (`LSC` / `L2`) sharing across XeCores.
- `Block -> Tile`: prefetch and L1-oriented staging matter more than CUDA-style `cp.async` narratives on pre-Xe4 platforms.
- `Tile -> MMA Atom`: the `L1 -> GRF -> XMX` path is part of the critical throughput story.

These points should be used together with roofline numbers, not as a substitute for them.