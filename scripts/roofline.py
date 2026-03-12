#!/usr/bin/env python3
"""Roofline calculator for GEMM / Attention.

Design goals:
- Zero dependencies
- Useful with either (a) only shape + peaks (theoretical, DRAM-level) or
  (b) profiler bytes + time (multi-level attribution: L1/LSC/DRAM)

This is a high-level model; it does not attempt to infer cache reuse/tiling.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


_DTYPE_BYTES = {
    "fp32": 4,
    "tf32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
}


def _load_presets() -> dict[str, dict[str, Any]]:
    """Load presets from references/hardware_presets.json.

    The JSON is the source of truth. Each preset must use the new schema:
        {
            "<preset>": {
                "roofline": {"bw_mem_gbs": ..., "bw_lsc_gbs": ..., "bw_l1_gbs": ...},
                "fields": {...}
            }
        }

    Each preset keeps all HW fields under `fields` and the roofline-facing data
    under `roofline`.
    """

    fallback: dict[str, dict[str, Any]] = {
        "bmg580": {
            "roofline": {
                "frequency_mhz": 2850.0,
                "peak_tflops_by_dtype": {"fp16": 117.0, "bf16": 117.0},
                "bw_mem_gbs": 456.0,
                "bw_l1_gbs_per_xe": 729.6,
                "bw_l1_gbs": 14592.0,
                "bw_lsc_gbs": 4377.6,
            },
            "fields": {},
        }
    }

    presets_path = Path(__file__).resolve().parent.parent / "references" / "hardware_presets.json"
    if not presets_path.exists():
        return fallback

    try:
        payload = json.loads(presets_path.read_text(encoding="utf-8"))
    except Exception:
        return fallback

    presets: dict[str, dict[str, Any]] = {}
    raw = payload.get("presets", {}) if isinstance(payload, dict) else {}
    if not isinstance(raw, dict):
        return fallback

    for name, fields in raw.items():
        if not isinstance(name, str) or not isinstance(fields, dict):
            continue
        roofline_fields = fields.get("roofline")
        hw_fields = fields.get("fields")
        if not isinstance(roofline_fields, dict) or not isinstance(hw_fields, dict):
            continue
        presets[name] = fields

    return presets or fallback


_PRESETS: dict[str, dict[str, Any]] = _load_presets()


def _as_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _derive_roofline(preset: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(preset, dict):
        return {}

    roofline = dict(preset.get("roofline", {})) if isinstance(preset.get("roofline"), dict) else {}
    fields = preset.get("fields", {}) if isinstance(preset.get("fields"), dict) else {}

    # Allow raw fields to backfill roofline metadata.
    frequency_mhz = _as_float(roofline.get("frequency_mhz"))
    if frequency_mhz is None:
        frequency_mhz = _as_float(fields.get("Frequency (MHz)"))
        if frequency_mhz is not None:
            roofline["frequency_mhz"] = frequency_mhz

    peak_by_dtype = roofline.get("peak_tflops_by_dtype")
    if not isinstance(peak_by_dtype, dict):
        peak_by_dtype = {}
    for dtype, field_name in (
        ("fp16", "FP16 theoretical Peak FLOPS (TFLOPS)"),
        ("bf16", "BF16 theoretical Peak FLOPS (TFLOPS)"),
    ):
        if dtype not in peak_by_dtype:
            value = _as_float(fields.get(field_name))
            if value is not None:
                peak_by_dtype[dtype] = value
    roofline["peak_tflops_by_dtype"] = peak_by_dtype

    if _as_float(roofline.get("bw_mem_gbs")) is None:
        value = _as_float(fields.get("memory GB/s"))
        if value is not None:
            roofline["bw_mem_gbs"] = value

    # Cache-level roofs use read bandwidth only.
    if _as_float(roofline.get("bw_l1_gbs_per_xe")) is None and frequency_mhz is not None:
        l1_rd = _as_float(fields.get("l1 rd rate/clock"))
        if l1_rd is not None:
            roofline["bw_l1_gbs_per_xe"] = l1_rd * frequency_mhz / 1e3

    if _as_float(roofline.get("bw_l1_gbs")) is None and frequency_mhz is not None:
        bw_l1_gbs_per_xe = _as_float(roofline.get("bw_l1_gbs_per_xe"))
        xecore_count = _as_float(fields.get("XeCore#"))
        if bw_l1_gbs_per_xe is not None and xecore_count is not None:
            roofline["bw_l1_gbs"] = bw_l1_gbs_per_xe * xecore_count

    if _as_float(roofline.get("bw_lsc_gbs")) is None and frequency_mhz is not None:
        l3_rate = _as_float(fields.get("l3 rate/bank/clock"))
        l3_bank = _as_float(fields.get("L3 BANK"))
        if l3_rate is not None and l3_bank is not None:
            roofline["bw_lsc_gbs"] = l3_rate * l3_bank * frequency_mhz / 1e3

    return roofline


def _resolve_peak_tflops(explicit: Optional[float], dtype: str, preset: Optional[dict[str, Any]]) -> Optional[float]:
    if explicit is not None:
        return explicit

    roofline = _derive_roofline(preset)
    peak_by_dtype = roofline.get("peak_tflops_by_dtype")
    if not isinstance(peak_by_dtype, dict):
        return None

    value = peak_by_dtype.get(dtype)
    return _as_float(value)


def _resolve_bandwidth_overrides(args: argparse.Namespace, preset: Optional[dict[str, Any]]) -> None:
    roofline = _derive_roofline(preset)
    if args.bw_mem_gbs is None:
        args.bw_mem_gbs = _as_float(roofline.get("bw_mem_gbs"))
    if args.bw_lsc_gbs is None:
        args.bw_lsc_gbs = _as_float(roofline.get("bw_lsc_gbs"))
    if args.bw_l1_gbs is None:
        args.bw_l1_gbs = _as_float(roofline.get("bw_l1_gbs"))


@dataclass(frozen=True)
class Roofs:
    peak_tflops: float
    bw_l1_gbs: Optional[float] = None
    bw_lsc_gbs: Optional[float] = None
    bw_mem_gbs: Optional[float] = None


@dataclass(frozen=True)
class Measured:
    time_ms: Optional[float] = None
    bytes_l1: Optional[float] = None
    bytes_lsc: Optional[float] = None
    bytes_mem: Optional[float] = None


def _tflops_to_flops_per_s(tflops: float) -> float:
    return tflops * 1e12


def _gbs_to_bytes_per_s(gbs: float) -> float:
    return gbs * 1e9


def _fmt_num(x: Optional[float], unit: str = "") -> str:
    if x is None:
        return "n/a"
    if x == 0:
        return f"0{unit}"
    absx = abs(x)
    if absx >= 1e12:
        return f"{x/1e12:.3g}e12{unit}"
    if absx >= 1e9:
        return f"{x/1e9:.3g}e9{unit}"
    if absx >= 1e6:
        return f"{x/1e6:.3g}e6{unit}"
    if absx >= 1e3:
        return f"{x/1e3:.3g}e3{unit}"
    return f"{x:.6g}{unit}"


def _ai(flops: float, bytes_moved: Optional[float]) -> Optional[float]:
    if bytes_moved is None or bytes_moved <= 0:
        return None
    return flops / bytes_moved


def _roof_perf_tflops(ai_flop_per_byte: Optional[float], bw_gbs: Optional[float]) -> Optional[float]:
    if ai_flop_per_byte is None or bw_gbs is None:
        return None
    # ai [flop/byte] * bw [GB/s] => flop/s; convert to TFLOP/s
    return (ai_flop_per_byte * _gbs_to_bytes_per_s(bw_gbs)) / 1e12


def _min_roof_label(peak_tflops: float, roofs: dict[str, Optional[float]]) -> tuple[str, float]:
    # roofs is mapping label->tflops (already computed). Ignore None.
    active = {k: v for k, v in roofs.items() if v is not None}
    active["compute"] = peak_tflops
    # bottleneck is the minimal attainable perf
    label = min(active, key=lambda k: active[k])
    return label, active[label]


def _bound_to_user_label(active_roof: str) -> str:
    if active_roof == "compute":
        return "compute bound"
    if active_roof == "l1":
        return "L1 bound"
    if active_roof == "lsc":
        return "LSC bound"
    if active_roof == "mem":
        return "memory bound"
    return f"{active_roof} bound"


def gemm_flops(m: int, n: int, k: int) -> float:
    return float(2 * m * n * k)


def gemm_bytes_min_mem(m: int, n: int, k: int, bytes_per_elem: int) -> float:
    # Minimal global traffic: read A,B once and write C once.
    return float((m * k + k * n + m * n) * bytes_per_elem)


def attn_flops_fused(b: int, h: int, s: int, d: int) -> float:
    # QK^T: (SxD)*(DxS) => 2*S*S*D
    # PV:   (SxS)*(SxD) => 2*S*S*D
    # Total ~ 4*S*S*D per head
    return float(b * h * 4 * s * s * d)


def attn_bytes_min_mem(b: int, h: int, s: int, d: int, bytes_per_elem: int) -> float:
    # Minimal global traffic for fused attention (no materialized scores):
    # read Q,K,V and write O
    elems = (3 + 1) * b * h * s * d
    return float(elems * bytes_per_elem)


def _print_section(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def analyze(flops: float, bytes_min_mem: float, dtype: str, roofs: Roofs, measured: Measured) -> int:
    _print_section("Inputs")
    print(f"dtype: {dtype} ({_DTYPE_BYTES[dtype]} bytes/elem)")
    print(f"FLOPs (est): {_fmt_num(flops)}")
    print(f"Bytes_mem (est min): {_fmt_num(bytes_min_mem, 'B')}")
    print(f"peak_tflops: {roofs.peak_tflops}")
    print(f"bw_l1_gbs: {_fmt_num(roofs.bw_l1_gbs)}")
    print(f"bw_lsc_gbs: {_fmt_num(roofs.bw_lsc_gbs)}")
    print(f"bw_mem_gbs: {_fmt_num(roofs.bw_mem_gbs)}")

    _print_section("Arithmetic Intensity")
    ai_mem = _ai(flops, measured.bytes_mem if measured.bytes_mem is not None else bytes_min_mem)
    ai_l1 = _ai(flops, measured.bytes_l1)
    ai_lsc = _ai(flops, measured.bytes_lsc)

    print(f"AI_mem  (flop/byte): {_fmt_num(ai_mem)}")
    print(f"AI_lsc  (flop/byte): {_fmt_num(ai_lsc)} (requires bytes_lsc)")
    print(f"AI_l1   (flop/byte): {_fmt_num(ai_l1)} (requires bytes_l1)")

    _print_section("Roofline Ceilings (TFLOP/s)")
    roof_l1 = _roof_perf_tflops(ai_l1, roofs.bw_l1_gbs)
    roof_lsc = _roof_perf_tflops(ai_lsc, roofs.bw_lsc_gbs)
    roof_mem = _roof_perf_tflops(ai_mem, roofs.bw_mem_gbs)

    if roof_l1 is not None:
        print(f"L1  roof: {_fmt_num(roof_l1)}")
    else:
        print("L1  roof: n/a (need bw_l1_gbs and bytes_l1)")

    if roof_lsc is not None:
        print(f"LSC roof: {_fmt_num(roof_lsc)}")
    else:
        print("LSC roof: n/a (need bw_lsc_gbs and bytes_lsc)")

    if roof_mem is not None:
        print(f"MEM roof: {_fmt_num(roof_mem)}")
    else:
        print("MEM roof: n/a (need bw_mem_gbs)")

    print(f"Compute roof: {_fmt_num(roofs.peak_tflops)}")

    # Determine active roof from what we have.
    # If we lack bytes_l1/bytes_lsc, we cannot meaningfully classify L1 vs LSC.
    available_roofs = {
        "l1": roof_l1,
        "lsc": roof_lsc,
        "mem": roof_mem,
    }

    active_roof, bound_tflops = _min_roof_label(roofs.peak_tflops, available_roofs)

    _print_section("Conclusion")
    print(f"Bound (roofline): {_bound_to_user_label(active_roof)}")
    print(f"Active ceiling: {_fmt_num(bound_tflops)} TFLOP/s")

    # Extra: if time is given, report achieved metrics and a sanity-check.
    if measured.time_ms is not None and measured.time_ms > 0:
        secs = measured.time_ms / 1e3
        achieved_tflops = (flops / secs) / 1e12
        print(f"Achieved: {_fmt_num(achieved_tflops)} TFLOP/s (from time_ms)")

        def _ach_bw(bytes_moved: Optional[float]) -> Optional[float]:
            if bytes_moved is None:
                return None
            return (bytes_moved / secs) / 1e9

        ach_mem = _ach_bw(measured.bytes_mem)
        ach_lsc = _ach_bw(measured.bytes_lsc)
        ach_l1 = _ach_bw(measured.bytes_l1)

        if ach_mem is not None:
            print(f"Achieved BW_mem: {_fmt_num(ach_mem)} GB/s")
        if ach_lsc is not None:
            print(f"Achieved BW_lsc: {_fmt_num(ach_lsc)} GB/s")
        if ach_l1 is not None:
            print(f"Achieved BW_l1 : {_fmt_num(ach_l1)} GB/s")

        # Basic consistency note
        if achieved_tflops > roofs.peak_tflops * 1.05:
            print("Note: achieved TFLOP/s exceeds peak_tflops; check peak_tflops/dtype/FLOPs model.")

    # Guidance on missing info
    missing = []
    if roofs.bw_mem_gbs is None:
        missing.append("bw_mem_gbs")
    if roofs.bw_l1_gbs is not None and measured.bytes_l1 is None:
        missing.append("bytes_l1")
    if roofs.bw_lsc_gbs is not None and measured.bytes_lsc is None:
        missing.append("bytes_lsc")

    if missing:
        print("Missing for finer attribution: " + ", ".join(missing))
        if (measured.bytes_l1 is None and measured.bytes_lsc is None) and (roofs.bw_l1_gbs is not None or roofs.bw_lsc_gbs is not None):
            print("Tip: to decide L1 vs LSC bound, provide bytes_l1/bytes_lsc from profiler counters.")

    return 0


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dtype", required=True, choices=sorted(_DTYPE_BYTES.keys()))
    p.add_argument(
        "--peak-tflops",
        type=float,
        help="Peak TFLOP/s for the selected dtype; optional if the selected preset provides it",
    )
    p.add_argument(
        "--preset",
        choices=sorted(_PRESETS.keys()),
        help="Hardware preset to auto-fill known roofs from hardware_presets.json",
    )
    p.add_argument("--bw-l1-gbs", type=float)
    p.add_argument("--bw-lsc-gbs", type=float)
    p.add_argument("--bw-mem-gbs", type=float)
    p.add_argument("--time-ms", type=float)
    p.add_argument("--bytes-l1", type=float)
    p.add_argument("--bytes-lsc", type=float)
    p.add_argument("--bytes-mem", type=float)


def main() -> int:
    ap = argparse.ArgumentParser(description="Roofline calculator (GEMM/Attention)")
    sub = ap.add_subparsers(dest="op", required=True)

    ap_gemm = sub.add_parser("gemm", help="GEMM: C[M,N] = A[M,K] * B[K,N]")
    ap_gemm.add_argument("--m", type=int, required=True)
    ap_gemm.add_argument("--n", type=int, required=True)
    ap_gemm.add_argument("--k", type=int, required=True)
    _add_common_args(ap_gemm)

    ap_attn = sub.add_parser("attn", help="Attention (fused estimate): QK^T + PV")
    ap_attn.add_argument("--b", type=int, required=True, help="batch")
    ap_attn.add_argument("--h", type=int, required=True, help="heads")
    ap_attn.add_argument("--s", type=int, required=True, help="sequence length")
    ap_attn.add_argument("--d", type=int, required=True, help="head dimension")
    _add_common_args(ap_attn)

    args = ap.parse_args()

    preset = _PRESETS.get(args.preset) if getattr(args, "preset", None) else None
    if preset is not None:
        _resolve_bandwidth_overrides(args, preset)

    dtype = args.dtype
    bytes_per_elem = _DTYPE_BYTES[dtype]
    peak_tflops = _resolve_peak_tflops(args.peak_tflops, dtype, preset)
    if peak_tflops is None:
        ap.error("--peak-tflops is required unless the selected preset provides peak_tflops_by_dtype for the chosen dtype")

    roofs = Roofs(
        peak_tflops=peak_tflops,
        bw_l1_gbs=args.bw_l1_gbs,
        bw_lsc_gbs=args.bw_lsc_gbs,
        bw_mem_gbs=args.bw_mem_gbs,
    )
    measured = Measured(
        time_ms=args.time_ms,
        bytes_l1=args.bytes_l1,
        bytes_lsc=args.bytes_lsc,
        bytes_mem=args.bytes_mem,
    )

    if args.op == "gemm":
        flops = gemm_flops(args.m, args.n, args.k)
        bytes_mem = gemm_bytes_min_mem(args.m, args.n, args.k, bytes_per_elem)
        return analyze(flops, bytes_mem, dtype, roofs, measured)

    if args.op == "attn":
        flops = attn_flops_fused(args.b, args.h, args.s, args.d)
        bytes_mem = attn_bytes_min_mem(args.b, args.h, args.s, args.d, bytes_per_elem)
        return analyze(flops, bytes_mem, dtype, roofs, measured)

    raise RuntimeError(f"Unknown op: {args.op}")


if __name__ == "__main__":
    raise SystemExit(main())
