"""Check geology-report statements against the well's own logs.

Reports are free text of unknown provenance, and the chat model may quote them.
This module tags each measurable statement as:
  consistent         - the logs support the claimed value
  partly_consistent  - the claim overlaps the logged values but misses the typical value
  contradicted       - the logs disagree; both values are reported
  not_checkable      - needs mudlog, core or test data the logs cannot provide
Each check carries the actual log statistics and the rule used, so a reader
(or the model) can judge it. Heuristic by nature: it only parses statements
that name a curve, a gamma-ray / porosity range, NTG, or crossover.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.petrophysics import (
    PAY_MODEL, PHI_CUTOFF, SW_CUTOFF, VSH_CUTOFF,
    _well_gr_bounds, compute_net_pay, derive_vsh_phi_sw, read_las,
)

NOT_CHECKABLE = re.compile(
    r"mudlog|chromatograph|fluorescen|streaming cut|cuttings|\bcore\b|core plug|gas shows?|"
    r"shows recorded|temperature|°\s*api|casing", re.I)
DEPTH_RANGE = re.compile(r"(\d{3,5}(?:\.\d+)?)\s*m?\s*(?:-|–|to|and)\s*(\d{3,5}(?:\.\d+)?)\s*m\b", re.I)
NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?:\s*(?:-|–|to)\s*(\d+(?:\.\d+)?))?")
MNEMONIC = re.compile(r"`([A-Z][A-Z0-9_]{1,10})`")
GR_RANGE = re.compile(r"(\d+)\s*[-–]\s*(\d+)\s*API", re.I)
GR_BOUND = re.compile(r"([<>])\s*(\d+)\s*API", re.I)
PORO_RANGE = re.compile(r"porosity[^%]*?(\d+)\s*[-–]\s*(\d+)\s*%|\((\d+)\s*[-–]\s*(\d+)\s*%\)", re.I)
NTG = re.compile(r"NTG\s*=\s*(\d+(?:\.\d+)?)\s*%", re.I)

RANGE_TOL = 0.10   # widen claimed ranges by 10% before judging
VALUE_TOL = 0.10   # single values: within 10% (or 0.02 absolute for fractions)


def _clean(text: str) -> str:
    return re.sub(r"[*`]", "", text).strip(" -:")[:240]


def _stats(x: np.ndarray) -> Optional[Dict[str, float]]:
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return None
    return {"p10": round(float(np.percentile(x, 10)), 3), "p50": round(float(np.percentile(x, 50)), 3),
            "p90": round(float(np.percentile(x, 90)), 3), "mean": round(float(np.mean(x)), 3), "n": int(len(x))}


def _judge_range(lo: float, hi: float, st: Dict[str, float], stat: str) -> str:
    pad = RANGE_TOL * max(abs(hi - lo), abs(hi) * 0.1, 1e-9)
    if lo - pad <= st[stat] <= hi + pad:
        return "consistent"
    if st["p10"] <= hi + pad and st["p90"] >= lo - pad:
        return "partly_consistent"
    return "contradicted"


def _judge_value(v: float, st: Dict[str, float]) -> str:
    tol = max(VALUE_TOL * abs(v), 0.02 if abs(v) < 1 else 0)
    if abs(st["mean"] - v) <= tol:
        return "consistent"
    if st["p10"] <= v <= st["p90"]:
        return "partly_consistent"
    return "contradicted"


class _Well:
    """Log arrays plus clean-reservoir / pay masks, computed once per well."""

    def __init__(self, well_id: str) -> None:
        _, df = read_las(well_id)
        self.well_id = well_id
        self.df = df
        self.cols = {str(c).upper(): c for c in df.columns}
        vsh, phi, sw, _ = derive_vsh_phi_sw(df, self.cols, gr_bounds=_well_gr_bounds(df), **PAY_MODEL)
        ok = ~np.isnan(vsh) & ~np.isnan(phi)
        self.phi = phi
        self.clean = ok & (vsh <= VSH_CUTOFF) & (phi >= PHI_CUTOFF)
        self.pay = self.clean & ~np.isnan(sw) & (sw <= SW_CUTOFF)
        d = df["DEPTH"].values
        self.depth = d
        self.lo, self.hi = float(np.nanmin(d)), float(np.nanmax(d))
        if {"NEUT", "DENB"} <= set(self.cols):
            n = df[self.cols["NEUT"]].values
            dphi = (2.65 - df[self.cols["DENB"]].values) / 1.65
            self.crossover = (dphi - n) > 0.02
        else:
            self.crossover = None

    def window(self, top: float, bottom: float) -> np.ndarray:
        return (self.depth >= top) & (self.depth <= bottom)

    def selection(self, text: str) -> Tuple[str, np.ndarray]:
        t = text.lower()
        if re.search(r"\bpay\b|sweet", t):
            return "pay samples", self.pay
        if re.search(r"sand|porous|reservoir|clean", t):
            return "clean, porous samples", self.clean
        return "all samples", np.ones(len(self.depth), dtype=bool)


def _check_formation_row(w: _Well, cells: List[str]) -> List[Dict[str, Any]]:
    try:
        top, base = float(cells[1]), float(cells[2])
    except (IndexError, ValueError):
        return []
    name, lith = _clean(cells[0]), cells[4] if len(cells) > 4 else ""
    if base < w.lo or top > w.hi:
        return [{"statement": f"{name}: {_clean(lith)}", "interval_m": [top, base], "status": "not_checkable",
                 "reason": "interval is outside the logged depth range"}]
    win = w.window(top, base)
    out = []
    if "GR" in w.cols and (GR_RANGE.search(lith) or GR_BOUND.search(lith)):
        st = _stats(w.df[w.cols["GR"]].values[win])
        if not st:
            out.append({"statement": f"{name}: {_clean(lith)}", "interval_m": [top, base], "status": "not_checkable",
                        "reason": "the log has no gamma-ray data in this interval"})
        else:
            m = GR_RANGE.search(lith)
            if m:
                lo, hi = float(m.group(1)), float(m.group(2))
                status, claimed = _judge_range(lo, hi, st, "p50"), f"GR {lo:g}-{hi:g} API"
            else:
                b = GR_BOUND.search(lith)
                op, v = b.group(1), float(b.group(2))
                status = "consistent" if (st["p50"] < v if op == "<" else st["p50"] > v) else (
                    "partly_consistent" if (st["p10"] < v if op == "<" else st["p90"] > v) else "contradicted")
                claimed = f"GR {op} {v:g} API"
            out.append({"statement": f"{name}: {_clean(lith)}", "interval_m": [top, base], "claimed": claimed,
                        "actual": {"curve": "GR", "samples": "all samples", **st},
                        "rule": "median GR of the interval vs the claim", "status": status})
    p = PORO_RANGE.search(lith)
    if p:
        lo, hi = [float(g) for g in p.groups() if g is not None][:2]
        label, sel = w.selection(lith)
        st = _stats(w.phi[win & sel] * 100.0)
        if st:
            out.append({"statement": f"{name}: {_clean(lith)}", "interval_m": [top, base],
                        "claimed": f"porosity {lo:g}-{hi:g}%",
                        "actual": {"curve": "porosity (%)", "samples": label, **st},
                        "rule": "median porosity of the selected samples vs the claimed range",
                        "status": _judge_range(lo, hi, st, "p50")})
    return out


def _check_line(w: _Well, line: str, interval: Optional[Tuple[float, float]], context: str) -> List[Dict[str, Any]]:
    text = _clean(line)
    if NOT_CHECKABLE.search(line):
        return [{"statement": text, "status": "not_checkable",
                 "reason": "mudlog, core, test or completion data; well logs cannot confirm or refute it"}]
    m = DEPTH_RANGE.search(line)
    if m:
        interval = (float(m.group(1)), float(m.group(2)))
    if interval is None:
        return []
    top, base = interval
    win = w.window(top, base)
    if not np.any(win):
        return []
    label, sel = w.selection(line + " " + context)
    out: List[Dict[str, Any]] = []

    # Curve claims: `MNEMONIC` followed by a value or range in the same clause
    mentions = [mm for mm in MNEMONIC.finditer(line) if mm.group(1) in w.cols]
    for i, mm in enumerate(mentions):
        curve = mm.group(1)
        clause_end = mentions[i + 1].start() if i + 1 < len(mentions) else len(line)
        clause = line[mm.end():clause_end]
        num = next((n for n in NUMBER.finditer(clause)
                    if not re.match(r"\s*m\b", clause[n.end():])), None)  # skip depths like "1900-1920m"
        if not num:
            continue
        vals = w.df[w.cols[curve]].values
        st = _stats(vals[win & sel]) or _stats(vals[win])
        if not st:
            continue
        before = line[max(0, mm.start() - 60):mm.start()].lower()
        words = (before + clause[:num.start()]).lower()
        if num.group(2):
            lo, hi = float(num.group(1)), float(num.group(2))
            if re.search(r"under|below|less than|<", words):
                status = "consistent" if st["p50"] <= hi * (1 + RANGE_TOL) else (
                    "partly_consistent" if st["p10"] <= hi else "contradicted")
                rule, claimed = "median must be under the upper bound", f"{curve} under {hi:g}"
            else:
                stat = "p90" if re.search(r"up to|spike|peak|reach", words) else (
                    "p10" if re.search(r"dips? to|drops? to|as low as|down to", words) else "p50")
                status = _judge_range(lo, hi, st, stat)
                rule, claimed = f"{stat} of the selected samples vs the claimed range", f"{curve} {lo:g}-{hi:g}"
        else:
            v = float(num.group(1))
            status, rule, claimed = _judge_value(v, st), "mean of the selected samples vs the claimed value", f"{curve} {v:g}"
        out.append({"statement": text, "interval_m": [top, base], "claimed": claimed,
                    "actual": {"curve": curve, "samples": label, **st}, "rule": rule, "status": status})

    # Net-to-gross claims, recomputed with the app's own net pay
    n = NTG.search(line)
    if n:
        claimed = float(n.group(1))
        try:
            actual = round(compute_net_pay(w.well_id, top, base)["net_to_gross"] * 100, 1)
            status = "consistent" if abs(actual - claimed) <= 2 else (
                "partly_consistent" if abs(actual - claimed) <= 10 else "contradicted")
            out.append({"statement": text, "interval_m": [top, base], "claimed": f"NTG {claimed:g}%",
                        "actual": {"net_to_gross_pct": actual}, "rule": "net pay tool, same interval (±2 points)",
                        "status": status})
        except ValueError:
            pass

    # Density-neutron crossover claims
    if re.search(r"crossover", line, re.I) and w.crossover is not None and not mentions:
        frac = round(float(np.mean(w.crossover[win])) * 100, 1)
        status = "consistent" if frac >= 50 else ("partly_consistent" if frac >= 10 else "contradicted")
        out.append({"statement": text, "interval_m": [top, base], "claimed": "density-neutron crossover",
                    "actual": {"samples_with_crossover_pct": frac},
                    "rule": "share of samples where density porosity exceeds neutron by > 0.02",
                    "status": status,
                    "note": "crossover is a gas-like signature; it does not by itself prove gas vs oil"})
    return out


def verify_report(well_id: str, report_text: str) -> Dict[str, Any]:
    w = _Well(well_id)
    checks: List[Dict[str, Any]] = []
    interval: Optional[Tuple[float, float]] = None
    context = ""
    for raw in report_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 3 and re.match(r"^\d", cells[1] or ""):
                checks += _check_formation_row(w, cells)
            continue
        if line.startswith("#"):
            interval, context = None, ""
            continue
        is_header = line.endswith(":") or line.endswith("**:")
        m = DEPTH_RANGE.search(line)
        if is_header:
            interval = (float(m.group(1)), float(m.group(2))) if m else None
            context = line
            continue
        if line.startswith("-"):
            checks += _check_line(w, line, interval, context)

    summary = {s: sum(1 for c in checks if c["status"] == s)
               for s in ("consistent", "partly_consistent", "contradicted", "not_checkable")}
    return {"checked_against": f"{well_id} logs", "summary": summary, "checks": checks}
