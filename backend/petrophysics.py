"""Deterministic Petrophysical Tool Engine.

Handles tabular depth-series calculation and multi-track / crossplot generation
using lasio, pandas, numpy, and plotly.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import lasio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull

from backend.config import DATA_DIR


def _find_las_path(well_id: str) -> Path:
    clean_id = well_id.strip()
    candidates = [
        DATA_DIR / clean_id,
        DATA_DIR / f"{clean_id}.las",
        DATA_DIR / f"{clean_id.capitalize()}.las",
        DATA_DIR / f"{clean_id.lower()}.las",
        DATA_DIR / f"{clean_id.upper()}.las",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    for p in DATA_DIR.glob("*.las"):
        if p.stem.lower() == clean_id.lower() or p.stem.lower() in clean_id.lower() or clean_id.lower() in p.stem.lower():
            return p
    raise FileNotFoundError(f"LAS file for well '{well_id}' not found in {DATA_DIR}")


def read_las(well_id: str) -> tuple[lasio.LASFile, pd.DataFrame]:
    las_path = _find_las_path(well_id)
    las = lasio.read(str(las_path))
    df = las.df().reset_index()
    
    # Ensure DEPTH is a clean column
    depth_cols = [c for c in df.columns if str(c).upper() == "DEPTH"]
    if depth_cols:
        if depth_cols[0] != "DEPTH":
            df["DEPTH"] = df[depth_cols[0]]
    else:
        df["DEPTH"] = np.linspace(float(las.well.STRT.value), float(las.well.STOP.value), len(df))
    
    null_val = las.well.NULL.value if "NULL" in las.well else -999.25
    df = df.replace([null_val, -999.0, -999.25, -9999.0], np.nan)
    return las, df


def get_well_curves_summary(well_id: str) -> Dict[str, Any]:
    """Inspects the .las file and returns available curve mnemonics, units, and depth bounds."""
    las, df = read_las(well_id)
    
    curves_info = []
    for curve in las.curves:
        valid_count = int((df[curve.mnemonic].notna()).sum()) if curve.mnemonic in df.columns else 0
        min_val = float(df[curve.mnemonic].min()) if valid_count > 0 else None
        max_val = float(df[curve.mnemonic].max()) if valid_count > 0 else None
        curves_info.append({
            "mnemonic": curve.mnemonic,
            "unit": curve.unit or "",
            "description": curve.descr or "",
            "valid_samples": valid_count,
            "min": round(min_val, 4) if min_val is not None else None,
            "max": round(max_val, 4) if max_val is not None else None,
        })
        
    depth_col = df["DEPTH"].dropna()
    start_depth = float(depth_col.min()) if not depth_col.empty else float(las.well.STRT.value)
    stop_depth = float(depth_col.max()) if not depth_col.empty else float(las.well.STOP.value)
    step = float(las.well.STEP.value) if "STEP" in las.well else 0.1524

    return {
        "well_id": well_id,
        "well_name": las.well.WELL.value if "WELL" in las.well else well_id,
        "uwi": las.well.UWI.value if "UWI" in las.well else "UNKNOWN",
        "start_depth": round(start_depth, 2),
        "stop_depth": round(stop_depth, 2),
        "step": round(step, 4),
        "total_curves": len(curves_info),
        "curves": curves_info,
        "available_mnemonics": [c["mnemonic"] for c in curves_info],
    }


def _calc_crossover_polygons(
    d_arr: np.ndarray,
    v1_arr: np.ndarray,
    v2_arr: np.ndarray,
    condition_gt: bool = True
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Generates closed polygon coordinates between v1 and v2 where (v1 > v2) if condition_gt else (v1 < v2)."""
    valid_mask = (~np.isnan(v1_arr)) & (~np.isnan(v2_arr)) & (~np.isnan(d_arr))
    if np.sum(valid_mask) < 2:
        return [], []
    d = d_arr[valid_mask]
    v1 = v1_arr[valid_mask]
    v2 = v2_arr[valid_mask]

    mask = (v1 > v2) if condition_gt else (v1 < v2)
    if not np.any(mask):
        return [], []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    ends = np.where(diff == -1)[0]
    if mask[-1]:
        ends = np.r_[ends, len(mask) - 1]

    all_x: List[Optional[float]] = []
    all_y: List[Optional[float]] = []
    for s, e in zip(starts, ends):
        seg_d = list(d[s:e+1])
        seg_v1 = list(v1[s:e+1])
        seg_v2 = list(v2[s:e+1])

        start_x: List[float] = []
        start_y: List[float] = []
        if s > 0:
            d1 = v1[s] - v1[s-1]
            d2 = v2[s] - v2[s-1]
            denom = d1 - d2
            if abs(denom) > 1e-6:
                t = (v2[s-1] - v1[s-1]) / denom
                if 0.0 <= t <= 1.0:
                    start_y = [float(d[s-1] + t * (d[s] - d[s-1]))]
                    start_x = [float(v1[s-1] + t * d1)]

        end_x: List[float] = []
        end_y: List[float] = []
        if e < len(d) - 1:
            d1 = v1[e+1] - v1[e]
            d2 = v2[e+1] - v2[e]
            denom = d1 - d2
            if abs(denom) > 1e-6:
                t = (v2[e] - v1[e]) / denom
                if 0.0 <= t <= 1.0:
                    end_y = [float(d[e] + t * (d[e+1] - d[e]))]
                    end_x = [float(v1[e] + t * d1)]

        poly_x = start_x + seg_v1 + end_x + seg_v2[::-1] + start_x
        poly_y = start_y + seg_d + end_y + seg_d[::-1] + start_y
        all_x.extend(poly_x + [None])
        all_y.extend(poly_y + [None])
    return all_x, all_y


def plot_1d_well_log(well_id: str, curves: Optional[List[str]] = None, top_depth: Optional[float] = None, bottom_depth: Optional[float] = None, marker_depth: Optional[float] = None) -> Dict[str, Any]:
    """Slices the .las depth range using pandas and returns an interactive multi-track Plotly figure."""
    las, df = read_las(well_id)
    
    if top_depth is not None:
        df = df[df["DEPTH"] >= top_depth]
    if bottom_depth is not None:
        df = df[df["DEPTH"] <= bottom_depth]
        
    if df.empty:
        raise ValueError(f"No log samples found between {top_depth}m and {bottom_depth}m for {well_id}")
        
    df = df.sort_values("DEPTH")
    depth = df["DEPTH"].values
    
    cols = {c.upper(): c for c in df.columns}
    gr_col = cols.get("GR")
    rdeep_col = cols.get("RDEEP") or cols.get("ILD") or cols.get("LLD") or cols.get("RT")
    rmed_col = cols.get("RMED") or cols.get("ILM") or cols.get("LLS")
    rshal_col = cols.get("RSHAL") or cols.get("MSFL") or cols.get("SFLU")
    den_col = cols.get("DENB") or cols.get("RHOB") or cols.get("RHOZ")
    neut_col = cols.get("NEUT") or cols.get("NPHI") or cols.get("TNPH")
    cali_col = cols.get("CALI") or cols.get("CAL")
    
    has_cali = bool(cali_col and cali_col in df and np.any(~np.isnan(df[cali_col].values)))
    track1_title = "Track 1: Gamma Ray & Caliper" if has_cali else "Track 1: Gamma Ray & Bit Size"
    
    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        subplot_titles=[track1_title, "Track 2: Resistivity (log scale)", "Track 3: Density - Neutron Crossover"],
        horizontal_spacing=0.04
    )
    
    # Track 1: Gamma Ray, Caliper & Bit Size Reference
    if gr_col and gr_col in df:
        fig.add_trace(
            go.Scatter(
                x=df[gr_col].values, y=depth, name="GR (API)",
                line=dict(color="#15803d", width=1.6), mode="lines",
                connectgaps=True
            ),
            row=1, col=1
        )

        # Determine bit size reference
        bit_col = cols.get("BIT") or cols.get("BS")
        if has_cali:
            cali_vals = df[cali_col].values
            med_cali = float(np.nanmedian(cali_vals)) if np.any(~np.isnan(cali_vals)) else 8.5
            default_bs = 12.25 if med_cali > 10.5 else 8.5
        else:
            min_depth_val = float(np.nanmin(depth))
            default_bs = 12.25 if min_depth_val < 2500 else 8.5

        if bit_col and bit_col in df and np.any(~np.isnan(df[bit_col].values)):
            bit_vals = df[bit_col].values
            bit_name = "Bit Size (in)"
        else:
            bit_vals = np.full_like(depth, default_bs)
            bit_name = f"Bit Size Ref ({default_bs}\")"

        if has_cali:
            # Washout shading (CALI > BIT): subtle slate gray wash
            washout_x, washout_y = _calc_crossover_polygons(depth, cali_vals, bit_vals, condition_gt=True)
            if washout_x:
                fig.add_trace(
                    go.Scatter(
                        x=washout_x, y=washout_y,
                        fill="toself",
                        fillcolor="rgba(148, 163, 184, 0.32)",
                        line=dict(width=0), mode="lines",
                        name="Washout (> BS)",
                        hoverinfo="skip",
                        xaxis="x4", yaxis="y"
                    )
                )

            # Mudcake buildup shading (CALI < BIT): subtle warm amber wash
            mudcake_x, mudcake_y = _calc_crossover_polygons(depth, cali_vals, bit_vals, condition_gt=False)
            if mudcake_x:
                fig.add_trace(
                    go.Scatter(
                        x=mudcake_x, y=mudcake_y,
                        fill="toself",
                        fillcolor="rgba(217, 119, 6, 0.22)",
                        line=dict(width=0), mode="lines",
                        name="Mudcake (< BS)",
                        hoverinfo="skip",
                        xaxis="x4", yaxis="y"
                    )
                )

            # Add CALI to secondary top axis xaxis4
            fig.add_trace(
                go.Scatter(
                    x=cali_vals, y=depth, name="CALI (in)",
                    line=dict(color="#78350f", width=1.4, dash="dash"), mode="lines",
                    connectgaps=True,
                    xaxis="x4", yaxis="y"
                )
            )

        # Add Bit Size baseline
        fig.add_trace(
            go.Scatter(
                x=bit_vals, y=depth, name=bit_name,
                line=dict(color="#92400e", width=1.2, dash="dot"), mode="lines",
                connectgaps=True,
                xaxis="x4", yaxis="y"
            )
        )
            
    # Track 2: Resistivity (Standard 4-Decade Logarithmic Scale)
    res_traces = [
        (rdeep_col, "RDEEP (ohm.m)", "#b91c1c", "solid", 1.8),
        (rmed_col, "RMED (ohm.m)", "#ea580c", "longdash", 1.4),
        (rshal_col, "RSHAL (ohm.m)", "#0284c7", "solid", 1.3)
    ]
    for r_c, label, color, dash, width in res_traces:
        if r_c and r_c in df:
            r_vals = df[r_c].values
            valid_vals = np.where(r_vals > 0, r_vals, np.nan)
            if np.any(~np.isnan(valid_vals)):
                fig.add_trace(
                    go.Scatter(
                        x=valid_vals, y=depth, name=label,
                        line=dict(color=color, width=width, dash=dash), mode="lines",
                        connectgaps=True
                    ),
                    row=1, col=2
                )
                
    # Track 3: Density - Neutron Crossover (Strict Contiguous Closed Polygons on Textbook Dual Axis)
    has_porosity_track = False
    if den_col and den_col in df and neut_col and neut_col in df:
        has_porosity_track = True
        den_vals = df[den_col].values
        neut_vals = df[neut_col].values
        neut_scaled = neut_vals if np.nanmax(neut_vals) <= 1.0 else neut_vals / 100.0
        # SPWLA standard mapping: RHOB [1.95, 2.95] overlaying reversed NPHI [0.45, -0.15]
        # Equivalent RHOB = 2.70 - 1.6667 * NPHI
        neut_rho_equiv = 2.70 - (neut_scaled * 1.6667)

        v_mask = (~np.isnan(den_vals)) & (~np.isnan(neut_rho_equiv))
        v_depth = depth[v_mask]
        v_den = den_vals[v_mask]
        v_neut = neut_rho_equiv[v_mask]
        crossover_mask = (v_den < v_neut)

        if np.any(crossover_mask):
            diff = np.diff(crossover_mask.astype(int))
            starts = np.where(diff == 1)[0] + 1
            if crossover_mask[0]:
                starts = np.r_[0, starts]
            ends = np.where(diff == -1)[0]
            if crossover_mask[-1]:
                ends = np.r_[ends, len(crossover_mask) - 1]

            all_poly_x: List[Optional[float]] = []
            all_poly_y: List[Optional[float]] = []

            for s, e in zip(starts, ends):
                seg_d = list(v_depth[s:e+1])
                seg_den = list(v_den[s:e+1])
                seg_neut = list(v_neut[s:e+1])

                # Interpolate exact crossover entry point
                start_pt_x: List[float] = []
                start_pt_y: List[float] = []
                if s > 0:
                    d_den = v_den[s] - v_den[s-1]
                    d_neut = v_neut[s] - v_neut[s-1]
                    denom = d_den - d_neut
                    if abs(denom) > 1e-6:
                        t = (v_neut[s-1] - v_den[s-1]) / denom
                        if 0.0 <= t <= 1.0:
                            start_pt_y = [float(v_depth[s-1] + t * (v_depth[s] - v_depth[s-1]))]
                            start_pt_x = [float(v_den[s-1] + t * d_den)]

                # Interpolate exact crossover exit point
                end_pt_x: List[float] = []
                end_pt_y: List[float] = []
                if e < len(v_depth) - 1:
                    d_den = v_den[e+1] - v_den[e]
                    d_neut = v_neut[e+1] - v_neut[e]
                    denom = d_den - d_neut
                    if abs(denom) > 1e-6:
                        t = (v_neut[e] - v_den[e]) / denom
                        if 0.0 <= t <= 1.0:
                            end_pt_y = [float(v_depth[e] + t * (v_depth[e+1] - v_depth[e]))]
                            end_pt_x = [float(v_den[e] + t * d_den)]

                poly_x = start_pt_x + seg_den + end_pt_x + seg_neut[::-1] + start_pt_x
                poly_y = start_pt_y + seg_d + end_pt_y + seg_d[::-1] + start_pt_y
                all_poly_x.extend(poly_x + [None])
                all_poly_y.extend(poly_y + [None])

            if all_poly_x:
                fig.add_trace(
                    go.Scatter(
                        x=all_poly_x, y=all_poly_y,
                        fill="toself",
                        fillcolor="rgba(250, 204, 21, 0.45)",
                        line=dict(width=0), mode="lines",
                        name="Gas/Sand Crossover",
                        hoverinfo="skip"
                    ),
                    row=1, col=3
                )

        # Bottom RHOB curve (Solid Red, 1.95 - 2.95 g/cm³) on row 1, col 3 (xaxis3)
        fig.add_trace(
            go.Scatter(
                x=den_vals, y=depth, name="RHOB (g/cm³)",
                line=dict(color="#dc2626", width=1.5), mode="lines",
                connectgaps=True
            ),
            row=1, col=3
        )
        # Top NPHI curve (Solid Blue, 0.45 - -0.15 v/v reversed overlay) on secondary top axis xaxis5
        fig.add_trace(
            go.Scatter(
                x=neut_scaled, y=depth, name="NPHI (v/v)",
                line=dict(color="#2563eb", width=1.4, dash="solid"), mode="lines",
                connectgaps=True,
                xaxis="x5", yaxis="y3"
            )
        )

    min_d = float(np.nanmin(depth))
    max_d = float(np.nanmax(depth))
    
    # Configure synchronized spikeline cursor on all 3 depth tracks
    spike_cfg = dict(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1.5,
        spikedash="solid",
        spikecolor="#64748b"
    )
    fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1, **spike_cfg)
    fig.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=2, **spike_cfg)
    fig.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=3, **spike_cfg)
    
    # 1 Continuous Horizontal Line across all 3 graphs (xref='paper', yref='y')
    if marker_depth is None:
        candidate = 1908.0 if "1" in str(well_id) else 3650.0
        if min_d <= candidate <= max_d:
            marker_depth = candidate
        else:
            marker_depth = round(float(min_d + (max_d - min_d) * 0.5), 1)

    if min_d <= marker_depth <= max_d:
        fig.add_shape(
            type="line",
            xref="paper",
            yref="y",
            x0=0,
            x1=1,
            y0=marker_depth,
            y1=marker_depth,
            line=dict(color="#ef4444", width=2, dash="dash"),
            name="Depth Correlation Line"
        )
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=1.008,
            y=marker_depth,
            text=f"<b>Depth: {marker_depth:.1f}m</b>",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(color="#b91c1c", size=10, family="Inter, monospace"),
            bgcolor="rgba(254, 242, 242, 0.95)",
            bordercolor="#f87171",
            borderwidth=1,
            borderpad=3,
        )

    # Track 1 GR Axis: Avoid flat clipping when GR reaches ~200 API in shales
    gr_max = float(np.nanmax(df[gr_col].values)) if gr_col and gr_col in df and np.any(~np.isnan(df[gr_col].values)) else 150.0
    gr_upper = 200.0 if gr_max > 135.0 else 150.0
    fig.update_xaxes(
        title_text="GR (API)",
        range=[0, gr_upper],
        tickvals=[0, 50, 100, 150, 200] if gr_upper == 200.0 else [0, 50, 100, 150],
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text="Resistivity (ohm.m)",
        type="log",
        range=[float(np.log10(0.2)), float(np.log10(2000))],
        tickvals=[0.2, 1, 10, 100, 1000, 2000],
        ticktext=["0.2", "1", "10", "100", "1000", "2000"],
        dtick=1,
        showgrid=True,
        row=1, col=2
    )
    # Track 3 Bottom Axis: RHOB
    fig.update_xaxes(
        title_text="RHOB (g/cm³)",
        range=[1.95, 2.95],
        tickvals=[1.95, 2.20, 2.45, 2.70, 2.95],
        ticktext=["1.95", "2.20", "2.45", "2.70", "2.95"],
        title_font=dict(color="#dc2626", size=10),
        tickfont=dict(color="#dc2626", size=9),
        showgrid=True,
        row=1, col=3
    )
    
    # Move subplot title annotations up so they sit cleanly above top axes
    for a in fig.layout.annotations:
        if a.text and a.text.startswith("Track"):
            a.yshift = 32
            a.font = dict(size=11, color="#0f172a")

    depth_span = max_d - min_d
    plot_height = max(740, min(1200, int(depth_span * 7.4)))

    well_title = las.well.WELL.value if "WELL" in las.well else well_id
    layout_dict = dict(
        title=dict(
            text=f"<b>1D Petrophysical Log: {well_title}</b> ({min_d:.1f}m - {max_d:.1f}m)",
            x=0.5,
            y=0.985,
            xanchor="center",
            yanchor="top",
            font=dict(size=13, color="#0f172a")
        ),
        height=plot_height,
        template="plotly_white",
        hovermode="y",
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        margin=dict(l=60, r=120, t=118, b=70),
    )
    # Always configure xaxis4 on Track 1 for gauge / bit size / caliper
    layout_dict["xaxis4"] = dict(
        title=dict(text="CALI / Bit Size (in)" if has_cali else "Bit Size Ref (in)", font=dict(color="#78350f", size=10), standoff=6),
        range=[6.0, 16.0],
        tickvals=[6, 8, 10, 12, 14, 16],
        ticktext=["6", "8", "10", "12", "14", "16"],
        overlaying="x",
        side="top",
        showgrid=False,
        tickfont=dict(color="#78350f", size=9)
    )
    if has_porosity_track:
        layout_dict["xaxis5"] = dict(
            title=dict(text="NPHI (v/v)", font=dict(color="#2563eb", size=10), standoff=6),
            range=[0.45, -0.15],
            tickvals=[0.45, 0.30, 0.15, 0.0, -0.15],
            ticktext=["0.45", "0.30", "0.15", "0.00", "-0.15"],
            overlaying="x3",
            side="top",
            showgrid=False,
            tickfont=dict(color="#2563eb", size=9)
        )
    fig.update_layout(**layout_dict)
    
    return {
        "well_id": well_id,
        "top_depth": min_d,
        "bottom_depth": max_d,
        "figure_json": fig.to_json(),
        "summary": f"Generated 3-track composite log for {well_title} across {min_d:.1f}m - {max_d:.1f}m."
    }


def plot_2d_crossplot(well_id: str, x_curve: str, y_curve: str, z_curve: Optional[str] = None, top_depth: Optional[float] = None, bottom_depth: Optional[float] = None) -> Dict[str, Any]:
    """Generates an interactive 2D lithology crossplot with mineral trendlines."""
    las, df = read_las(well_id)
    
    if top_depth is not None:
        df = df[df["DEPTH"] >= top_depth]
    if bottom_depth is not None:
        df = df[df["DEPTH"] <= bottom_depth]
        
    cols = {c.upper(): c for c in df.columns}
    x_actual = cols.get(x_curve.upper())
    y_actual = cols.get(y_curve.upper())
    z_actual = cols.get(z_curve.upper()) if z_curve else None
    
    if not x_actual or not y_actual:
        raise ValueError(f"Curves '{x_curve}', '{y_curve}' not both present in {well_id}")
        
    cols_to_keep = ["DEPTH", x_actual, y_actual]
    if z_actual and z_actual in df.columns:
        cols_to_keep.append(z_actual)
        
    sub_df = df[cols_to_keep].dropna()
    if sub_df.empty:
        raise ValueError(f"No valid data points for {x_curve} vs {y_curve}")
        
    x_vals = sub_df[x_actual].values
    y_vals = sub_df[y_actual].values
    
    if "NEUT" in x_actual.upper() or "NPHI" in x_actual.upper():
        if np.nanmax(x_vals) <= 1.0:
            x_vals = x_vals * 100.0
        # Explicit axis title so readers don't need domain knowledge to parse units
        x_title = "Neutron Porosity — NPHI (%)"
        # Detect artificial quantization (e.g. integer-binned neutron porosities) and apply slight Gaussian jitter
        rounded_x = np.round(x_vals, 2)
        unique_diffs = np.diff(np.sort(np.unique(rounded_x)))
        if len(unique_diffs) > 0 and np.median(unique_diffs) >= 0.8:
            rng = np.random.default_rng(42)
            x_vals = x_vals + rng.normal(0.0, 0.22, size=len(x_vals))
    else:
        x_title = x_actual

    if "DEN" in y_actual.upper() or "RHOB" in y_actual.upper():
        # Explicit axis title so readers don't need domain knowledge to parse units
        y_title = "Bulk Density — RHOB (g/cc)"
    else:
        y_title = y_actual

    fig = go.Figure()
    # Slight transparency so dense-cluster centers read as darker than edge outliers
    marker_dict: Dict[str, Any] = dict(size=6, opacity=0.65)

    if z_actual and z_actual in sub_df.columns:
        z_vals = sub_df[z_actual].values
        marker_dict["color"] = z_vals
        marker_dict["colorscale"] = "Viridis"
        if "GR" in z_actual.upper():
            # One-line convention caption: saves readers from needing the GR/shaliness convention
            colorbar_title = f"<b>{z_actual} (API)</b><br><span style='font-size:9px'>Low GR = clean · High GR = shaly</span>"
        else:
            colorbar_title = f"<b>{z_actual}</b>"
        marker_dict["colorbar"] = dict(
            title=dict(text=colorbar_title, side="top"),
            thickness=14,
            len=0.72,
            y=0.45,
            x=1.02,
            xanchor="left"
        )
        marker_dict["showscale"] = True
        customdata = np.column_stack([sub_df["DEPTH"].values, z_vals])
        z_unit = " API" if "GR" in z_actual.upper() else ""
        hovertemplate = (
            f"<b>Depth:</b> %{{customdata[0]:.1f}} m<br>"
            f"<b>{x_actual}:</b> %{{x:.2f}}%<br>"
            f"<b>{y_actual}:</b> %{{y:.3f}} g/cc<br>"
            f"<b>{z_actual}:</b> %{{customdata[1]:.1f}}{z_unit}<extra></extra>"
        )
    else:
        marker_dict["color"] = "#2563eb"
        customdata = sub_df["DEPTH"].values
        hovertemplate = (
            f"<b>Depth:</b> %{{customdata:.1f}} m<br>"
            f"<b>{x_actual}:</b> %{{x:.2f}}%<br>"
            f"<b>{y_actual}:</b> %{{y:.3f}} g/cc<extra></extra>"
        )

    fig.add_trace(
        go.Scatter(
            x=x_vals, y=y_vals, mode="markers", marker=marker_dict,
            customdata=customdata,
            hovertemplate=hovertemplate,
            name=f"{y_actual} vs {x_actual}"
        )
    )
    
    annotations = []
    is_rhob_nphi = ("DEN" in y_actual.upper() or "RHOB" in y_actual.upper()) and ("NEUT" in x_actual.upper() or "NPHI" in x_actual.upper())
    if is_rhob_nphi:
        fig.update_yaxes(autorange="reversed")
        phi_pts = np.array([0, 10, 20, 30, 40])
        fig.add_trace(go.Scatter(x=phi_pts, y=2.65 - (phi_pts / 100.0) * 1.65, mode="lines+markers", name="Sandstone (2.65)", line=dict(color="#eab308", dash="dash")))
        fig.add_trace(go.Scatter(x=phi_pts, y=2.71 - (phi_pts / 100.0) * 1.71, mode="lines+markers", name="Limestone (2.71)", line=dict(color="#0284c7", dash="dash")))
        fig.add_trace(go.Scatter(x=phi_pts, y=2.87 - (phi_pts / 100.0) * 1.87, mode="lines+markers", name="Dolomite (2.87)", line=dict(color="#dc2626", dash="dash")))

        # Endmember callout annotations at zero porosity
        annotations = [
            dict(x=0, y=2.65, text="<b>Quartz (2.65)</b>", showarrow=True, arrowhead=2, ax=-55, ay=-12, font=dict(color="#b45309", size=9), bgcolor="rgba(254, 243, 199, 0.9)", bordercolor="#f59e0b", borderwidth=1),
            dict(x=0, y=2.71, text="<b>Calcite (2.71)</b>", showarrow=True, arrowhead=2, ax=-55, ay=2, font=dict(color="#0369a1", size=9), bgcolor="rgba(224, 242, 254, 0.9)", bordercolor="#0284c7", borderwidth=1),
            dict(x=0, y=2.87, text="<b>Dolomite (2.87)</b>", showarrow=True, arrowhead=2, ax=-55, ay=15, font=dict(color="#b91c1c", size=9), bgcolor="rgba(254, 226, 226, 0.9)", bordercolor="#ef4444", borderwidth=1),
        ]

        # Gas Correction Vector: park the text box in the petrophysically-empty
        # top-right corner (high NPHI + low RHOB is unphysical, so always whitespace)
        # with a thin leader line back into the gas cluster — never over the points.
        gas_mask = (x_vals < 10.0) & (y_vals < 2.42) & (y_vals > 2.10)
        if np.sum(gas_mask) >= 6:
            mean_gas_x = float(np.mean(x_vals[gas_mask]))
            mean_gas_y = float(np.mean(y_vals[gas_mask]))
            label_x = float(np.nanpercentile(x_vals, 96))
            # On the reversed y-axis, smaller y renders higher on screen ("up")
            label_y = float(min(mean_gas_y - 0.20, np.nanpercentile(y_vals, 6)))
            annotations.append(
                dict(
                    x=label_x,
                    y=label_y,
                    xref="x",
                    yref="y",
                    ax=mean_gas_x,
                    ay=mean_gas_y,
                    axref="x",
                    ayref="y",
                    text="<b>Gas Correction Vector</b><br><i>(Hydrocarbon crossover shift)</i>",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.0,
                    arrowwidth=1.2,
                    arrowcolor="#dc2626",
                    font=dict(color="#b91c1c", size=9, family="Inter, sans-serif"),
                    bgcolor="rgba(254, 242, 242, 0.95)",
                    bordercolor="#f87171",
                    borderwidth=1.2,
                    borderpad=4
                )
            )

        # Second rock population: high-NPHI + high-RHOB (+ high GR when available)
        # renders lower-right on screen — label it so the story reads at a glance.
        shale_mask = (x_vals >= float(np.nanpercentile(x_vals, 60))) & \
                     (y_vals >= float(np.nanpercentile(y_vals, 55)))
        if z_actual and "GR" in z_actual.upper() and "z_vals" in locals():
            shale_mask = shale_mask & (z_vals >= float(np.nanmedian(z_vals)))
        if np.sum(shale_mask) >= 8:
            sh_x = float(np.nanmedian(x_vals[shale_mask]))
            sh_y = float(np.nanmedian(y_vals[shale_mask]))
            annotations.append(
                dict(
                    x=sh_x,
                    y=sh_y,
                    text="<b>Shaly interval</b><br><i>(sand-shale mix)</i>",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.0,
                    arrowwidth=1.2,
                    arrowcolor="#0d9488",
                    ax=-80,
                    ay=-25,
                    font=dict(color="#0f766e", size=9, family="Inter, sans-serif"),
                    bgcolor="rgba(240, 253, 250, 0.92)",
                    bordercolor="#14b8a6",
                    borderwidth=1.2,
                    borderpad=4
                )
            )

    well_title = las.well.WELL.value if "WELL" in las.well else well_id
    if is_rhob_nphi:
        title_text = f"<b>Density–Neutron Crossplot: Lithology & Gas Effect</b> ({well_title})"
    else:
        title_text = f"<b>2D Lithology Crossplot: {y_actual} vs. {x_actual}</b> ({well_title})"
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.02,
            xanchor="left"
        ),
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        height=620,
        annotations=annotations,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=10)
        ),
        margin=dict(l=65, r=110, t=80, b=55),
    )
    
    return {
        "well_id": well_id,
        "figure_json": fig.to_json(),
        "data_points": len(sub_df),
        "summary": f"Generated 2D crossplot with {len(sub_df)} points for {well_title}."
    }


def compute_net_pay(
    well_id: str,
    top_depth: float,
    bottom_depth: float,
    vsh_cutoff: float = 0.3,
    phi_cutoff: float = 0.1,
    sw_cutoff: float = 0.5
) -> Dict[str, Any]:
    """Runs volumetric cutoff logic across the depth array."""
    las, df = read_las(well_id)
    sub_df = df[(df["DEPTH"] >= top_depth) & (df["DEPTH"] <= bottom_depth)].copy()
    if sub_df.empty:
        raise ValueError(f"No log samples found between {top_depth}m and {bottom_depth}m for {well_id}")
        
    sub_df = sub_df.sort_values("DEPTH")
    depths = sub_df["DEPTH"].values
    med_step = float(np.median(np.diff(depths))) if len(depths) > 1 else 0.1524
    cols = {c.upper(): c for c in sub_df.columns}
    
    # 1. Vsh (track the method for the audit footnote)
    if "VSHALE" in cols:
        vsh = sub_df[cols["VSHALE"]].values
        vsh_method = f"Vsh from {cols['VSHALE']} log direct"
    elif "GR" in cols:
        gr = sub_df[cols["GR"]].values
        gr_clean = float(np.nanpercentile(gr, 5)) if np.sum(~np.isnan(gr)) > 10 else 20.0
        gr_shale = float(np.nanpercentile(gr, 95)) if np.sum(~np.isnan(gr)) > 10 else 120.0
        igr = np.clip((gr - gr_clean) / max(gr_shale - gr_clean, 1.0), 0.0, 1.0)
        vsh = 0.083 * (np.power(2.0, 3.7 * igr) - 1.0)
        vsh_method = (f"Vsh from GR via Larionov-T "
                      f"(GRclean P5={gr_clean:.0f} / GRshale P95={gr_shale:.0f} API)")
    else:
        vsh = np.zeros(len(depths))
        vsh_method = "Vsh = 0 assumed (no GR/VSHALE)"

    # 2. Porosity
    if "PHIE" in cols:
        phi = sub_df[cols["PHIE"]].values
        phi_method = f"Porosity from {cols['PHIE']} log direct"
    elif "DENB" in cols or "RHOB" in cols:
        den_col = cols.get("DENB") or cols.get("RHOB")
        phi = np.clip((2.65 - sub_df[den_col].values) / (2.65 - 1.0), 0.0, 0.45)
        phi_method = f"Porosity from density ({den_col}, rho_ma 2.65 / rho_f 1.0)"
    else:
        phi = np.full(len(depths), 0.15)
        phi_method = "Porosity = 0.15 assumed (no PHIE/DENB)"

    # 3. Water Saturation
    if "SWE" in cols:
        sw = sub_df[cols["SWE"]].values
        sw_method = f"Sw from {cols['SWE']} log direct"
    elif "RDEEP" in cols:
        rdeep = np.maximum(sub_df[cols["RDEEP"]].values, 0.1)
        sw = np.sqrt(np.clip((1.0 * 0.05) / (np.maximum(phi, 0.01)**2.0 * rdeep), 0.0, 1.0))
        sw_method = "Sw via Archie-type (a·Rw 0.05, m = n = 2)"
    else:
        sw = np.ones(len(depths))
        sw_method = "Sw = 1 assumed (no SWE/RDEEP)"

    valid = (~np.isnan(depths)) & (~np.isnan(vsh)) & (~np.isnan(phi)) & (~np.isnan(sw))
    is_res = (vsh <= vsh_cutoff) & (phi >= phi_cutoff) & valid
    is_pay = is_res & (sw <= sw_cutoff) & valid
    
    gross = float(depths[-1] - depths[0]) if len(depths) > 1 else med_step
    net_res = float(np.sum(is_res) * med_step)
    net_pay = float(np.sum(is_pay) * med_step)
    wet_res_m = float(np.sum(is_res & (~is_pay)) * med_step)
    non_res_m = max(0.0, gross - net_res)
    
    ntg = round(net_pay / gross, 4) if gross > 0 else 0.0
    res_to_gross = round(net_res / gross, 4) if gross > 0 else 0.0

    # Cumulative Pay profile vs depth
    cum_pay = np.cumsum(is_pay.astype(float) * med_step)
    sample_stride = max(1, len(depths) // 80)
    cum_pay_curve = [
        {"depth": round(float(depths[i]), 2), "cum_pay_m": round(float(cum_pay[i]), 2)}
        for i in range(0, len(depths), sample_stride)
    ]
    if len(depths) > 0 and (len(depths) - 1) % sample_stride != 0:
        cum_pay_curve.append({"depth": round(float(depths[-1]), 2), "cum_pay_m": round(float(cum_pay[-1]), 2)})

    # Facies breakdown
    facies_breakdown = {
        "pay_m": round(net_pay, 2),
        "pay_pct": round((net_pay / gross) * 100.0, 1) if gross > 0 else 0.0,
        "wet_reservoir_m": round(wet_res_m, 2),
        "wet_reservoir_pct": round((wet_res_m / gross) * 100.0, 1) if gross > 0 else 0.0,
        "non_reservoir_m": round(non_res_m, 2),
        "non_reservoir_pct": round((non_res_m / gross) * 100.0, 1) if gross > 0 else 0.0
    }

    # Cutoff Sensitivity Grid (Vsh in [0.2, 0.3, 0.4] x Phi in [0.08, 0.10, 0.12])
    cutoff_sensitivity = []
    for v_cut in [0.20, 0.30, 0.40]:
        for p_cut in [0.08, 0.10, 0.12]:
            s_res = (vsh <= v_cut) & (phi >= p_cut) & valid
            s_pay = s_res & (sw <= sw_cutoff) & valid
            s_pay_m = float(np.sum(s_pay) * med_step)
            cutoff_sensitivity.append({
                "vsh_cutoff": v_cut,
                "phi_cutoff": p_cut,
                "sw_cutoff": sw_cutoff,
                "net_pay_m": round(s_pay_m, 2),
                "net_to_gross": round(s_pay_m / gross, 4) if gross > 0 else 0.0
            })

    # Deterministic uncertainty from the cutoff ensemble spread:
    # P50 = base case, P90 = conservative (10th pct), P10 = optimistic (90th pct).
    sens_vals = np.array([c["net_pay_m"] for c in cutoff_sensitivity], dtype=float)
    p90_m = round(float(np.percentile(sens_vals, 10)), 2)
    p50_m = round(net_pay, 2)
    p10_m = round(float(np.percentile(sens_vals, 90)), 2)
    net_pay_uncertainty = {
        "p90_m": p90_m,
        "p50_m": p50_m,
        "p10_m": p10_m,
        "plus_minus_m": round((p10_m - p90_m) / 2.0, 2),
        "basis": (f"P90–P10 spread across 3×3 Vsh/Phi cutoff ensemble "
                  f"(Vsh 0.20–0.40, Phi 0.08–0.12, Sw ≤ {sw_cutoff:g})")
    }

    methodology = (f"{vsh_method}; {phi_method}; {sw_method}. "
                   f"Cutoffs Vsh ≤ {vsh_cutoff:g}, Phi ≥ {phi_cutoff:g}, Sw ≤ {sw_cutoff:g}.")

    return {
        "well_id": well_id,
        "well_name": las.well.WELL.value if "WELL" in las.well else well_id,
        "top_depth": round(float(top_depth), 2),
        "bottom_depth": round(float(bottom_depth), 2),
        "gross_interval_m": round(gross, 2),
        "net_reservoir_m": round(net_res, 2),
        "net_pay_m": round(net_pay, 2),
        "net_to_gross": ntg,
        "reservoir_to_gross": res_to_gross,
        "pay_zone_averages": {
            "average_porosity": round(float(np.nanmean(phi[is_pay])) if np.any(is_pay) else 0.0, 4),
            "average_water_saturation": round(float(np.nanmean(sw[is_pay])) if np.any(is_pay) else 0.0, 4),
            "average_shale_volume": round(float(np.nanmean(vsh[is_pay])) if np.any(is_pay) else 0.0, 4),
        },
        "cum_pay_curve": cum_pay_curve,
        "facies_breakdown": facies_breakdown,
        "cutoff_sensitivity": cutoff_sensitivity,
        "net_pay_uncertainty": net_pay_uncertainty,
        "methodology": methodology
    }


# ==========================================
# 4 NEW ADVANCED PETROPHYSICAL TOOLS
# ==========================================

def compare_vshale_methods(well_id: str, top_depth: float, bottom_depth: float) -> Dict[str, Any]:
    """Compares 4 shale volume methods: Linear, Larionov (Tertiary), Steiber, and Clavier."""
    las, df = read_las(well_id)
    sub = df[(df["DEPTH"] >= top_depth) & (df["DEPTH"] <= bottom_depth)].dropna(subset=["DEPTH", "GR"]).copy()
    if sub.empty:
        raise ValueError(f"No valid Gamma Ray data in {top_depth}m - {bottom_depth}m for {well_id}")
        
    depths = sub["DEPTH"].values
    gr = sub["GR"].values
    gr_min = float(np.percentile(gr, 2))
    gr_max = float(np.percentile(gr, 98))
    if gr_max == gr_min:
        gr_max += 1.0
        
    # 1. Linear Index (I_GR)
    igr = np.clip((gr - gr_min) / (gr_max - gr_min), 0.0, 1.0)
    
    # 2. Larionov for Tertiary Rocks
    vsh_larionov = np.clip(0.083 * (np.power(2.0, 3.7 * igr) - 1.0), 0.0, 1.0)
    
    # 3. Steiber Method
    vsh_steiber = np.clip(igr / (3.0 - 2.0 * igr), 0.0, 1.0)
    
    # 4. Clavier Method
    vsh_clavier = np.clip(1.7 - np.sqrt(np.maximum(0.0, 3.38 - np.square(igr + 0.7))), 0.0, 1.0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=igr, y=depths, name="Linear (I_GR)", line=dict(color="#64748b", dash="dash", width=1.3)))
    fig.add_trace(go.Scatter(x=vsh_larionov, y=depths, name="Larionov (Tertiary)", line=dict(color="#2563eb", width=1.6)))
    fig.add_trace(go.Scatter(x=vsh_steiber, y=depths, name="Steiber", line=dict(color="#059669", width=1.5)))
    fig.add_trace(go.Scatter(x=vsh_clavier, y=depths, name="Clavier", line=dict(color="#d97706", width=1.5)))
    
    # Shading clean sand zone (<0.3)
    fig.add_vline(x=0.3, line_width=1, line_dash="dot", line_color="#ef4444", annotation_text="0.30 Cutoff")
    
    fig.update_yaxes(autorange="reversed", title_text="Depth (m)")
    fig.update_xaxes(title_text="Shale Volume (fraction)", range=[0, 1.0])
    fig.update_layout(
        title=f"<b>Shale Volume Model Comparison: {well_id}</b> ({top_depth:.1f}m - {bottom_depth:.1f}m)",
        template="plotly_white",
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=55, r=25, t=60, b=65),
    )
    
    return {
        "well_id": well_id,
        "top_depth": top_depth,
        "bottom_depth": bottom_depth,
        "figure_json": fig.to_json(),
        "averages": {
            "linear_avg": round(float(np.mean(igr)), 4),
            "larionov_avg": round(float(np.mean(vsh_larionov)), 4),
            "steiber_avg": round(float(np.mean(vsh_steiber)), 4),
            "clavier_avg": round(float(np.mean(vsh_clavier)), 4),
        }
    }


def calculate_archie_saturation(
    well_id: str,
    top_depth: float,
    bottom_depth: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    rw: float = 0.05
) -> Dict[str, Any]:
    """Calculates continuous Archie water saturation (Sw) and Bulk Volume Hydrocarbon (BVH)."""
    las, df = read_las(well_id)
    sub = df[(df["DEPTH"] >= top_depth) & (df["DEPTH"] <= bottom_depth)].copy()
    if sub.empty:
        raise ValueError(f"No samples in interval {top_depth}m - {bottom_depth}m")
        
    depths = sub["DEPTH"].values
    cols = {c.upper(): c for c in sub.columns}
    
    # Porosity
    if "PHIE" in cols:
        phi = sub[cols["PHIE"]].values
    elif "DENB" in cols:
        phi = np.clip((2.65 - sub[cols["DENB"]].values) / 1.65, 0.0, 0.45)
    else:
        phi = np.full(len(depths), 0.20)
        
    # Resistivity
    r_col = cols.get("RDEEP") or cols.get("ILD") or cols.get("RT")
    rt = sub[r_col].values if r_col else np.full(len(depths), 10.0)
    rt = np.maximum(rt, 0.1)
    phi_safe = np.maximum(phi, 0.01)
    
    # Archie: Sw = ((a * Rw) / (phi^m * Rt))^(1/n)
    sw = np.power(np.clip((a * rw) / (np.power(phi_safe, m) * rt), 0.0, 1.0), 1.0 / n)
    shc = 1.0 - sw
    bvh = phi * shc  # Bulk Volume Hydrocarbon
    
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=["Water vs Hydrocarbon Saturation", "Bulk Volume Hydrocarbon (BVH)"])
    
    # Track 1: Saturation
    fig.add_trace(go.Scatter(x=sw * 100.0, y=depths, name="Sw (%)", line=dict(color="#2563eb", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=shc * 100.0, y=depths, name="So (%)", line=dict(color="#16a34a", width=1.4)), row=1, col=1)
    
    # Track 2: BVH
    fig.add_trace(go.Scatter(x=bvh, y=depths, name="BVH (frac)", fill="tozerox", fillcolor="rgba(22, 163, 74, 0.3)", line=dict(color="#15803d", width=1.3)), row=1, col=2)
    
    fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1)
    fig.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=2)
    fig.update_xaxes(title_text="Saturation (%)", range=[0, 100], row=1, col=1)
    fig.update_xaxes(title_text="BVH (v/v)", range=[0, 0.25], row=1, col=2)
    
    fig.update_layout(
        title=f"<b>Archie Saturation & Hydrocarbon Volume: {well_id}</b> (Rw={rw}, m={m}, n={n})",
        template="plotly_white",
        height=620,
        margin=dict(l=55, r=25, t=60, b=50),
    )
    
    return {
        "well_id": well_id,
        "figure_json": fig.to_json(),
        "average_sw": round(float(np.nanmean(sw)), 4),
        "average_shc": round(float(np.nanmean(shc)), 4),
        "average_bvh": round(float(np.nanmean(bvh)), 4),
        "max_bvh": round(float(np.nanmax(bvh)), 4)
    }


def compute_sonic_porosity_wyllie(
    well_id: str,
    top_depth: float,
    bottom_depth: float,
    dt_matrix: float = 55.5,
    dt_fluid: float = 189.0
) -> Dict[str, Any]:
    """Computes Wyllie time-average sonic porosity from DTCOMP and compares against density porosity."""
    las, df = read_las(well_id)
    cols = {c.upper(): c for c in df.columns}
    dt_col = cols.get("DTCOMP") or cols.get("DT")
    if not dt_col:
        raise ValueError(f"Compressional sonic curve (DTCOMP/DT) not found in {well_id}")
        
    sub = df[(df["DEPTH"] >= top_depth) & (df["DEPTH"] <= bottom_depth)].dropna(subset=["DEPTH", dt_col]).copy()
    depths = sub["DEPTH"].values
    dt = sub[dt_col].values
    
    # Wyllie Equation: Phi_S = (DT - DT_ma) / (DT_fl - DT_ma)
    phi_sonic = np.clip((dt - dt_matrix) / (dt_fluid - dt_matrix), 0.0, 0.45)
    
    # Density porosity for comparison
    den_col = cols.get("DENB") or cols.get("RHOB")
    phi_density = np.clip((2.65 - sub[den_col].values) / 1.65, 0.0, 0.45) if den_col else np.full(len(depths), np.nan)
    
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=["Compressional Sonic (DTCOMP)", "Sonic vs. Density Porosity"])
    fig.add_trace(go.Scatter(x=dt, y=depths, name="DTCOMP (us/ft)", line=dict(color="#0f172a", width=1.3)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=phi_sonic * 100.0, y=depths, name="Sonic Porosity (%)", line=dict(color="#0284c7", width=1.4)), row=1, col=2)
    if not np.all(np.isnan(phi_density)):
        fig.add_trace(go.Scatter(x=phi_density * 100.0, y=depths, name="Density Porosity (%)", line=dict(color="#dc2626", dash="dot", width=1.4)), row=1, col=2)
        
    fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1)
    fig.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=2)
    fig.update_xaxes(title_text="DT (us/ft)", range=[40, 140], row=1, col=1)
    fig.update_xaxes(title_text="Porosity (%)", range=[0, 40], row=1, col=2)
    
    fig.update_layout(
        title=f"<b>Sonic Petrophysics: Wyllie Porosity ({well_id})</b>",
        template="plotly_white",
        height=620,
        margin=dict(l=55, r=25, t=60, b=50),
    )
    
    return {
        "well_id": well_id,
        "figure_json": fig.to_json(),
        "avg_sonic_porosity": round(float(np.nanmean(phi_sonic)), 4),
        "dt_matrix": dt_matrix,
        "dt_fluid": dt_fluid
    }


def scan_reservoir_sweetspots(well_id: str, min_thickness: float = 1.5) -> Dict[str, Any]:
    """Scans the entire well depth array to automatically delineate prospective hydrocarbon sweet spots."""
    las, df = read_las(well_id)
    cols = {c.upper(): c for c in df.columns}
    
    depths = df["DEPTH"].values
    med_step = float(np.median(np.diff(depths))) if len(depths) > 1 else 0.1524
    
    # Derive Vsh, Phi, Sw across entire well
    if "VSHALE" in cols:
        vsh = df[cols["VSHALE"]].values
    elif "GR" in cols:
        gr = df[cols["GR"]].values
        vsh = np.clip((gr - 25.0) / 100.0, 0.0, 1.0)
    else:
        vsh = np.zeros(len(depths))

    if "PHIE" in cols:
        phi = df[cols["PHIE"]].values
    elif "DENB" in cols:
        phi = np.clip((2.65 - df[cols["DENB"]].values) / 1.65, 0.0, 0.40)
    else:
        phi = np.full(len(depths), 0.10)

    if "SWE" in cols:
        sw = df[cols["SWE"]].values
    elif "RDEEP" in cols:
        rdeep = np.maximum(df[cols["RDEEP"]].values, 0.1)
        sw = np.sqrt(np.clip(0.05 / (np.maximum(phi, 0.01)**2.0 * rdeep), 0.0, 1.0))
    else:
        sw = np.ones(len(depths))

    # Pay criteria: Vsh <= 0.3, Phi >= 0.1, Sw <= 0.5
    is_pay = (vsh <= 0.3) & (phi >= 0.10) & (sw <= 0.50) & (~np.isnan(depths)) & (~np.isnan(vsh)) & (~np.isnan(phi)) & (~np.isnan(sw))
    
    # Identify contiguous zones
    zones = []
    in_zone = False
    start_idx = 0
    
    for i in range(len(is_pay)):
        if is_pay[i] and not in_zone:
            in_zone = True
            start_idx = i
        elif not is_pay[i] and in_zone:
            in_zone = False
            thickness = (i - start_idx) * med_step
            if thickness >= min_thickness:
                z_depths = depths[start_idx:i]
                z_phi = phi[start_idx:i]
                z_sw = sw[start_idx:i]
                z_vsh = vsh[start_idx:i]
                avg_p = float(np.mean(z_phi))
                avg_s = float(np.mean(z_sw))
                hcpv = thickness * avg_p * (1.0 - avg_s)
                zones.append({
                    "zone_name": f"Sweet Spot {len(zones) + 1}",
                    "top_depth": round(float(z_depths[0]), 2),
                    "base_depth": round(float(z_depths[-1]), 2),
                    "thickness_m": round(thickness, 2),
                    "avg_porosity": round(avg_p, 4),
                    "avg_sw": round(avg_s, 4),
                    "avg_vsh": round(float(np.mean(z_vsh)), 4),
                    "hcpv_index": round(hcpv, 3)
                })

    # Sort zones by hydrocarbon potential
    zones.sort(key=lambda x: x["hcpv_index"], reverse=True)
    for idx, z in enumerate(zones):
        z["rank"] = idx + 1
        z["zone_name"] = f"Rank #{idx + 1}: Zone {z['top_depth']}m - {z['base_depth']}m ({z['thickness_m']}m Pay)"

    return {
        "well_id": well_id,
        "total_sweetspots_found": len(zones),
        "total_pay_thickness_m": round(sum(z["thickness_m"] for z in zones), 2),
        "sweetspots": zones
    }


def compute_permeability_timur_coates(
    well_id: str,
    top_depth: float,
    bottom_depth: float,
    model: str = "timur"
) -> Dict[str, Any]:
    """Calculates continuous permeability (k) in mD and flow capacity (k*h) using Timur or Coates model."""
    las, df = read_las(well_id)
    sub = df[(df["DEPTH"] >= top_depth) & (df["DEPTH"] <= bottom_depth)].copy()
    if sub.empty:
        raise ValueError(f"No samples in interval {top_depth}m - {bottom_depth}m")
        
    depths = sub["DEPTH"].values
    cols = {c.upper(): c for c in sub.columns}
    med_step = float(np.median(np.diff(depths))) if len(depths) > 1 else 0.1524
    
    # Porosity
    if "PHIE" in cols:
        phi = sub[cols["PHIE"]].values
    elif "DENB" in cols:
        phi = np.clip((2.65 - sub[cols["DENB"]].values) / 1.65, 0.0, 0.45)
    else:
        phi = np.full(len(depths), 0.20)
        
    # Saturation
    if "SWE" in cols:
        sw = sub[cols["SWE"]].values
    elif "RDEEP" in cols:
        rdeep = np.maximum(sub[cols["RDEEP"]].values, 0.1)
        sw = np.sqrt(np.clip(0.05 / (np.maximum(phi, 0.01)**2.0 * rdeep), 0.0, 1.0))
    else:
        sw = np.full(len(depths), 0.40)
        
    phi_safe = np.clip(phi, 0.01, 0.45)
    swirr = np.clip(np.minimum(sw, 0.05 / phi_safe), 0.05, 0.95)
    
    if model.lower() == "coates":
        k_md = np.power(100.0 * np.square(phi_safe) * ((1.0 - swirr) / swirr), 2.0)
    else:
        k_md = 0.136 * (np.power(phi_safe * 100.0, 4.4)) / np.square(swirr * 100.0)
        
    k_md = np.clip(k_md, 0.001, 20000.0)
    kh_total = float(np.nansum(k_md * med_step))
    avg_k = float(np.nanmean(k_md))
    max_k = float(np.nanmax(k_md))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k_md, y=depths, name=f"Permeability ({model.capitalize()})", line=dict(color="#0284c7", width=1.5)))
    
    fig.add_vline(x=1.0, line_width=1, line_dash="dot", line_color="#ef4444", annotation_text="1 mD (Tight)")
    fig.add_vline(x=10.0, line_width=1, line_dash="dash", line_color="#f59e0b", annotation_text="10 mD (Fair)")
    fig.add_vline(x=100.0, line_width=1, line_dash="solid", line_color="#10b981", annotation_text="100 mD (Good)")
    
    fig.update_yaxes(autorange="reversed", title_text="Depth (m)")
    fig.update_xaxes(type="log", title_text="Intrinsic Permeability (mD)", range=[-2, 4])
    fig.update_layout(
        title=f"<b>Permeability & Reservoir Flow Profile: {well_id}</b> ({model.capitalize()} Model)",
        template="plotly_white",
        height=620,
        margin=dict(l=55, r=25, t=60, b=50),
    )
    
    return {
        "well_id": well_id,
        "model": model,
        "top_depth": top_depth,
        "bottom_depth": bottom_depth,
        "figure_json": fig.to_json(),
        "average_permeability_md": round(avg_k, 2),
        "max_permeability_md": round(max_k, 2),
        "flow_capacity_kh_md_m": round(kh_total, 2),
        "reservoir_quality_class": "Excellent (>100 mD)" if avg_k > 100 else ("Good (10-100 mD)" if avg_k > 10 else "Fair/Tight (<10 mD)")
    }


def plot_crossplot_picket(
    well_id: str,
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None,
    rw: float = 0.05,
    m: float = 2.0,
    n: float = 2.0
) -> Dict[str, Any]:
    """Generates a classic Archie Picket Plot (log(Rt) vs log(Phi)) with iso-saturation trendlines."""
    las, df = read_las(well_id)
    cols = {c.upper(): c for c in df.columns}
    
    sub = df.copy()
    if top_depth is not None:
        sub = sub[sub["DEPTH"] >= top_depth]
    if bottom_depth is not None:
        sub = sub[sub["DEPTH"] <= bottom_depth]
        
    depths = sub["DEPTH"].values
    
    if "PHIE" in cols:
        phi = sub[cols["PHIE"]].values
    elif "DENB" in cols:
        phi = np.clip((2.65 - sub[cols["DENB"]].values) / 1.65, 0.01, 0.45)
    else:
        phi = np.full(len(depths), 0.20)
        
    r_col = cols.get("RDEEP") or cols.get("ILD") or cols.get("RT")
    rt = sub[r_col].values if r_col else np.full(len(depths), 10.0)
    
    gr_col = cols.get("GR")
    gr = sub[gr_col].values if gr_col else np.full(len(depths), 50.0)
    
    valid = (phi > 0.01) & (rt > 0.1) & (~np.isnan(phi)) & (~np.isnan(rt))
    phi_v = phi[valid]
    rt_v = rt[valid]
    gr_v = gr[valid]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rt_v,
        y=phi_v,
        mode="markers",
        marker=dict(
            size=5,
            color=gr_v,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="GR (API)")
        ),
        name="Well Log Samples"
    ))
    
    phi_line = np.linspace(0.02, 0.40, 50)
    for sw_val, color, dash in [(1.0, "#dc2626", "solid"), (0.50, "#f59e0b", "dash"), (0.25, "#16a34a", "dot")]:
        rt_line = (1.0 * rw) / (np.power(phi_line, m) * np.power(sw_val, n))
        fig.add_trace(go.Scatter(
            x=rt_line,
            y=phi_line,
            mode="lines",
            line=dict(color=color, dash=dash, width=1.8),
            name=f"Sw = {int(sw_val*100)}%"
        ))
        
    fig.update_xaxes(type="log", title_text="True Formation Resistivity Rt (ohm.m)", range=[-1, 3])
    fig.update_yaxes(type="log", title_text="Effective Porosity Phi (v/v)", range=[-2, -0.3])
    fig.update_layout(
        title=f"<b>Archie Picket Plot: {well_id}</b> (Rw={rw} ohm.m, m={m}, n={n})",
        template="plotly_white",
        height=620,
        margin=dict(l=55, r=25, t=60, b=50),
    )
    
    return {
        "well_id": well_id,
        "rw": rw,
        "m": m,
        "n": n,
        "samples_plotted": int(np.sum(valid)),
        "figure_json": fig.to_json()
    }


def generate_reservoir_composite_report(
    well_id: str,
    top_depth: float,
    bottom_depth: float
) -> Dict[str, Any]:
    """Generates an all-in-one zonal petrophysical dossier combining cutoffs, saturations, and flow capacity."""
    net_pay_data = compute_net_pay(well_id, top_depth, bottom_depth)
    archie_data = calculate_archie_saturation(well_id, top_depth, bottom_depth)
    perm_data = compute_permeability_timur_coates(well_id, top_depth, bottom_depth)
    
    gross_h = net_pay_data["gross_interval_m"]
    net_pay_h = net_pay_data["net_pay_m"]
    ntg = net_pay_data["net_to_gross"]
    avg_phi = net_pay_data["pay_zone_averages"]["average_porosity"]
    avg_sw = net_pay_data["pay_zone_averages"]["average_water_saturation"]
    avg_vsh = net_pay_data["pay_zone_averages"]["average_shale_volume"]
    
    hcpv = round(net_pay_h * avg_phi * (1.0 - avg_sw), 3)
    fluid_type = "Gas Sand (High Resistivity & Crossover)" if (avg_sw < 0.35 and avg_phi > 0.15) else ("Hydrocarbon Sand" if avg_sw < 0.50 else "Water Sand / Wet Formation")
    
    return {
        "well_id": well_id,
        "reservoir_zone": f"{top_depth:.1f}m - {bottom_depth:.1f}m",
        "gross_thickness_m": gross_h,
        "net_reservoir_m": net_pay_data["net_reservoir_m"],
        "net_pay_m": net_pay_h,
        "net_to_gross": ntg,
        "average_porosity": avg_phi,
        "average_water_saturation": avg_sw,
        "average_hydrocarbon_saturation": round(1.0 - avg_sw, 4),
        "average_shale_volume": avg_vsh,
        "hydrocarbon_pore_volume_hcpv_m": hcpv,
        "flow_capacity_kh_md_m": perm_data["flow_capacity_kh_md_m"],
        "average_permeability_md": perm_data["average_permeability_md"],
        "reservoir_quality_classification": perm_data["reservoir_quality_class"],
        "interpreted_fluid_regime": fluid_type
    }


def plot_3d_petrophysical_cube(
    well_id: str,
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None
) -> Dict[str, Any]:
    """Generates an advanced 3D Petrophysical Cluster Space:
    - X-axis: Neutron Porosity (NPHI/NEUT, v/v)
    - Y-axis: Bulk Density (RHOB/DENB, g/cc, reversed)
    - Z-axis: Compressional Slowness (DT/DTCOMP, µs/ft)
    - 3D Mineral Matrix Surfaces (Rhomb/Triangle planes & trend lines)
    - Projected 2D "Shadow" contours/scatter on Floor and Walls
    - Isosurface Envelopes (Volumetric density shells: 50% core hull & 80% containment shell)
    - Discrete Lithofacies Traces (Toggleable in legend)
    - Detailed Hover Readout (Depth, Lithology, Phie, DENB, NEUT, DT, GR, Rt)
    """
    las, df = read_las(well_id)
    cols = {c.upper(): c for c in df.columns}
    sub = df.copy()
    if top_depth is not None:
        sub = sub[sub["DEPTH"] >= top_depth]
    if bottom_depth is not None:
        sub = sub[sub["DEPTH"] <= bottom_depth]
    
    depths = sub["DEPTH"].values
    if len(depths) == 0:
        raise ValueError("No data samples available in selected depth range.")
        
    neut_col = cols.get("NEUT") or cols.get("NPHI") or cols.get("CNC")
    denb_col = cols.get("DENB") or cols.get("RHOB") or cols.get("RHOZ")
    dt_col = cols.get("DTCOMP") or cols.get("DT") or cols.get("DTC")
    gr_col = cols.get("GR")
    r_col = cols.get("RDEEP") or cols.get("ILD") or cols.get("RT")
    phie_col = cols.get("PHIE")
    
    neut = sub[neut_col].values if neut_col else np.full(len(depths), 0.20)
    denb = sub[denb_col].values if denb_col else np.full(len(depths), 2.45)
    dt = sub[dt_col].values if dt_col else np.full(len(depths), 85.0)
    gr = sub[gr_col].values if gr_col else np.full(len(depths), 65.0)
    rt = sub[r_col].values if r_col else np.full(len(depths), 5.0)
    rt = np.where(np.isnan(rt) | (rt <= 0), 5.0, rt)
    
    if phie_col:
        phie = sub[phie_col].values
    else:
        phie = np.clip((2.65 - denb) / 1.65, 0.0, 0.45)
        
    valid = (~np.isnan(neut)) & (~np.isnan(denb)) & (~np.isnan(dt)) & (~np.isnan(gr)) & (denb > 1.2) & (denb < 3.2)
    v_depths = depths[valid]
    v_neut = neut[valid]
    v_denb = denb[valid]
    v_dt = dt[valid]
    v_gr = gr[valid]
    v_rt = rt[valid]
    v_phie = phie[valid]
    
    if len(v_depths) == 0:
        raise ValueError("No valid overlapping petrophysical samples in depth range.")

    p05_x, p995_x = float(np.nanpercentile(v_neut, 0.5)), float(np.nanpercentile(v_neut, 99.5))
    min_x = min(0.0, max(-0.02, p05_x - 0.02))
    max_x = min(0.46, max(0.36, p995_x + 0.02))

    p05_y, p995_y = float(np.nanpercentile(v_denb, 0.5)), float(np.nanpercentile(v_denb, 99.5))
    min_y = max(2.05, p05_y - 0.04)
    max_y = min(2.92, max(2.88, p995_y + 0.02))

    p05_z, p99_z = float(np.nanpercentile(v_dt, 0.5)), float(np.nanpercentile(v_dt, 99.0))
    min_z = min(43.0, max(40.0, p05_z - 3.0))
    max_z = min(160.0, max(120.0, p99_z + 6.0))

    floor_z = min_z - 1.5
    back_y = max_y + 0.02
    side_x = max_x + 0.015

    fig = go.Figure()

    # 1. Projected 2D "Shadow" Scatter on Floor and Walls (High Contrast)
    fig.add_trace(go.Scatter3d(
        x=v_neut,
        y=v_denb,
        z=np.full_like(v_neut, floor_z),
        mode="markers",
        marker=dict(size=3.0, color="rgba(30, 41, 59, 0.65)", symbol="circle"),
        hoverinfo="none",
        name="Floor: Density-Neutron Shadow",
        legendgroup="shadows"
    ))
    fig.add_trace(go.Scatter3d(
        x=v_neut,
        y=np.full_like(v_neut, back_y),
        z=v_dt,
        mode="markers",
        marker=dict(size=3.0, color="rgba(51, 65, 85, 0.60)", symbol="circle"),
        hoverinfo="none",
        name="Back: Neutron-Sonic Shadow",
        legendgroup="shadows"
    ))
    fig.add_trace(go.Scatter3d(
        x=np.full_like(v_denb, side_x),
        y=v_denb,
        z=v_dt,
        mode="markers",
        marker=dict(size=3.0, color="rgba(51, 65, 85, 0.60)", symbol="circle"),
        hoverinfo="none",
        name="Side: Density-Sonic Shadow",
        legendgroup="shadows"
    ))

    # 2. 3D Mineral Matrix Surfaces & Trend Lines (Rhomb/Triangle Grid)
    # Mineral matrix triangle at 0% porosity
    fig.add_trace(go.Scatter3d(
        x=[0.00, 0.00, 0.02, 0.00],
        y=[2.65, 2.71, 2.87, 2.65],
        z=[55.5, 47.5, 43.5, 55.5],
        mode="lines+markers+text",
        line=dict(color="#1E293B", width=3.5, dash="solid"),
        marker=dict(size=4.5, color="#0F172A"),
        text=["Quartz (2.65)", "Calcite (2.71)", "Dolomite (2.87)", ""],
        textposition="top right",
        textfont=dict(size=10, color="#1E293B"),
        name="0% Porosity Mineral Triangle",
        legendgroup="minerals"
    ))

    # Lithology Porosity Trends (0% to 35% porosity)
    phi_grid = np.linspace(0.0, 0.35, 20)
    
    # Sandstone / Quartz line
    fig.add_trace(go.Scatter3d(
        x=phi_grid,
        y=2.65 * (1 - phi_grid) + 1.0 * phi_grid,
        z=55.5 * (1 - phi_grid) + 189.0 * phi_grid,
        mode="lines",
        line=dict(color="#EAB308", width=3),
        name="Sandstone Calibration Line",
        legendgroup="minerals"
    ))
    # Limestone / Calcite line
    fig.add_trace(go.Scatter3d(
        x=phi_grid,
        y=2.71 * (1 - phi_grid) + 1.0 * phi_grid,
        z=47.5 * (1 - phi_grid) + 189.0 * phi_grid,
        mode="lines",
        line=dict(color="#3B82F6", width=3),
        name="Limestone Calibration Line",
        legendgroup="minerals"
    ))
    # Dolomite line
    fig.add_trace(go.Scatter3d(
        x=0.02 + 0.98 * phi_grid,
        y=2.87 * (1 - phi_grid) + 1.0 * phi_grid,
        z=43.5 * (1 - phi_grid) + 189.0 * phi_grid,
        mode="lines",
        line=dict(color="#A855F7", width=3),
        name="Dolomite Calibration Line",
        legendgroup="minerals"
    ))

    # Iso-Porosity Rungs (10%, 20%, 30%)
    for p_iso in [0.10, 0.20, 0.30]:
        rung_x = [p_iso, p_iso, 0.02 + 0.98 * p_iso]
        rung_y = [2.65 * (1 - p_iso) + 1.0 * p_iso, 2.71 * (1 - p_iso) + 1.0 * p_iso, 2.87 * (1 - p_iso) + 1.0 * p_iso]
        rung_z = [55.5 * (1 - p_iso) + 189.0 * p_iso, 47.5 * (1 - p_iso) + 189.0 * p_iso, 43.5 * (1 - p_iso) + 189.0 * p_iso]
        fig.add_trace(go.Scatter3d(
            x=rung_x, y=rung_y, z=rung_z,
            mode="lines+text",
            line=dict(color="#94A3B8", width=1.8, dash="dash"),
            text=["", f"Φ = {int(p_iso*100)}%", ""],
            textposition="top center",
            textfont=dict(size=9, color="#64748B"),
            name=f"Iso-Porosity {int(p_iso*100)}% Rung",
            legendgroup="minerals",
            showlegend=(p_iso == 0.20)
        ))

    # Mineral Matrix Calibration Surface (Mesh3D)
    surf_pts = np.array([
        [0.00, 2.65, 55.5],
        [0.00, 2.71, 47.5],
        [0.02, 2.87, 43.5],
        [0.30, 2.65 * 0.7 + 0.3, 55.5 * 0.7 + 189 * 0.3],
        [0.30, 2.71 * 0.7 + 0.3, 47.5 * 0.7 + 189 * 0.3],
        [0.02 + 0.98 * 0.3, 2.87 * 0.7 + 0.3, 43.5 * 0.7 + 189 * 0.3]
    ])
    fig.add_trace(go.Mesh3d(
        x=surf_pts[:, 0],
        y=surf_pts[:, 1],
        z=surf_pts[:, 2],
        i=[0, 0, 0, 1, 1, 3],
        j=[1, 3, 4, 4, 5, 4],
        k=[2, 4, 1, 5, 2, 5],
        color="#CBD5E1",
        opacity=0.18,
        name="Mineral Matrix Calibration Sheet",
        legendgroup="minerals",
        showlegend=True
    ))

    # 3. Volumetric Density Shells / Isosurface Envelopes
    norm_n = (v_neut - np.mean(v_neut)) / (np.std(v_neut) + 1e-6)
    norm_b = (v_denb - np.mean(v_denb)) / (np.std(v_denb) + 1e-6)
    norm_t = (v_dt - np.mean(v_dt)) / (np.std(v_dt) + 1e-6)
    dist = np.sqrt(norm_n**2 + norm_b**2 + norm_t**2)

    # 50% core density hull
    p50_mask = dist <= np.percentile(dist, 50)
    if np.sum(p50_mask) >= 12:
        try:
            pts_50 = np.column_stack([v_neut[p50_mask], v_denb[p50_mask], v_dt[p50_mask]])
            hull_50 = ConvexHull(pts_50)
            fig.add_trace(go.Mesh3d(
                x=pts_50[:, 0], y=pts_50[:, 1], z=pts_50[:, 2],
                i=hull_50.simplices[:, 0], j=hull_50.simplices[:, 1], k=hull_50.simplices[:, 2],
                color="#F59E0B",
                opacity=0.22,
                name="50% Core Density Hull",
                lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5),
                showlegend=True
            ))
        except Exception:
            pass

    # 80% regional containment shell
    p80_mask = dist <= np.percentile(dist, 80)
    if np.sum(p80_mask) >= 16:
        try:
            pts_80 = np.column_stack([v_neut[p80_mask], v_denb[p80_mask], v_dt[p80_mask]])
            hull_80 = ConvexHull(pts_80)
            fig.add_trace(go.Mesh3d(
                x=pts_80[:, 0], y=pts_80[:, 1], z=pts_80[:, 2],
                i=hull_80.simplices[:, 0], j=hull_80.simplices[:, 1], k=hull_80.simplices[:, 2],
                color="#3B82F6",
                opacity=0.08,
                name="80% Regional Containment Shell",
                lighting=dict(ambient=0.7, diffuse=0.6, roughness=0.6),
                showlegend=True
            ))
        except Exception:
            pass

    # 4. Discrete Lithofacies Classification & Toggleable Traces
    facies_labels = []
    for d, n, b, t, g, r in zip(v_depths, v_neut, v_denb, v_dt, v_gr, v_rt):
        if b >= 2.65 and n < 0.18:
            facies_labels.append("Tight Carbonate")
        elif g < 65 and b < 2.40 and r >= 8.0:
            facies_labels.append("Clean Gas Pay")
        elif g < 65 and b < 2.45:
            facies_labels.append("Clean Water Sand")
        elif g < 95 and b < 2.58:
            facies_labels.append("Shaly Sand")
        else:
            facies_labels.append("Shale / Clay")
    facies_arr = np.array(facies_labels)

    facies_config = [
        ("Clean Gas Pay", "#F59E0B"),       # Amber/Gold
        ("Clean Water Sand", "#06B6D4"),     # Cyan/Teal
        ("Shaly Sand", "#10B981"),          # Emerald Green
        ("Shale / Clay", "#64748B"),        # Slate Gray
        ("Tight Carbonate", "#3B82F6")      # Royal Blue
    ]

    for fname, fcolor in facies_config:
        fmask = (facies_arr == fname)
        count = int(np.sum(fmask))
        if count == 0:
            continue
            
        m_depth = v_depths[fmask]
        m_neut = v_neut[fmask]
        m_denb = v_denb[fmask]
        m_dt = v_dt[fmask]
        m_gr = v_gr[fmask]
        m_rt = v_rt[fmask]
        m_phie = v_phie[fmask]
        
        # Marker sizes dynamically weighted by log(Rt)
        clean_rt = np.nan_to_num(m_rt, nan=5.0, posinf=50.0, neginf=0.1)
        marker_sizes = np.clip(3.5 + 2.0 * np.log10(np.maximum(clean_rt, 0.1)), 3.5, 7.5)
        
        hover_texts = [
            f"<b>{fname}</b><br>"
            f"Depth (MD): <b>{d:.2f} m</b><br>"
            f"Neutron Porosity (NPHI): <b>{n:.3f} v/v</b><br>"
            f"Bulk Density (RHOB): <b>{b:.2f} g/cc</b><br>"
            f"Sonic Slowness (DT): <b>{t:.1f} µs/ft</b><br>"
            f"Gamma Ray (GR): <b>{g:.1f} API</b><br>"
            f"Deep Resistivity (Rt): <b>{r:.2f} Ω·m</b><br>"
            f"Effective Porosity (Φe): <b>{p*100:.1f}%</b>"
            for d, n, b, t, g, r, p in zip(m_depth, m_neut, m_denb, m_dt, m_gr, m_rt, m_phie)
        ]
        
        fig.add_trace(go.Scatter3d(
            x=m_neut,
            y=m_denb,
            z=m_dt,
            mode="markers",
            marker=dict(
                size=marker_sizes,
                color=fcolor,
                opacity=0.88,
                line=dict(color="#0F172A", width=0.5)
            ),
            text=hover_texts,
            hoverinfo="text",
            name=f"{fname} ({count} pts)",
            legendgroup="facies"
        ))

    # 5. Visual Anchor: Gas Effect Drop-Lines / Leader Lines down to Sandstone Matrix
    gas_mask = (facies_arr == "Clean Gas Pay")
    if np.any(gas_mask):
        g_neut = v_neut[gas_mask]
        g_denb = v_denb[gas_mask]
        g_dt = v_dt[gas_mask]
        
        # Subsample up to 50 prominent leader lines to keep canvas sharp & uncluttered
        step = max(1, len(g_neut) // 50)
        lead_x: List[Optional[float]] = []
        lead_y: List[Optional[float]] = []
        lead_z: List[Optional[float]] = []
        for n, b, t in zip(g_neut[::step], g_denb[::step], g_dt[::step]):
            phi_eq = float(np.clip((2.65 - b) / 1.65, 0.05, 0.35))
            mat_x = phi_eq
            mat_y = 2.65 * (1.0 - phi_eq) + 1.0 * phi_eq
            mat_z = 55.5 * (1.0 - phi_eq) + 189.0 * phi_eq
            lead_x.extend([float(n), mat_x, None])
            lead_y.extend([float(b), mat_y, None])
            lead_z.extend([float(t), mat_z, None])
            
        fig.add_trace(go.Scatter3d(
            x=lead_x,
            y=lead_y,
            z=lead_z,
            mode="lines",
            line=dict(color="#EF4444", width=1.5, dash="dot"),
            opacity=0.60,
            hoverinfo="none",
            name="Gas Displacement Vectors",
            legendgroup="facies"
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>3D Petrophysical Cluster Space: {well_id}</b> ({v_depths.min():.1f}m – {v_depths.max():.1f}m)",
            x=0.02,
            xanchor="left",
            font=dict(size=14, color="#0F172A")
        ),
        template="plotly_white",
        scene=dict(
            xaxis=dict(
                title=dict(text="Neutron Porosity (NPHI, v/v)", font=dict(size=11, color="#0F172A")),
                range=[min_x, side_x],
                backgroundcolor="#F8FAFC",
                gridcolor="#E2E8F0",
                showbackground=True,
                zerolinecolor="#CBD5E1"
            ),
            yaxis=dict(
                title=dict(text="Bulk Density (RHOB, g/cc)", font=dict(size=11, color="#0F172A")),
                range=[back_y, min_y],
                backgroundcolor="#F1F5F9",
                gridcolor="#E2E8F0",
                showbackground=True,
                zerolinecolor="#CBD5E1"
            ),
            zaxis=dict(
                title=dict(text="Compressional Slowness (DT, µs/ft)", font=dict(size=11, color="#0F172A")),
                range=[floor_z, max_z],
                backgroundcolor="#F8FAFC",
                gridcolor="#E2E8F0",
                showbackground=True,
                zerolinecolor="#CBD5E1"
            ),
            camera=dict(
                eye=dict(x=1.35, y=-1.50, z=0.92),
                center=dict(x=0.0, y=0.0, z=-0.05)
            ),
            aspectmode="cube"
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.92)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=10, color="#334155"),
            itemsizing="constant"
        ),
        height=680,
        margin=dict(l=30, r=170, t=50, b=40)
    )

    return {
        "well_id": well_id,
        "samples_rendered": int(len(v_depths)),
        "min_depth": float(v_depths.min()),
        "max_depth": float(v_depths.max()),
        "facies_counts": {
            fname: int(np.sum(facies_arr == fname))
            for fname, _ in facies_config
        },
        "figure_json": fig.to_json()
    }


def plot_3d_wellbore_trajectory(
    well_id: str,
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None,
    color_by: Optional[str] = "sweetspots",
    highlight_top: Optional[float] = None,
    highlight_base: Optional[float] = None,
    highlight_label: Optional[str] = None,
    show_horizon: bool = True
) -> Dict[str, Any]:
    """Generates an interactive 3D Subsurface Wellbore Trajectory with sweet spot pay zones,
    dynamic trajectory attribute coloring (Sweet Spots, RDEEP, GR, TVDSS), a contoured
    geological reservoir horizon, and optional sweet-spot target beacon highlight.
    """
    las, df = read_las(well_id)
    cols = {c.upper(): c for c in df.columns}
    # When a target highlight is requested, expand the effective view to always
    # include the highlight interval. Otherwise the amber beacon (drawn at true
    # hole coordinates) can appear floating far from a wellbore segment that
    # was sliced to a range excluding the target.
    has_highlight = highlight_top is not None and highlight_base is not None
    view_top = top_depth
    view_bot = bottom_depth
    if has_highlight:
        view_top = highlight_top if view_top is None else min(view_top, highlight_top)
        view_bot = highlight_base if view_bot is None else max(view_bot, highlight_base)
    sub = df.copy()
    if view_top is not None:
        sub = sub[sub["DEPTH"] >= view_top]
    if view_bot is not None:
        sub = sub[sub["DEPTH"] <= view_bot]

    depths = sub["DEPTH"].values
    if len(depths) == 0:
        raise ValueError("No data samples available.")

    tvd_col = cols.get("TVDSS") or cols.get("TVD")
    tvd = sub[tvd_col].values if tvd_col else -depths

    # Always use the well's absolute start as the deviation reference so that
    # the displayed (filtered) trace and any beacon highlight share the same XY frame.
    d0 = df["DEPTH"].values[0]
    dev_x = 45.0 * np.sin((depths - d0) / 140.0)
    dev_y = 60.0 * (1.0 - np.cos((depths - d0) / 180.0))
    z = tvd

    if "VSHALE" in cols:
        vsh = sub[cols["VSHALE"]].values
    elif "GR" in cols:
        vsh = np.clip((sub[cols["GR"]].values - 25.0) / 100.0, 0.0, 1.0)
    else:
        vsh = np.zeros(len(depths))

    if "PHIE" in cols:
        phi = sub[cols["PHIE"]].values
    elif "DENB" in cols:
        phi = np.clip((2.65 - sub[cols["DENB"]].values) / 1.65, 0.0, 0.40)
    else:
        phi = np.full(len(depths), 0.15)

    rdeep_col = cols.get("RDEEP") or cols.get("ILD") or cols.get("RT")
    rdeep_vals = sub[rdeep_col].values if rdeep_col else np.full(len(depths), 10.0)
    rdeep_vals = np.maximum(rdeep_vals, 0.1)

    if "SWE" in cols:
        sw = sub[cols["SWE"]].values
    else:
        sw = np.sqrt(np.clip(0.05 / (np.maximum(phi, 0.01)**2.0 * rdeep_vals), 0.0, 1.0))

    is_pay = (vsh <= 0.3) & (phi >= 0.10) & (sw <= 0.50) & (~np.isnan(depths))

    fig = go.Figure()

    active_attr = (color_by or "sweetspots").lower()

    if active_attr == "rdeep":
        log_r = np.log10(np.clip(rdeep_vals, 0.2, 2000.0))
        fig.add_trace(go.Scatter3d(
            x=dev_x, y=dev_y, z=z,
            mode="markers+lines",
            marker=dict(
                size=5,
                color=log_r,
                colorscale="Turbo",
                colorbar=dict(
                    title=dict(text="<b>RDEEP (Ω·m)</b>", side="top"),
                    tickvals=[float(np.log10(v)) for v in [0.2, 1, 10, 100, 1000]],
                    ticktext=["0.2", "1", "10", "100", "1000"],
                    thickness=14,
                    len=0.65,
                    x=1.02
                ),
                opacity=0.9
            ),
            line=dict(color="#64748b", width=3),
            text=[f"Depth: {d:.1f}m<br>Rt: {r:.2f} Ω·m" for d, r in zip(depths, rdeep_vals)],
            hoverinfo="text",
            name="Wellbore (Resistivity Rt)"
        ))
    elif active_attr == "gr":
        gr_vals = sub[cols["GR"]].values if "GR" in cols else np.full(len(depths), 50.0)
        fig.add_trace(go.Scatter3d(
            x=dev_x, y=dev_y, z=z,
            mode="markers+lines",
            marker=dict(
                size=5,
                color=gr_vals,
                colorscale="Viridis",
                colorbar=dict(title=dict(text="<b>GR (API)</b>", side="top"), thickness=14, len=0.65, x=1.02),
                opacity=0.9
            ),
            line=dict(color="#64748b", width=3),
            text=[f"Depth: {d:.1f}m<br>GR: {g:.1f} API" for d, g in zip(depths, gr_vals)],
            hoverinfo="text",
            name="Wellbore (Gamma Ray)"
        ))
    elif active_attr == "tvdss":
        fig.add_trace(go.Scatter3d(
            x=dev_x, y=dev_y, z=z,
            mode="markers+lines",
            marker=dict(
                size=5,
                color=z,
                colorscale="Blues_r",
                colorbar=dict(title=dict(text="<b>TVDSS (m)</b>", side="top"), thickness=14, len=0.65, x=1.02),
                opacity=0.9
            ),
            line=dict(color="#64748b", width=3),
            text=[f"MD: {d:.1f}m<br>TVDSS: {t:.1f}m" for d, t in zip(depths, z)],
            hoverinfo="text",
            name="Wellbore (TVDSS Depth)"
        ))
    else:
        # Default: Sweet Spots vs Overburden
        fig.add_trace(go.Scatter3d(
            x=dev_x[~is_pay],
            y=dev_y[~is_pay],
            z=z[~is_pay],
            mode="markers+lines",
            marker=dict(size=3.5, color="#64748b", opacity=0.7),
            line=dict(color="#94a3b8", width=3),
            name="Overburden / Non-Pay"
        ))

        if np.any(is_pay):
            fig.add_trace(go.Scatter3d(
                x=dev_x[is_pay],
                y=dev_y[is_pay],
                z=z[is_pay],
                mode="markers+lines",
                marker=dict(size=7, color="#10b981", opacity=1.0),
                line=dict(color="#059669", width=6),
                name="Hydrocarbon Pay Zone (Sweet Spot)"
            ))

    # ── Geological Reservoir Horizon ────────────────────────────────────────
    if show_horizon and np.any(is_pay):
        pay_top_z = float(z[is_pay][0])
        gx, gy = np.meshgrid(np.linspace(-160, 160, 40), np.linspace(-120, 220, 40))
        # Structural dip: 3.5 cm/m East (0.035), -2 cm/m North (−0.020)
        dip_x, dip_y = 0.042, -0.025
        # Realistic undulation: low-amplitude sinusoidal fault drag
        undulation = (
            3.5 * np.sin(gx / 80.0) * np.cos(gy / 110.0)
            + 1.8 * np.sin(gx / 45.0 + 0.7)
        )
        gz = pay_top_z + dip_x * gx + dip_y * gy + undulation

        fig.add_trace(go.Surface(
            x=gx,
            y=gy,
            z=gz,
            opacity=0.52,
            colorscale="Earth",         # geological look
            showscale=True,
            colorbar=dict(
                title=dict(text="<b>TVDSS (m)</b>", side="right"),
                thickness=10,
                len=0.45,
                x=0.02,
                xanchor="left",
                tickfont=dict(size=9)
            ),
            contours=dict(
                z=dict(show=True, usecolormap=True, width=1.5, project=dict(z=False))
            ),
            hovertemplate="Horizon TVDSS: %{z:.1f} m<extra>Top Reservoir Horizon</extra>",
            name="Top Reservoir Horizon"
        ))

        # Elevated leader-line pin above the horizon
        pin_x = float(dev_x[np.where(is_pay)[0][0]])
        pin_y = float(dev_y[np.where(is_pay)[0][0]])
        dip_deg = float(np.degrees(np.arctan(np.sqrt(dip_x**2 + dip_y**2))))
        label_z_offset = abs(float(z.max() - z.min())) * 0.08 + 12.0

        # Vertical leader line from horizon to label
        fig.add_trace(go.Scatter3d(
            x=[pin_x, pin_x],
            y=[pin_y, pin_y],
            z=[pay_top_z, pay_top_z - label_z_offset],
            mode="lines",
            line=dict(color="#f59e0b", width=2, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
            name="_horizon_leader"
        ))

        fig.add_trace(go.Scatter3d(
            x=[pin_x],
            y=[pin_y],
            z=[pay_top_z - label_z_offset],
            mode="text+markers",
            marker=dict(size=9, color="#f59e0b", symbol="diamond", line=dict(color="#b45309", width=1.5)),
            text=[f"<b>Marker: Top Reservoir Horizon</b><br>TVDSS ≈ {abs(pay_top_z):.1f} m | Dip: ~{dip_deg:.1f}° SE"],
            textposition="top center",
            textfont=dict(color="#78350f", size=11, family="monospace"),
            hoverinfo="text",
            name="Formation Top Pick"
        ))

    # ── Sweet Spot Highlight Beacon ──────────────────────────────────────────
    hl_focus = None
    highlight_matched = False
    if has_highlight:
        # Derive the highlight from the SAME displayed arrays (depths/dev_x/
        # dev_y/z) so it is mathematically impossible for the beacon to sit
        # off-hole. The view slice above was already expanded to include the
        # highlight interval, so this mask is normally non-empty.
        hl_mask = (depths >= highlight_top) & (depths <= highlight_base)
        if np.any(hl_mask):
            highlight_matched = True
            hx = dev_x[hl_mask]
            hy = dev_y[hl_mask]
            hz = z[hl_mask]

            # Thick amber "glowing" segment
            fig.add_trace(go.Scatter3d(
                x=hx, y=hy, z=hz,
                mode="markers+lines",
                marker=dict(size=11, color="#f59e0b", opacity=1.0,
                            line=dict(color="#fef3c7", width=2)),
                line=dict(color="#d97706", width=10),
                text=[f"🎯 TARGET SWEET SPOT<br>{highlight_label or ''}<br>MD: {d:.1f}m"
                      for d in depths[hl_mask]],
                hoverinfo="text",
                name=f"🎯 Target: {highlight_label or 'Sweet Spot'}"
            ))

            # Callout beacon pin: keep it tightly coupled to the wellbore so the
            # visual clearly corresponds. Same XY as the wellbore mid-point, with
            # only a small Z offset + an explicit leader line back to the hole.
            mid_idx = len(hx) // 2
            mx = float(hx[mid_idx])
            my = float(hy[mid_idx])
            mz = float(hz[mid_idx])
            hl_thickness = max(float(highlight_base - highlight_top), 1.0)
            # Small clamped offset scaled to the TARGET (not the view span, which
            # can be 100m+ and used to leave the beacon floating ~20m off-hole).
            beacon_offset = float(np.clip(hl_thickness * 1.2, 3.0, 8.0))
            # z decreases with depth here (TVDSS negative / -MD), so deeper =
            # smaller z. Place the beacon just below the target (deeper) and
            # draw a leader back up to the wellbore.
            beacon_z = mz - beacon_offset

            # Leader line: wellbore mid-point -> beacon (makes correspondence explicit)
            fig.add_trace(go.Scatter3d(
                x=[mx, mx],
                y=[my, my],
                z=[mz, beacon_z],
                mode="lines",
                line=dict(color="#ef4444", width=4, dash="solid"),
                showlegend=False,
                hoverinfo="skip",
                name="_target_leader"
            ))

            fig.add_trace(go.Scatter3d(
                x=[mx],
                y=[my],
                z=[beacon_z],
                mode="text+markers",
                marker=dict(size=14, color="#ef4444", symbol="circle",
                            line=dict(color="#fca5a5", width=3)),
                text=[f"<b>🎯 TARGET SWEET SPOT</b><br>{highlight_label or ''}<br>{highlight_top:.0f}m – {highlight_base:.0f}m MD"],
                textposition="bottom center",
                textfont=dict(color="#7f1d1d", size=12, family="monospace"),
                hoverinfo="text",
                name="Target Beacon"
            ))

            # Zoom the scene to the highlight so a thin (few-m) target is not a
            # sub-pixel dot inside a 400m context window. Window scales with the
            # target thickness, clamped to a usable overview. Applied AFTER the
            # base layout below so axis titles/camera defaults don't wipe it.
            xy_margin = float(np.clip(hl_thickness * 6.0, 25.0, 80.0))
            z_margin = float(np.clip(hl_thickness * 8.0, 20.0, 90.0))
            hl_focus = (mx, my, mz, xy_margin, z_margin)

    fig_title = f"<b>3D Subsurface Wellbore Trajectory: {well_id}</b> (Color by: {active_attr.upper()})"
    if has_highlight:
        if highlight_matched:
            fig_title += f" | 🎯 Zone {highlight_top:.0f}–{highlight_base:.0f}m"
        else:
            fig_title += " | ⚠️ Target outside data range"
    fig.update_layout(
        title=fig_title,
        template="plotly_white",
        scene=dict(
            xaxis_title="Easting X (m)",
            yaxis_title="Northing Y (m)",
            zaxis_title="Subsea Depth TVDSS (m)",
            camera=dict(eye=dict(x=1.65, y=1.65, z=0.95)),
            bgcolor="#F8FAFC"
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.90)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=10)
        ),
        height=660,
        margin=dict(l=20, r=160, t=50, b=30),
    )

    if hl_focus is not None:
        mx, my, mz, xy_margin, z_margin = hl_focus
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[mx - xy_margin, mx + xy_margin]),
                yaxis=dict(range=[my - xy_margin, my + xy_margin]),
                zaxis=dict(range=[mz - z_margin, mz + z_margin]),
                camera=dict(eye=dict(x=1.65, y=1.65, z=0.95)),
            )
        )

    return {
        "well_id": well_id,
        "total_depth_samples": int(len(depths)),
        "pay_samples": int(np.sum(is_pay)),
        "color_by": active_attr,
        "view_top_m": float(view_top) if view_top is not None else None,
        "view_base_m": float(view_bot) if view_bot is not None else None,
        "highlight_matched": bool(highlight_matched) if has_highlight else None,
        "figure_json": fig.to_json()
    }

