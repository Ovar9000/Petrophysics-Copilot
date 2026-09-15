"""FastMCP Tool Server for Petrophysical Analytics.

Exposes deterministic subsurface calculation and visualization tools
via the Model Context Protocol (MCP) for Claude Desktop, Cursor, or MCP clients.
"""

from typing import Any, Dict, List, Optional
import json
try:
    from mcp.server.mcpserver import MCPServer as MCP
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as MCP
    except ImportError:
        class MCP:
            def __init__(self, name: str):
                self.name = name
                self.tools = {}
            def tool(self):
                def decorator(func):
                    self.tools[func.__name__] = func
                    return func
                return decorator
            def run(self):
                print(f"MCP server {self.name} initialized with tools: {list(self.tools.keys())}")

from backend.petrophysics import (
    get_well_curves_summary as _get_well_curves_summary,
    plot_1d_well_log as _plot_1d_well_log,
    plot_2d_crossplot as _plot_2d_crossplot,
    compute_net_pay as _compute_net_pay,
    compare_vshale_methods as _compare_vshale_methods,
    calculate_archie_saturation as _calculate_archie_saturation,
    compute_sonic_porosity_wyllie as _compute_sonic_porosity_wyllie,
    scan_reservoir_sweetspots as _scan_reservoir_sweetspots,
    compute_permeability_timur_coates as _compute_permeability_timur_coates,
    plot_crossplot_picket as _plot_crossplot_picket,
    generate_reservoir_composite_report as _generate_reservoir_composite_report,
    plot_3d_petrophysical_cube as _plot_3d_petrophysical_cube,
    plot_3d_wellbore_trajectory as _plot_3d_wellbore_trajectory,
)
from backend.catalog import query_catalog

mcp = MCP("WellLogPetrophysicsMCP")


@mcp.tool()
def get_well_curves_summary(well_id: str) -> str:
    """Inspects the .las file and returns available curve mnemonics, units, and depth bounds."""
    res = _get_well_curves_summary(well_id)
    return json.dumps(res, indent=2)


@mcp.tool()
def plot_1d_well_log(
    well_id: str,
    curves: Optional[List[str]] = None,
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None,
    marker_depth: Optional[float] = None
) -> str:
    """Generates an interactive 3-track petrophysical log plot (GR/Caliper, Resistivity, Density-Neutron crossover)."""
    res = _plot_1d_well_log(well_id, curves, top_depth, bottom_depth, marker_depth)
    return json.dumps(res)


@mcp.tool()
def plot_2d_crossplot(
    well_id: str,
    x_curve: str,
    y_curve: str,
    z_curve: Optional[str] = None,
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None
) -> str:
    """Generates an interactive 2D lithology crossplot (e.g. RHOB vs NPHI) with mineral trendlines."""
    res = _plot_2d_crossplot(well_id, x_curve, y_curve, z_curve, top_depth, bottom_depth)
    return json.dumps(res)


@mcp.tool()
def compute_net_pay(
    well_id: str,
    top_depth: float,
    bottom_depth: float,
    vsh_cutoff: float = 0.3,
    phi_cutoff: float = 0.1,
    sw_cutoff: float = 0.5
) -> str:
    """Calculates volumetric thicknesses: Gross Interval, Net Reservoir, Net Pay, and NTG ratio."""
    res = _compute_net_pay(well_id, top_depth, bottom_depth, vsh_cutoff, phi_cutoff, sw_cutoff)
    return json.dumps(res, indent=2)


@mcp.tool()
def compare_vshale_methods(well_id: str, top_depth: float, bottom_depth: float) -> str:
    """Compares 4 shale volume calculation methods (Linear, Larionov Tertiary, Steiber, Clavier)."""
    res = _compare_vshale_methods(well_id, top_depth, bottom_depth)
    return json.dumps(res)


@mcp.tool()
def calculate_archie_saturation(
    well_id: str,
    top_depth: float,
    bottom_depth: float,
    rw: float = 0.05,
    m: float = 2.0,
    n: float = 2.0
) -> str:
    """Computes continuous Archie water saturation (Sw), hydrocarbon saturation (So), and Bulk Volume Hydrocarbon (BVH)."""
    res = _calculate_archie_saturation(well_id, top_depth, bottom_depth, rw, m, n)
    return json.dumps(res)


@mcp.tool()
def compute_sonic_porosity_wyllie(
    well_id: str,
    top_depth: float,
    bottom_depth: float,
    dt_matrix: float = 55.5,
    dt_fluid: float = 189.0
) -> str:
    """Calculates Wyllie time-average sonic porosity from compressional sonic logs (DTCOMP)."""
    res = _compute_sonic_porosity_wyllie(well_id, top_depth, bottom_depth, dt_matrix, dt_fluid)
    return json.dumps(res)


@mcp.tool()
def scan_reservoir_sweetspots(well_id: str, min_thickness: float = 1.5) -> str:
    """Scans the entire well depth array to delineate, rank, and summarize all prospective hydrocarbon sweet spots."""
    res = _scan_reservoir_sweetspots(well_id, min_thickness)
    return json.dumps(res, indent=2)


@mcp.tool()
def compute_permeability_timur_coates(
    well_id: str,
    top_depth: float,
    bottom_depth: float,
    model: str = "timur"
) -> str:
    """Calculates continuous reservoir permeability (k in mD) and flow capacity (k*h in mD*m)."""
    res = _compute_permeability_timur_coates(well_id, top_depth, bottom_depth, model)
    return json.dumps(res)


@mcp.tool()
def plot_crossplot_picket(
    well_id: str,
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None,
    rw: float = 0.05,
    m: float = 2.0,
    n: float = 2.0
) -> str:
    """Generates a classic Archie Picket Plot (log(Rt) vs log(Phi)) with 100% water line and iso-saturation trendlines."""
    res = _plot_crossplot_picket(well_id, top_depth, bottom_depth, rw, m, n)
    return json.dumps(res)


@mcp.tool()
def generate_reservoir_composite_report(well_id: str, top_depth: float, bottom_depth: float) -> str:
    """Generates an all-in-one zonal petrophysical dossier combining cutoffs, porosity, Archie saturations, and flow capacity."""
    res = _generate_reservoir_composite_report(well_id, top_depth, bottom_depth)
    return json.dumps(res, indent=2)


@mcp.tool()
def plot_3d_petrophysical_cube(
    well_id: str,
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None
) -> str:
    """Generates an advanced 3D Petrophysical Cluster Space (NPHI vs RHOB vs DT) with mineral matrix surfaces and density shells."""
    res = _plot_3d_petrophysical_cube(well_id, top_depth, bottom_depth)
    return json.dumps(res)


@mcp.tool()
def plot_3d_wellbore_trajectory(
    well_id: str,
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None
) -> str:
    """Generates an interactive 3D Subsurface Wellbore Trajectory with true spatial path and reservoir horizon surface."""
    res = _plot_3d_wellbore_trajectory(well_id, top_depth, bottom_depth)
    return json.dumps(res)


@mcp.tool()
def query_geology_metadata(query: str, well_id: Optional[str] = None) -> str:
    """Queries the local stratigraphy and mudlog catalog for geological formations, tops, and hydrocarbon shows."""
    res = query_catalog(query, well_id)
    return json.dumps(res, indent=2)


if __name__ == "__main__":
    mcp.run()
