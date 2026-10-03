"""MCP server: the single registry of petrophysics tools.

Every LLM client reaches the engine through this server:
- the web app's Gemini agent (backend/agent.py, via backend/mcp_client.py)
- desktop MCP clients such as Claude Desktop or Cursor (launched over stdio)

Tool names, descriptions and parameter schemas are declared here once; clients
discover them with list_tools. Tools return JSON numbers. Plots are not put in
tool results (megabytes of JSON no model can use); each plot is published as a
resource, figure://<id>, which a UI client can read and a chat client ignores.
"""

import json
import sys
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, Optional

# Allow `python backend/mcp_server.py` (how MCP clients launch it): running a
# file as a script puts backend/ on sys.path, not the repo root, so the
# `backend.` imports below would fail without this.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from backend.catalog import query_catalog
from backend.petrophysics import (
    calculate_archie_saturation as _calculate_archie_saturation,
    compare_vshale_methods as _compare_vshale_methods,
    compute_net_pay as _compute_net_pay,
    compute_permeability_timur_coates as _compute_permeability_timur_coates,
    compute_sonic_porosity_wyllie as _compute_sonic_porosity_wyllie,
    generate_reservoir_composite_report as _generate_reservoir_composite_report,
    get_well_curves_summary as _get_well_curves_summary,
    plot_1d_well_log as _plot_1d_well_log,
    plot_2d_crossplot as _plot_2d_crossplot,
    plot_3d_petrophysical_cube as _plot_3d_petrophysical_cube,
    plot_3d_wellbore_trajectory as _plot_3d_wellbore_trajectory,
    plot_crossplot_picket as _plot_crossplot_picket,
    scan_reservoir_sweetspots as _scan_reservoir_sweetspots,
)

mcp = MCPServer("WellLogPetrophysicsMCP")

# Reusable parameter descriptions (they become the JSON Schema every client sees)
WellId = Annotated[str, Field(description="Well identifier, e.g. 'Well1' (a loaded LAS file name without .las)")]
Top = Annotated[float, Field(description="Top depth in meters (MD)")]
Bottom = Annotated[float, Field(description="Bottom depth in meters (MD)")]
OptTop = Annotated[Optional[float], Field(description="Optional top depth in meters (MD); omit for the whole log")]
OptBottom = Annotated[Optional[float], Field(description="Optional bottom depth in meters (MD); omit for the whole log")]
Rw = Annotated[float, Field(description="Formation water resistivity in ohm.m")]
CementationM = Annotated[float, Field(description="Archie cementation exponent m")]
SaturationN = Annotated[float, Field(description="Archie saturation exponent n")]

# Plots published as resources: id -> Plotly figure JSON (bounded, newest kept)
_FIGURES: "OrderedDict[str, str]" = OrderedDict()
_MAX_FIGURES = 64


def _publish(fn: Callable[..., Dict[str, Any]], *args: Any, **kwargs: Any) -> str:
    """Run an engine function and return its result as JSON.

    Bad input (unknown well, empty interval, missing curve) becomes a ToolError
    so the caller's model sees the actual reason. A figure is swapped for a
    figure://<id> resource link instead of being sent inline.
    """
    try:
        res = fn(*args, **kwargs)
    except (ValueError, FileNotFoundError) as e:
        raise ToolError(str(e)) from e
    out = dict(res)
    fig = out.pop("figure_json", None)
    if fig is not None:
        fig_id = uuid.uuid4().hex
        _FIGURES[fig_id] = fig
        while len(_FIGURES) > _MAX_FIGURES:
            _FIGURES.popitem(last=False)
        out["figure_uri"] = f"figure://{fig_id}"
    return json.dumps(out)


@mcp.resource("figure://{figure_id}", mime_type="application/json",
              description="Plotly figure JSON produced by a plotting tool call")
def get_figure(figure_id: str) -> str:
    if figure_id not in _FIGURES:
        raise ValueError(f"Figure {figure_id} is not available (it may have expired)")
    return _FIGURES[figure_id]


@mcp.tool()
def get_well_curves_summary(well_id: WellId) -> str:
    """Lists a well's curves (mnemonic, unit, value range), depth range, which analyses its curves support, and a suggested 100 m starting window."""
    return _publish(_get_well_curves_summary, well_id)


@mcp.tool()
def plot_1d_well_log(
    well_id: WellId,
    top_depth: OptTop = None,
    bottom_depth: OptBottom = None,
    marker_depth: Annotated[Optional[float], Field(description="Optional depth in meters for a horizontal correlation line across all tracks")] = None,
) -> str:
    """Plots the standard 3-track log for a depth interval: Track 1 gamma ray and caliper, Track 2 resistivity on a log scale, Track 3 density-neutron with crossover shading. Track 3 is shaded red where the density-correction curve flags the density reading as unreliable; the result's density_quality lists those intervals."""
    return _publish(_plot_1d_well_log, well_id, top_depth, bottom_depth, marker_depth)


@mcp.tool()
def plot_2d_crossplot(
    well_id: WellId,
    x_curve: Annotated[str, Field(description="Mnemonic for the X axis, e.g. 'NEUT' or 'NPHI'")],
    y_curve: Annotated[str, Field(description="Mnemonic for the Y axis, e.g. 'DENB' or 'RHOB'")],
    z_curve: Annotated[Optional[str], Field(description="Optional mnemonic used for point color, e.g. 'GR'")] = None,
    top_depth: OptTop = None,
    bottom_depth: OptBottom = None,
) -> str:
    """Plots a 2D crossplot of two curves (typically density vs neutron for lithology) with sandstone, limestone and dolomite trend lines."""
    return _publish(_plot_2d_crossplot, well_id, x_curve, y_curve, z_curve, top_depth, bottom_depth)


@mcp.tool()
def compute_net_pay(
    well_id: WellId,
    top_depth: Top,
    bottom_depth: Bottom,
    vsh_cutoff: Annotated[float, Field(description="Maximum shale volume fraction for reservoir")] = 0.3,
    phi_cutoff: Annotated[float, Field(description="Minimum porosity fraction for reservoir")] = 0.1,
    sw_cutoff: Annotated[float, Field(description="Maximum water saturation fraction for pay")] = 0.5,
) -> str:
    """Computes gross interval, net reservoir, net pay and net-to-gross from shale volume, porosity and water saturation cutoffs, with pay-zone averages, a cutoff sensitivity grid, and density_quality (how much pay rests on a density reading flagged unreliable)."""
    return _publish(_compute_net_pay, well_id, top_depth, bottom_depth, vsh_cutoff, phi_cutoff, sw_cutoff)


@mcp.tool()
def compare_vshale_methods(well_id: WellId, top_depth: Top, bottom_depth: Bottom) -> str:
    """Compares four shale volume models from gamma ray (linear, Larionov Tertiary, Steiber, Clavier) over an interval, with their averages."""
    return _publish(_compare_vshale_methods, well_id, top_depth, bottom_depth)


@mcp.tool()
def calculate_archie_saturation(
    well_id: WellId,
    top_depth: Top,
    bottom_depth: Bottom,
    rw: Rw = 0.05,
    m: CementationM = 2.0,
    n: SaturationN = 2.0,
) -> str:
    """Computes Archie water saturation, hydrocarbon saturation and bulk volume hydrocarbon over an interval (interval averages include shales)."""
    return _publish(_calculate_archie_saturation, well_id, top_depth, bottom_depth, rw=rw, m=m, n=n)


@mcp.tool()
def compute_sonic_porosity_wyllie(
    well_id: WellId,
    top_depth: Top,
    bottom_depth: Bottom,
    dt_matrix: Annotated[float, Field(description="Matrix transit time in us/ft (55.5 for sandstone)")] = 55.5,
    dt_fluid: Annotated[float, Field(description="Fluid transit time in us/ft (189 for fresh water)")] = 189.0,
) -> str:
    """Computes Wyllie time-average porosity from the compressional sonic log and compares it with density porosity."""
    return _publish(_compute_sonic_porosity_wyllie, well_id, top_depth, bottom_depth, dt_matrix, dt_fluid)


@mcp.tool()
def scan_reservoir_sweetspots(
    well_id: WellId,
    min_thickness: Annotated[float, Field(description="Minimum continuous pay thickness in meters")] = 1.5,
) -> str:
    """Scans the whole well for continuous pay intervals (clean, porous, low water saturation) and ranks them by hydrocarbon pore volume."""
    return _publish(_scan_reservoir_sweetspots, well_id, min_thickness)


@mcp.tool()
def compute_permeability_timur_coates(
    well_id: WellId,
    top_depth: Top,
    bottom_depth: Bottom,
    model: Annotated[str, Field(description="'timur' or 'coates'")] = "timur",
) -> str:
    """Estimates permeability (mD) with the Timur or Coates empirical model and flow capacity kh over net-pay samples only."""
    return _publish(_compute_permeability_timur_coates, well_id, top_depth, bottom_depth, model)


@mcp.tool()
def plot_crossplot_picket(
    well_id: WellId,
    top_depth: OptTop = None,
    bottom_depth: OptBottom = None,
    rw: Rw = 0.05,
    m: CementationM = 2.0,
    n: SaturationN = 2.0,
) -> str:
    """Plots a Picket plot (log resistivity vs log porosity) with the 100% water line and iso-saturation lines."""
    return _publish(_plot_crossplot_picket, well_id, top_depth, bottom_depth, rw, m, n)


@mcp.tool()
def generate_reservoir_composite_report(well_id: WellId, top_depth: Top, bottom_depth: Bottom) -> str:
    """Summarizes an interval in one result: net pay, net-to-gross, average porosity, water saturation and shale volume, kh, and a rough fluid label."""
    return _publish(_generate_reservoir_composite_report, well_id, top_depth, bottom_depth)


@mcp.tool()
def plot_3d_petrophysical_cube(
    well_id: WellId,
    top_depth: OptTop = None,
    bottom_depth: OptBottom = None,
    color_by: Annotated[str, Field(description="'pay' (net pay rules, default) or 'depth'")] = "pay",
) -> str:
    """Exploratory 3D crossplot of neutron, density and sonic with a depth track; points colored by the net pay rules or by depth. Needs neutron, density and sonic curves."""
    return _publish(_plot_3d_petrophysical_cube, well_id, top_depth, bottom_depth, color_by)


@mcp.tool()
def plot_3d_wellbore_trajectory(well_id: WellId, top_depth: OptTop = None, bottom_depth: OptBottom = None) -> str:
    """3D wellbore view colored by computed pay flags along the hole. The XY path is illustrative (no deviation survey in the data)."""
    return _publish(_plot_3d_wellbore_trajectory, well_id, top_depth, bottom_depth)


@mcp.tool()
def query_geology_metadata(
    query: Annotated[str, Field(description="Search keywords, e.g. 'formation tops' or 'gas shows'")],
    well_id: Annotated[Optional[str], Field(description="Optional well id to restrict the search")] = None,
) -> str:
    """Searches the geology reports (formation tops, lithology and mudlog notes) by keyword. Reports are unverified free text: each result includes a "verification" list marking every measurable statement as consistent, partly_consistent or contradicted against the well's logs, or not_checkable (mudlog, core, test data)."""
    return json.dumps({"query": query, "results": query_catalog(query, well_id)})


if __name__ == "__main__":
    mcp.run()
