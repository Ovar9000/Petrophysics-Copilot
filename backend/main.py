"""FastAPI backend application for Petrophysical Copilot.
"""

from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import DATA_DIR
from backend.petrophysics import (
    get_well_curves_summary,
    plot_1d_well_log,
    plot_2d_crossplot,
    compute_net_pay,
    compare_vshale_methods,
    calculate_archie_saturation,
    compute_sonic_porosity_wyllie,
    scan_reservoir_sweetspots,
    compute_permeability_timur_coates,
    plot_crossplot_picket,
    generate_reservoir_composite_report,
    plot_3d_petrophysical_cube,
    plot_3d_wellbore_trajectory,
)
from backend.catalog import query_catalog, init_catalog
from backend.agent import run_agent_turn

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_catalog()
    yield


app = FastAPI(
    title="Petrophysical Copilot API",
    version="2.0.0",
    description="Streamlined, high-performance subsurface analytics engine.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: Optional[str] = Field(default=None)
    query: Optional[str] = Field(default=None)
    well_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    chat_history: Optional[List[Dict[str, str]]] = Field(default=None)


class Plot1DRequest(BaseModel):
    well_id: str
    top_depth: Optional[float] = None
    bottom_depth: Optional[float] = None
    marker_depth: Optional[float] = None


class CrossplotRequest(BaseModel):
    well_id: str
    x_curve: str
    y_curve: str
    z_curve: Optional[str] = None
    top_depth: Optional[float] = None
    bottom_depth: Optional[float] = None


class NetPayRequest(BaseModel):
    well_id: str
    top_depth: float
    bottom_depth: float
    vsh_cutoff: float = 0.3
    phi_cutoff: float = 0.1
    sw_cutoff: float = 0.5


class VshComparisonRequest(BaseModel):
    well_id: str
    top_depth: float
    bottom_depth: float


class ArchieRequest(BaseModel):
    well_id: str
    top_depth: float
    bottom_depth: float
    rw: float = 0.05
    m: float = 2.0
    n: float = 2.0


class SonicRequest(BaseModel):
    well_id: str
    top_depth: float
    bottom_depth: float
    dt_matrix: float = 55.5
    dt_fluid: float = 189.0


class SweetspotRequest(BaseModel):
    well_id: str
    min_thickness: float = 1.5


class PermeabilityRequest(BaseModel):
    well_id: str
    top_depth: float
    bottom_depth: float
    model: str = "timur"


class PicketRequest(BaseModel):
    well_id: str
    top_depth: Optional[float] = None
    bottom_depth: Optional[float] = None
    rw: float = 0.05
    m: float = 2.0
    n: float = 2.0


class CompositeReportRequest(BaseModel):
    well_id: str
    top_depth: float
    bottom_depth: float


class PlotCubeRequest(BaseModel):
    """3D petrophysical cluster cube — no trajectory/highlight options apply."""
    well_id: str
    top_depth: Optional[float] = None
    bottom_depth: Optional[float] = None


class Plot3DRequest(BaseModel):
    """3D wellbore trajectory with optional sweet-spot highlight."""
    well_id: str
    top_depth: Optional[float] = None
    bottom_depth: Optional[float] = None
    color_by: Optional[str] = "sweetspots"
    highlight_top: Optional[float] = None
    highlight_base: Optional[float] = None
    highlight_label: Optional[str] = None
    show_horizon: Optional[bool] = True


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "engine": "pure_python_deterministic",
        "data_dir": str(DATA_DIR)
    }


@app.get("/api/wells")
def list_wells():
    wells = []
    for f in DATA_DIR.glob("*.las"):
        try:
            summary = get_well_curves_summary(f.stem)
            wells.append(summary)
        except Exception as e:
            wells.append({"file": f.name, "error": str(e)})
    return {"wells": wells}


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    try:
        active_query = (req.message or req.query or "").strip()
        if not active_query:
            raise HTTPException(status_code=400, detail="Query message required.")
            
        response = run_agent_turn(active_query, req.chat_history)
        return {
            "text": response.get("text", ""),
            "figures": response.get("figures", []),
            "tool_calls": response.get("tool_calls", []),
            "session_id": req.session_id,
            "well_id": req.well_id or "Well1"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/curves")
def curves_endpoint(well_id: str):
    return get_well_curves_summary(well_id)


@app.post("/api/tools/plot_1d")
def plot_1d_endpoint(req: Plot1DRequest):
    return plot_1d_well_log(req.well_id, req.top_depth, req.bottom_depth, req.marker_depth)


@app.post("/api/tools/crossplot")
def crossplot_endpoint(req: CrossplotRequest):
    return plot_2d_crossplot(req.well_id, req.x_curve, req.y_curve, req.z_curve, req.top_depth, req.bottom_depth)


@app.post("/api/tools/net_pay")
def net_pay_endpoint(req: NetPayRequest):
    return compute_net_pay(req.well_id, req.top_depth, req.bottom_depth, req.vsh_cutoff, req.phi_cutoff, req.sw_cutoff)


@app.post("/api/tools/vsh_comparison")
def vsh_endpoint(req: VshComparisonRequest):
    return compare_vshale_methods(req.well_id, req.top_depth, req.bottom_depth)


@app.post("/api/tools/archie")
def archie_endpoint(req: ArchieRequest):
    return calculate_archie_saturation(req.well_id, req.top_depth, req.bottom_depth, rw=req.rw, m=req.m, n=req.n)


@app.post("/api/tools/sonic_porosity")
def sonic_endpoint(req: SonicRequest):
    return compute_sonic_porosity_wyllie(req.well_id, req.top_depth, req.bottom_depth, req.dt_matrix, req.dt_fluid)


@app.post("/api/tools/sweetspots")
def sweetspots_endpoint(req: SweetspotRequest):
    return scan_reservoir_sweetspots(req.well_id, req.min_thickness)


@app.post("/api/tools/permeability")
def permeability_endpoint(req: PermeabilityRequest):
    return compute_permeability_timur_coates(req.well_id, req.top_depth, req.bottom_depth, req.model)


@app.post("/api/tools/picket")
def picket_endpoint(req: PicketRequest):
    return plot_crossplot_picket(req.well_id, req.top_depth, req.bottom_depth, req.rw, req.m, req.n)


@app.post("/api/tools/composite_report")
def composite_report_endpoint(req: CompositeReportRequest):
    return generate_reservoir_composite_report(req.well_id, req.top_depth, req.bottom_depth)


@app.post("/api/tools/plot_3d_cube")
def plot_3d_cube_endpoint(req: PlotCubeRequest):
    return plot_3d_petrophysical_cube(req.well_id, req.top_depth, req.bottom_depth)


@app.post("/api/tools/plot_3d_trajectory")
def plot_3d_trajectory_endpoint(req: Plot3DRequest):
    return plot_3d_wellbore_trajectory(
        req.well_id, req.top_depth, req.bottom_depth, req.color_by,
        req.highlight_top, req.highlight_base, req.highlight_label,
        req.show_horizon if req.show_horizon is not None else True
    )


@app.get("/api/tools/catalog")
def catalog_endpoint(q: str = "", well: Optional[str] = None):
    return query_catalog(q, well)
