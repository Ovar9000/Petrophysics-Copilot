"""Comprehensive test suite for all 8 deterministic petrophysical tools and catalog.
"""

import sys
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

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


def test_catalog():
    records = init_catalog()
    assert len(records) >= 2
    res = query_catalog("formation tops", well_name="Well1")
    assert len(res) > 0
    assert "formation_tops" in res[0]
    print("[PASS] Catalog initialized and queried successfully.")


def test_curves_summary():
    s1 = get_well_curves_summary("Well1")
    assert s1["total_curves"] > 5
    assert "DEPTH" in s1["available_mnemonics"] or "GR" in s1["available_mnemonics"]

    s2 = get_well_curves_summary("Well2")
    assert "GR" in s2["available_mnemonics"]
    print("[PASS] Curves summary verified for Well1 and Well2.")


def test_plot_1d():
    res = plot_1d_well_log("Well1", top_depth=1850.0, bottom_depth=1950.0)
    assert "figure_json" in res
    fig = json.loads(res["figure_json"])
    assert len(fig["data"]) >= 3
    print("[PASS] 1D multi-track composite log generated.")


def test_crossplot():
    res = plot_2d_crossplot("Well1", x_curve="NEUT", y_curve="DENB", z_curve="GR", top_depth=1850.0, bottom_depth=1950.0)
    assert "figure_json" in res
    assert res["data_points"] > 0
    print("[PASS] 2D Lithology crossplot with matrix lines verified.")


def test_net_pay():
    res = compute_net_pay("Well1", top_depth=1850.0, bottom_depth=1950.0, vsh_cutoff=0.3, phi_cutoff=0.1, sw_cutoff=0.5)
    assert res["gross_interval_m"] > 0
    assert res["net_pay_m"] > 0
    assert 0.0 < res["net_to_gross"] <= 1.0
    print("[PASS] Net pay volumetric calculation verified.")


def test_vsh_methods():
    res = compare_vshale_methods("Well1", top_depth=1850.0, bottom_depth=1950.0)
    assert "averages" in res
    assert 0.0 <= res["averages"]["larionov_avg"] <= 1.0
    assert "figure_json" in res
    print("[PASS] Multi-method shale volume comparison verified.")


def test_archie_saturation():
    res = calculate_archie_saturation("Well1", top_depth=1850.0, bottom_depth=1950.0, rw=0.05, m=2.0, n=2.0)
    assert "average_sw" in res
    assert "average_bvh" in res
    assert "figure_json" in res
    print("[PASS] Archie saturation and Bulk Volume Hydrocarbon verified.")


def test_sonic_porosity():
    res = compute_sonic_porosity_wyllie("Well1", top_depth=1850.0, bottom_depth=1950.0)
    assert "avg_sonic_porosity" in res
    assert res["avg_sonic_porosity"] > 0
    assert "figure_json" in res
    print("[PASS] Wyllie time-average sonic porosity verified.")


def test_sweetspot_scanner():
    res = scan_reservoir_sweetspots("Well1", min_thickness=1.5)
    assert res["total_sweetspots_found"] >= 1
    assert len(res["sweetspots"]) >= 1
    top_zone = res["sweetspots"][0]
    assert top_zone["thickness_m"] > 0
    assert top_zone["avg_porosity"] > 0.1
    print("[PASS] Automated reservoir sweet spot scanner verified.")


def test_permeability():
    res = compute_permeability_timur_coates("Well1", top_depth=1850.0, bottom_depth=1950.0)
    assert "average_permeability_md" in res
    assert res["average_permeability_md"] > 0
    assert res["flow_capacity_kh_md_m"] > 0
    assert "figure_json" in res
    print("[PASS] Permeability & flow capacity (kh) verified.")


def test_picket_plot():
    res = plot_crossplot_picket("Well1", top_depth=1850.0, bottom_depth=1950.0)
    assert "samples_plotted" in res
    assert res["samples_plotted"] > 0
    assert "figure_json" in res
    print("[PASS] Archie Picket crossplot verified.")


def test_reservoir_composite_report():
    res = generate_reservoir_composite_report("Well1", top_depth=1900.0, bottom_depth=1925.0)
    assert res["net_pay_m"] > 0
    assert res["hydrocarbon_pore_volume_hcpv_m"] > 0
    assert "flow_capacity_kh_md_m" in res
    print("[PASS] Reservoir composite petrophysical dossier verified.")


def test_3d_cube():
    res = plot_3d_petrophysical_cube("Well1", top_depth=1850.0, bottom_depth=1950.0)
    assert res["samples_rendered"] > 0
    assert "figure_json" in res
    print("[PASS] 3D Petrophysical Cluster Cube verified.")


def test_3d_trajectory():
    res = plot_3d_wellbore_trajectory("Well1", top_depth=1850.0, bottom_depth=1950.0)
    assert res["total_depth_samples"] > 0
    assert "figure_json" in res
    print("[PASS] 3D Subsurface Wellbore Trajectory & Horizon verified.")


def test_figure_meta_tags():
    from backend.petrophysics import plot_3d_petrophysical_cube
    cases = [
        (plot_1d_well_log("Well1", top_depth=1850.0, bottom_depth=1950.0), "1d", "Well1"),
        (plot_2d_crossplot("Well1", x_curve="NEUT", y_curve="DENB", z_curve="GR", top_depth=1850.0, bottom_depth=1950.0), "2d", "Well1"),
        (plot_3d_wellbore_trajectory("Well2", top_depth=3649.0, bottom_depth=3734.0), "3d", "Well2"),
        (plot_3d_petrophysical_cube("Well1", top_depth=1850.0, bottom_depth=1950.0), "3d", "Well1"),
    ]
    for res, kind, well in cases:
        meta = json.loads(res["figure_json"])["layout"].get("meta") or {}
        assert meta.get("plot_kind") == kind, f"plot_kind for {well}: {meta}"
        assert meta.get("well_id") == well, f"well_id tag: {meta}"
    print("[PASS] Figure layout.meta plot_kind + well_id tags verified.")


# --- Regression tests for the audit findings -------------------------------

def test_ntg_never_exceeds_one_in_all_pay_zone():
    zone = scan_reservoir_sweetspots("Well1")["sweetspots"][0]
    res = compute_net_pay("Well1", zone["top_depth"], zone["base_depth"])
    assert res["net_pay_m"] <= res["gross_interval_m"]
    assert res["net_to_gross"] <= 1.0


def test_scanner_and_net_pay_agree_on_same_zone():
    for well in ("Well1", "Well2"):
        zone = scan_reservoir_sweetspots(well)["sweetspots"][0]
        res = compute_net_pay(well, zone["top_depth"], zone["base_depth"])
        assert abs(res["net_pay_m"] - zone["thickness_m"]) < 0.01, (well, zone, res["net_pay_m"])


def test_net_pay_is_additive_across_windows():
    # Well2 has no VSHALE log, so this exercises the GR-derived Vsh path.
    # Split at a depth between samples so no sample is counted twice.
    whole = compute_net_pay("Well2", 3600.0, 3700.0)["net_pay_m"]
    upper = compute_net_pay("Well2", 3600.0, 3650.05)["net_pay_m"]
    lower = compute_net_pay("Well2", 3650.05, 3700.0)["net_pay_m"]
    assert abs((upper + lower) - whole) < 0.01


def test_kh_counts_pay_only():
    res = compute_permeability_timur_coates("Well2", 3590.0, 3850.0)
    net = compute_net_pay("Well2", 3590.0, 3850.0)
    assert res["net_pay_samples"] > 0
    # Sum over pay samples only: kh / avg k is the pay thickness.
    assert abs(res["flow_capacity_kh_md_m"] / res["average_permeability_md"] - net["net_pay_m"]) < 0.5


def test_well_id_cannot_escape_data_dir():
    import pytest
    for bad in ("../data/Well1.las", str(BASE_DIR / "data" / "Well1.las"), "..\\Well1"):
        with pytest.raises(FileNotFoundError):
            get_well_curves_summary(bad)


def test_sonic_empty_interval_raises():
    import pytest
    with pytest.raises(ValueError):
        compute_sonic_porosity_wyllie("Well1", 99990.0, 99999.0)


def test_api_error_codes():
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post("/api/chat", json={"message": "  "}).status_code == 400
    assert client.post("/api/tools/sonic_porosity",
                       json={"well_id": "Well1", "top_depth": 99990, "bottom_depth": 99999}).status_code == 400
    assert client.post("/api/tools/net_pay",
                       json={"well_id": "NoSuchWell", "top_depth": 1, "bottom_depth": 2}).status_code == 404


def test_offline_summary_is_grounded():
    from backend.agent import _generate_offline_summary
    out = _generate_offline_summary("What is the net pay in this interval of Well 2?", [], [], "Well1")
    assert "Offline summary" in out["text"]
    assert "### Well2" in out["text"]  # "this" must not trigger the greeting branch
    assert {t["name"] for t in out["tool_calls"]} >= {"scan_reservoir_sweetspots", "generate_reservoir_composite_report"}
    # Every value in the table must come from the recorded composite report.
    comp = next(t["result"] for t in out["tool_calls"] if t["name"] == "generate_reservoir_composite_report")
    assert f"{comp['net_pay_m']:.2f} m" in out["text"]


def test_mcp_server_imports_as_script():
    import subprocess
    code = ("import runpy, sys; sys.argv=['x']; "
            "g = runpy.run_path(r'%s', run_name='not_main'); print('ok')" % (BASE_DIR / "backend" / "mcp_server.py"))
    out = subprocess.run([sys.executable, "-c", code], cwd=str(BASE_DIR / "backend"),
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]


if __name__ == "__main__":
    test_catalog()
    test_curves_summary()
    test_plot_1d()
    test_crossplot()
    test_net_pay()
    test_vsh_methods()
    test_archie_saturation()
    test_sonic_porosity()
    test_sweetspot_scanner()
    test_permeability()
    test_picket_plot()
    test_reservoir_composite_report()
    test_3d_cube()
    test_3d_trajectory()
    test_figure_meta_tags()
    print("\n>>> ALL 14 PETROPHYSICAL TOOLS PASSED VERIFICATION! <<<")
