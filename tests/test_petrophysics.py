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
    print("\n>>> ALL 14 PETROPHYSICAL TOOLS PASSED VERIFICATION! <<<")
