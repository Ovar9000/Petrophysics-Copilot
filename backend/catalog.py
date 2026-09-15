"""Lightweight in-memory Well & Stratigraphy Catalog.

Replaces heavy external vector DB infrastructure with zero-latency, pure-Python
structured search over LAS headers and geological reports.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import lasio

from backend.config import DATA_DIR

_CATALOG_CACHE: List[Dict[str, Any]] = []


def init_catalog() -> List[Dict[str, Any]]:
    """Loads well headers and markdown geology reports into memory."""
    global _CATALOG_CACHE
    _CATALOG_CACHE = []

    wells = [
        {"file": "Well1.las", "report": "Well1_geology_report.md", "id": "Well1"},
        {"file": "Well2.las", "report": "Well2_geology_report.md", "id": "Well2"}
    ]

    for item in wells:
        las_path = DATA_DIR / item["file"]
        rep_path = DATA_DIR / item["report"]
        if not las_path.exists():
            continue

        las = lasio.read(str(las_path))
        well_name = str(las.well.WELL.value if "WELL" in las.well else item["id"]).strip()
        uwi = str(las.well.UWI.value if "UWI" in las.well else "UNKNOWN").strip()
        start_depth = float(las.well.STRT.value) if "STRT" in las.well else 0.0
        stop_depth = float(las.well.STOP.value) if "STOP" in las.well else 3000.0
        step = float(las.well.STEP.value) if "STEP" in las.well else 0.1524
        curves = [str(c.mnemonic).strip() for c in las.curves]

        formation_tops = ""
        lithology_notes = ""
        if rep_path.exists():
            content = rep_path.read_text(encoding="utf-8")
            parts = content.split("##")
            for p in parts:
                if "Stratigraphy & Formation Tops" in p:
                    formation_tops = p.strip()
                elif "Petrophysical & Mudlog Evaluation" in p:
                    lithology_notes = p.strip()
            if not formation_tops:
                formation_tops = content[:1200]
            if not lithology_notes:
                lithology_notes = content[1200:]

        record = {
            "well_name": well_name,
            "uwi": uwi,
            "start_depth": round(start_depth, 2),
            "stop_depth": round(stop_depth, 2),
            "step": round(step, 4),
            "available_curves": curves,
            "formation_tops": formation_tops,
            "lithology_notes": lithology_notes,
            "search_corpus": f"{well_name} {uwi} {formation_tops} {lithology_notes}".lower()
        }
        _CATALOG_CACHE.append(record)

    return _CATALOG_CACHE


def query_catalog(query_text: str, well_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Instant keyword and relevance search across stratigraphy catalog."""
    global _CATALOG_CACHE
    if not _CATALOG_CACHE:
        init_catalog()

    q_lower = query_text.lower()
    q_words = [w for w in q_lower.split() if len(w) > 2]
    
    scored: List[tuple[int, Dict[str, Any]]] = []
    for item in _CATALOG_CACHE:
        if well_name and well_name.lower() not in item["well_name"].lower():
            continue

        score = 0
        for w in q_words:
            if w in item["search_corpus"]:
                score += 1
                
        clean_item = {k: v for k, v in item.items() if k != "search_corpus"}
        scored.append((score, clean_item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored]


# Initialize at import
init_catalog()
