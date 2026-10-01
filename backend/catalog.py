"""Lightweight in-memory Well & Stratigraphy Catalog.

Replaces heavy external vector DB infrastructure with zero-latency, pure-Python
structured search over LAS headers and geological reports.
"""

from typing import Any, Dict, List, Optional
import lasio

from backend.config import DATA_DIR

_CATALOG_CACHE: List[Dict[str, Any]] = []
_CATALOG_SIGNATURE: tuple = ()


def _data_signature() -> tuple:
    """Names + modification times of the files the catalog is built from. The
    MCP server runs in its own process, so it cannot be told about an upload;
    comparing this signature lets each process notice changes itself."""
    files = list(DATA_DIR.glob("*.las")) + list(DATA_DIR.glob("*_geology_report.md"))
    return tuple(sorted((p.name, p.stat().st_mtime) for p in files))


def init_catalog() -> List[Dict[str, Any]]:
    """Loads well headers and markdown geology reports into memory."""
    global _CATALOG_CACHE, _CATALOG_SIGNATURE
    _CATALOG_CACHE = []
    _CATALOG_SIGNATURE = _data_signature()

    # Every LAS in DATA_DIR (including uploads); a geology report is optional.
    wells = [
        {"file": p.name, "report": f"{p.stem}_geology_report.md", "id": p.stem}
        for p in sorted(DATA_DIR.glob("*.las"))
    ]

    for item in wells:
        las_path = DATA_DIR / item["file"]
        rep_path = DATA_DIR / item["report"]
        if not las_path.exists():
            continue

        try:
            las = lasio.read(str(las_path))
        except Exception:
            continue  # an unreadable file must not take the whole catalog down
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
            "well_id": item["id"],
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


def _score_item(query_words: List[str], item: Dict[str, Any]) -> int:
    """Count query-word hits in the item corpus (length-normalized later if needed)."""
    return sum(1 for w in query_words if w in item["search_corpus"])


def query_catalog(query_text: str, well_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Instant keyword and relevance search across stratigraphy catalog."""
    global _CATALOG_CACHE
    if not _CATALOG_CACHE or _CATALOG_SIGNATURE != _data_signature():
        init_catalog()

    q_words = [w for w in query_text.lower().split() if len(w) > 2]

    scored: List[tuple[int, Dict[str, Any]]] = []
    for item in _CATALOG_CACHE:
        # Match the file-based well id exactly, or a substring of the LAS header name
        wanted = (well_name or "").lower()
        if wanted and wanted != item["well_id"].lower() and wanted not in item["well_name"].lower():
            continue

        clean_item = {k: v for k, v in item.items() if k != "search_corpus"}
        scored.append((_score_item(q_words, item), clean_item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored]
