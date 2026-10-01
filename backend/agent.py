"""Chat agent: Gemini chooses tools, the MCP server runs them.

Flow per turn: the tool list comes from the MCP server (backend/mcp_server.py)
via the MCP client (backend/mcp_client.py) and is converted to Gemini function
declarations; each Gemini functionCall is executed with an MCP call_tool. The
model only picks tools and fills in arguments; every number shown to the user
comes from the deterministic engine behind the MCP server.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional
import httpx
from backend import mcp_client
from backend.config import GEMINI_API_KEY, GEMINI_MODEL
from backend.petrophysics import default_window, get_well_curves_summary, list_well_ids

logger = logging.getLogger(__name__)

_GEMINI_TYPES = {"string": "STRING", "number": "NUMBER", "integer": "INTEGER",
                 "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT"}


def _to_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an MCP tool's JSON Schema to Gemini's OpenAPI-style subset:
    upper-case types, Optional[X] (anyOf X|null) -> nullable X, defaults moved
    into the description (Gemini's schema has no default field), titles dropped."""
    out: Dict[str, Any] = {}
    options = schema.get("anyOf")
    if options:
        non_null = [o for o in options if o.get("type") != "null"]
        if non_null:
            out.update(_to_gemini_schema(non_null[0]))
        if len(non_null) < len(options):
            out["nullable"] = True
    if schema.get("type") in _GEMINI_TYPES:
        out["type"] = _GEMINI_TYPES[schema["type"]]
    description = schema.get("description", out.get("description", ""))
    if "default" in schema and schema["default"] is not None:
        description = f"{description} (default {schema['default']})".strip()
    if description:
        out["description"] = description
    if "enum" in schema:
        out["enum"] = [str(v) for v in schema["enum"]]
    if schema.get("type") == "object":
        out["properties"] = {k: _to_gemini_schema(v) for k, v in schema.get("properties", {}).items()}
        if schema.get("required"):
            out["required"] = list(schema["required"])
    if schema.get("type") == "array" and "items" in schema:
        out["items"] = _to_gemini_schema(schema["items"])
    return out


def gemini_function_declarations() -> List[Dict[str, Any]]:
    """The MCP server's tools, as Gemini function declarations."""
    return [{"name": t.name, "description": t.description or "", "parameters": _to_gemini_schema(t.input_schema)}
            for t in mcp_client.list_tools()]


SYSTEM_PROMPT = """You are a petrophysics assistant for well log (LAS) data. You answer by calling tools and reporting what they return.
The wells currently loaded (the user can drop in new LAS files at any time) are listed at the end of these instructions; use their exact ids as well_id.

Rules:
1. GROUNDING (most important):
   - Every number you state must come from a tool result in this conversation. Never estimate, recall or invent values.
   - If a question needs something no tool computes (fluid contacts, pressures, completion or perforation advice, core data), say plainly that the data or tool does not provide it. Do not fill the gap.
   - Mention the key assumptions a tool reports (cutoffs, "assumed" defaults, the methodology string, "illustrative" notes).
   - The 3D wellbore path and top-pay surface are illustrative, not surveyed or mapped; never describe their positions or dip as measured.
2. WELLS:
   - Always say which well a result belongs to, using a `### <well id>` header when discussing several.
   - If the question names no well and is not a comparison, use the active well given below.
   - Only call a tool if the well has the curves it needs (listed below); otherwise say which curve is missing.
3. STYLE:
   - Be concise and plain. Explain what a log or number means in simple terms when helpful (e.g. "low gamma ray usually means clean sand").
   - Use short sections or a small table only when they help. No marketing language.
   - Do not add "what to look at" or "what to ask next" sections: the app shows its own, built from the tool results.
"""


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MAX_TOOL_TURNS = 3
TURN_BUDGET_S = 60.0  # hard ceiling for one chat turn across all models

SYNTHESIS_PROMPT = (
    "Summarize the tool results above for the user's question. Use only numbers that appear "
    "in those results, state the main assumptions they report, and say plainly if the "
    "question needs something the tools did not compute."
)


def _post_gemini(model_name: str, payload: Dict[str, Any], timeout: float) -> Optional[Dict[str, Any]]:
    """POST generateContent; returns the JSON body or None (logged) on failure."""
    url = f"{GEMINI_BASE_URL}/{model_name}:generateContent"
    try:
        # Key in a header, not the query string, so it never lands in URLs/logs.
        resp = httpx.post(url, json=payload, timeout=timeout,
                          headers={"x-goog-api-key": GEMINI_API_KEY})
    except httpx.HTTPError as e:
        logger.warning("Gemini %s request failed: %s", model_name, e)
        return None
    if resp.status_code != 200:
        logger.warning("Gemini %s returned HTTP %s: %s", model_name, resp.status_code, resp.text[:300])
        return None
    return resp.json()


def _candidate_parts(resp_json: Dict[str, Any]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    candidate = (resp_json.get("candidates") or [{}])[0]
    content = candidate.get("content", {}) or {}
    return content, content.get("parts", []) or []


def run_agent_turn(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    active_well: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs one chat turn: Gemini picks tools, the MCP server runs them, Gemini summarizes."""
    if not GEMINI_API_KEY:
        return _generate_offline_summary(query, [], [], active_well)

    models_to_try = list(dict.fromkeys([GEMINI_MODEL, "gemini-3.5-flash-lite", "gemini-3.5-flash"]))
    system_text = (SYSTEM_PROMPT + "\nLoaded wells:\n" + _wells_context()
                   + f"\nActive well selected in the UI: {active_well or 'none'}\n")
    declarations = gemini_function_declarations()  # discovered from the MCP server

    contents = []
    if chat_history:
        for msg in chat_history[-6:]:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": query}]})

    generated_figures: List[str] = []
    executed_tools_info: List[Dict[str, Any]] = []
    deadline = time.monotonic() + TURN_BUDGET_S

    def remaining(cap: float) -> float:
        return min(cap, deadline - time.monotonic())

    for model_name in models_to_try:
        if remaining(25.0) <= 1.0:
            logger.warning("Chat turn budget of %.0fs exhausted", TURN_BUDGET_S)
            break
        current_contents = list(contents)

        for _ in range(MAX_TOOL_TURNS):
            timeout = remaining(25.0)
            if timeout <= 1.0:
                break
            resp_json = _post_gemini(model_name, {
                "system_instruction": {"parts": [{"text": system_text}]},
                "contents": current_contents,
                "tools": [{"function_declarations": declarations}],
                "generation_config": {"temperature": 0.2}
            }, timeout)
            if resp_json is None:
                break

            content, parts = _candidate_parts(resp_json)
            function_calls = [p["functionCall"] for p in parts if "functionCall" in p]

            if function_calls:
                current_contents.append(content)
                tool_resps = []
                for fc in function_calls:
                    fn_name = fc["name"]
                    fn_args = fc.get("args", {})
                    tool_result, fig_json = mcp_client.call_tool(fn_name, fn_args)
                    if fig_json:
                        generated_figures.append(fig_json)
                    executed_tools_info.append({"name": fn_name, "args": fn_args, "result": tool_result,
                                                "via": f"MCP ({mcp_client.transport()})"})
                    tool_resps.append({"functionResponse": {"name": fn_name, "response": {"result": tool_result}}})
                current_contents.append({"role": "user", "parts": tool_resps})
                continue

            text_out = "".join(p.get("text", "") for p in parts if "text" in p).strip()
            if text_out:
                return {"text": text_out, "figures": generated_figures, "tool_calls": executed_tools_info}
            break

        # Tools ran but the model hit the turn limit without writing an answer.
        if executed_tools_info and remaining(30.0) > 1.0:
            current_contents.append({"role": "user", "parts": [{"text": SYNTHESIS_PROMPT}]})
            resp_json = _post_gemini(model_name, {
                "system_instruction": {"parts": [{"text": system_text}]},
                "contents": current_contents,
                "generation_config": {"temperature": 0.2}
            }, remaining(30.0))
            if resp_json is not None:
                _, parts = _candidate_parts(resp_json)
                text_out = "".join(p.get("text", "") for p in parts if "text" in p).strip()
                if text_out:
                    return {"text": text_out, "figures": generated_figures, "tool_calls": executed_tools_info}

    # Every model failed (quota, network, timeout): fall back to a fixed-format
    # report built only from deterministic tool results.
    return _generate_offline_summary(query, generated_figures, executed_tools_info, active_well)


_GREETING_RE = re.compile(r"\b(hi|hello|hey|help|who are you|what can you do|capabilities|overview)\b")
MAX_OFFLINE_WELLS = 4


def _wells_context() -> str:
    """One line per loaded well for the system prompt (ids, range, curves)."""
    lines = []
    for well in list_well_ids():
        try:
            s = get_well_curves_summary(well)
        except Exception as e:
            lines.append(f"- {well}: could not be read ({e})")
            continue
        win = s["default_window"]
        missing = [k for k, v in s["capabilities"].items() if not v]
        lines.append(
            f"- {s['well_id']}: {s['start_depth']}-{s['stop_depth']} m MD, curves "
            f"{', '.join(s['available_mnemonics'])}. Suggested starting window {win['top']}-{win['bottom']} m."
            + (f" Missing: {', '.join(missing)}." if missing else ""))
    return "\n".join(lines) or "- (no wells loaded; ask the user to drop a .las file)"


def _wells_named_in(query: str, well_ids: List[str]) -> List[str]:
    """Well ids mentioned in the query; spaces are ignored so 'well 1' matches 'Well1'."""
    compact = re.sub(r"\s+", "", query.lower())
    return [w for w in well_ids if re.search(re.escape(w.lower()) + r"(?![0-9a-z])", compact)]


def _fmt(value: Any, spec: str, scale: float = 1.0) -> str:
    return format(value * scale, spec) if isinstance(value, (int, float)) else "n/a"


def _generate_offline_summary(
    query: str,
    figures: List[str],
    executed_tools: List[Dict[str, Any]],
    active_well: Optional[str] = None,
) -> Dict[str, Any]:
    """Fixed-format report used when the LLM is unavailable.

    Every number printed here comes from a tool run in this function (recorded
    in tool_calls for the UI audit badges); nothing is hard-coded or guessed.
    """
    q = query.lower()

    def run(name: str, args: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[str]]:
        result, fig = mcp_client.call_tool(name, args)
        executed_tools.append({"name": name, "args": args, "result": result,
                               "via": f"MCP ({mcp_client.transport()})"})
        return result, fig

    note = ("_Offline summary: the language model is unavailable, so this is a fixed-format report "
            "built only from tool results. Ask again later for a written interpretation._")

    all_wells = list_well_ids()
    if not all_wells:
        return {"text": note + "\n\nNo wells are loaded yet. Drop a .las file to get started.",
                "figures": figures, "tool_calls": executed_tools}

    # Short greetings / "what can you do" -> list the data actually loaded.
    if _GREETING_RE.search(q) and len(q.split()) <= 6:
        lines = [note, "", "### Loaded wells"]
        for well in all_wells:
            info, _ = run("get_well_curves_summary", {"well_id": well})
            if "error" in info:
                lines.append(f"* **{well}**: could not be read ({info['error']})")
            else:
                lines.append(f"* **{well}**: {info['start_depth']}–{info['stop_depth']} m MD, "
                             f"{info['total_curves']} curves")
        return {"text": "\n".join(lines), "figures": figures, "tool_calls": executed_tools}

    named = _wells_named_in(query, all_wells)
    if "compare" in q or "both" in q or "all wells" in q:
        wells = (named if len(named) > 1 else all_wells)[:MAX_OFFLINE_WELLS]
    elif named:
        wells = named[:MAX_OFFLINE_WELLS]
    else:
        wells = [active_well if active_well in all_wells else all_wells[0]]

    sections = [note]
    for well in wells:
        sections.append(f"\n### {well}")
        scan, _ = run("scan_reservoir_sweetspots", {"well_id": well, "min_thickness": 1.5})
        if "error" in scan:
            sections.append(f"Sweet-spot scan failed: {scan['error']}")
            continue
        zones = scan.get("sweetspots", [])
        n_zones = scan.get('total_sweetspots_found', 0)
        sections.append(f"Sweet-spot scan (Vsh ≤ 0.30, φ ≥ 0.10, Sw ≤ 0.50, zones ≥ 1.5 m): "
                        f"**{n_zones}** zone{'' if n_zones == 1 else 's'}, "
                        f"**{_fmt(scan.get('total_pay_thickness_m'), '.2f')} m** of pay in total.")

        if zones:
            best = zones[0]
            top, base = best["top_depth"], best["base_depth"]
            comp, _ = run("generate_reservoir_composite_report",
                          {"well_id": well, "top_depth": top, "bottom_depth": base})
            if "error" in comp:
                sections.append(f"Composite report failed: {comp['error']}")
            else:
                sections += [
                    f"\nHighest-ranked zone: **{top}–{base} m** ({best['thickness_m']} m).",
                    "",
                    "| Metric | Value |",
                    "| --- | --- |",
                    f"| Net pay | {_fmt(comp.get('net_pay_m'), '.2f')} m |",
                    f"| Net-to-gross | {_fmt(comp.get('net_to_gross'), '.2f')} |",
                    f"| Average porosity | {_fmt(comp.get('average_porosity'), '.1f', 100)} % |",
                    f"| Average water saturation | {_fmt(comp.get('average_water_saturation'), '.1f', 100)} % |",
                    f"| Average shale volume | {_fmt(comp.get('average_shale_volume'), '.1f', 100)} % |",
                    f"| Flow capacity kh (Timur, pay only) | {_fmt(comp.get('flow_capacity_kh_md_m'), '.1f')} mD·m |",
                    f"| Rough fluid label (from Sw and porosity only) | {comp.get('interpreted_fluid_regime', 'n/a')} |",
                ]
            plot_top, plot_bot = top - 20.0, base + 20.0
        else:
            sections.append("No zone passed all cutoffs.")
            win = default_window(well)
            plot_top, plot_bot = win["top"], win["bottom"]

        if not figures:
            _, fig = run("plot_1d_well_log", {"well_id": well, "top_depth": plot_top, "bottom_depth": plot_bot})
            if fig:
                figures.append(fig)

    return {"text": "\n".join(sections).strip(), "figures": figures, "tool_calls": executed_tools}
