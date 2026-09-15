"""Full-stack Hybrid Agentic Well Log RAG & Petrophysical Analytics Dashboard.
"""

import json
import streamlit as st
import plotly.io as pio
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Well Log RAG & Petrophysical Analytics",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a237e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #455a64;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #1976d2;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .badge-weaviate {
        background-color: #00796b;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .badge-tools {
        background-color: #d84315;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Welcome! I am your **Petrophysical AI Assistant**. I can semantically query geological stratigraphy & mudlog data via **Weaviate RAG**, as well as execute deterministic calculations (1D multi-track logs, 2D crossplots, and Net Pay volumetrics) on your `.las` files (`Well1` and `Well2`).\n\nTry one of the quick prompts in the sidebar or ask a question below!",
            "figures": [],
            "tools": []
        }
    ]

if "active_figure" not in st.session_state:
    st.session_state.active_figure = None


# Sidebar: Well Inventory & Quick Actions
with st.sidebar:
    st.image("https://img.icons8.com/color/96/oil-rig.png", width=64)
    st.markdown("### 🛢️ Well Inventory")
    
    try:
        r = requests.get(f"{BACKEND_URL}/api/wells", timeout=5)
        if r.status_code == 200:
            wells_data = r.json().get("wells", [])
            for w in wells_data:
                with st.expander(f"📍 **{w.get('well_name', 'Well')}** ({w.get('uwi')})"):
                    st.write(f"**Depth**: {w.get('start_depth')}m - {w.get('stop_depth')}m")
                    st.write(f"**Step**: {w.get('step')}m")
                    st.write(f"**Curves ({w.get('total_curves')})**:")
                    st.caption(", ".join(w.get("available_mnemonics", [])[:12]) + "...")
        else:
            st.warning("Backend connecting...")
    except Exception:
        st.info("Backend service initializing at `http://127.0.0.1:8000`")

    st.divider()
    st.markdown("### 💡 Quick Prompt Suggestions")
    
    suggestions = [
        ("Stratigraphy Query", "What formation occurs around 1900m in Well 1?"),
        ("1D Multi-Track Log", "Plot the 1D well log curves for Well 1 from 1850m to 1950m."),
        ("2D Lithology Crossplot", "Generate a Density vs Neutron crossplot colored by GR for Well 1."),
        ("Net Pay Volumetrics", "Calculate net pay for Well 1 between 1850m and 1950m with Vsh cutoff 0.3, Porosity cutoff 0.1, and Sw cutoff 0.5."),
        ("Well 2 Deep Target", "Tell me about the hydrocarbon shows in Well 2 and plot its curves between 3600m and 3800m.")
    ]
    
    selected_prompt = None
    for label, prompt in suggestions:
        if st.button(f"👉 {label}", use_container_width=True, key=label):
            selected_prompt = prompt


# Main Header
col_hdr1, col_hdr2 = st.columns([3, 1])
with col_hdr1:
    st.markdown('<div class="main-header">Hybrid Agentic Well Log RAG & Petrophysics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Semantic Stratigraphy & Mudlog Intelligence + Deterministic Multi-Track LAS Tool Engine</div>', unsafe_allow_html=True)
with col_hdr2:
    st.markdown("""
    <div style="text-align: right; padding-top: 10px;">
        <span class="badge-weaviate">Layer 1: Weaviate RAG</span><br>
        <span class="badge-tools" style="margin-top: 4px; display: inline-block;">Layer 2: LAS Tools</span>
    </div>
    """, unsafe_allow_html=True)

# Tabs: Chat & Visualization vs Direct Petrophysical Workbench
tab_agent, tab_manual = st.tabs(["🤖 Agent Chat & Visualization Deck", "🔬 Direct Petrophysical Workbench"])

with tab_agent:
    # Two-Column Side-by-Side Layout
    chat_col, viz_col = st.columns([1, 1], gap="medium")
    
    with chat_col:
        st.markdown("#### 💬 Conversational Petrophysicist")
        chat_container = st.container(height=650)
        
        with chat_container:
            for i, msg in enumerate(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("tools"):
                        with st.expander("🛠️ Executed Tool Calls"):
                            for t in msg["tools"]:
                                st.code(f"Tool: {t.get('name')}\nArgs: {json.dumps(t.get('args'), indent=2)}")
        
        # User input handling
        user_input = st.chat_input("Ask about formations, mudlog shows, net pay cutoffs, or curve plots...")
        active_query = selected_prompt or user_input
        
        if active_query:
            st.session_state.messages.append({"role": "user", "content": active_query, "figures": [], "tools": []})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(active_query)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing geological context & petrophysical logs..."):
                        try:
                            # Format conversation history for API
                            history_payload = [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages[:-1]
                            ]
                            resp = requests.post(
                                f"{BACKEND_URL}/api/chat",
                                json={"query": active_query, "chat_history": history_payload},
                                timeout=45
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                bot_text = data.get("text", "")
                                bot_figs = data.get("figures", [])
                                bot_tools = data.get("tool_calls", [])
                                
                                st.markdown(bot_text)
                                if bot_tools:
                                    with st.expander("🛠️ Executed Tool Calls"):
                                        for t in bot_tools:
                                            st.code(f"Tool: {t.get('name')}\nArgs: {json.dumps(t.get('args'), indent=2)}")
                                
                                if bot_figs:
                                    st.session_state.active_figure = bot_figs[-1]
                                    
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": bot_text,
                                    "figures": bot_figs,
                                    "tools": bot_tools
                                })
                            else:
                                err_msg = f"Backend error: {resp.text}"
                                st.error(err_msg)
                                st.session_state.messages.append({"role": "assistant", "content": err_msg, "figures": [], "tools": []})
                        except Exception as ex:
                            err_msg = f"Connection error: {str(ex)}"
                            st.error(err_msg)
                            st.session_state.messages.append({"role": "assistant", "content": err_msg, "figures": [], "tools": []})
            st.rerun()

    with viz_col:
        st.markdown("#### 📊 Interactive Visualization Deck")
        
        if st.session_state.active_figure:
            try:
                fig_dict = json.loads(st.session_state.active_figure)
                fig = pio.from_json(st.session_state.active_figure)
                st.plotly_chart(fig, use_container_width=True)
                
                # Download figure JSON or HTML
                st.download_button(
                    label="💾 Download Plotly JSON Spec",
                    data=st.session_state.active_figure,
                    file_name="petrophysical_log.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error rendering Plotly chart: {e}")
        else:
            st.info("📈 When you ask the Agent to plot a 1D well log or generate a 2D crossplot, the interactive Plotly figure will appear here dynamically.")
            st.image("https://raw.githubusercontent.com/plotly/datasets/master/well_logs.png", caption="Multi-Track Petrophysical Log Preview", use_container_width=True)


with tab_manual:
    st.markdown("### 🔬 Direct Deterministic Petrophysical Toolkit")
    st.markdown("Execute exact calculations and generate plots directly without natural language routing.")
    
    pcol1, pcol2 = st.columns([1, 2])
    
    with pcol1:
        well_choice = st.selectbox("Select Well", ["Well1", "Well2"], index=0)
        tool_mode = st.radio("Select Workflow", ["1D Multi-Track Log", "2D Lithology Crossplot", "Net Pay Volumetrics"])
        
        if tool_mode == "1D Multi-Track Log":
            default_top = 1850.0 if well_choice == "Well1" else 3600.0
            default_bot = 1950.0 if well_choice == "Well1" else 3800.0
            top_d = st.number_input("Top Depth (m)", value=default_top, step=10.0)
            bot_d = st.number_input("Bottom Depth (m)", value=default_bot, step=10.0)
            execute_btn = st.button("🚀 Render 1D Log", type="primary")
            
        elif tool_mode == "2D Lithology Crossplot":
            x_c = st.selectbox("X Curve", ["NEUT", "NPHI", "DTCOMP"], index=0)
            y_c = st.selectbox("Y Curve", ["DENB", "RHOB", "PEF"], index=0)
            z_c = st.selectbox("Color Curve (Z)", ["GR", "RDEEP", "None"], index=0)
            execute_btn = st.button("🚀 Generate Crossplot", type="primary")
            
        else:
            default_top = 1850.0 if well_choice == "Well1" else 3600.0
            default_bot = 1950.0 if well_choice == "Well1" else 3800.0
            top_d = st.number_input("Top Depth (m)", value=default_top, step=10.0)
            bot_d = st.number_input("Bottom Depth (m)", value=default_bot, step=10.0)
            vsh_cut = st.slider("Vshale Cutoff", 0.05, 0.60, 0.30, 0.05)
            phi_cut = st.slider("Porosity Cutoff", 0.05, 0.30, 0.10, 0.01)
            sw_cut = st.slider("Water Saturation Cutoff", 0.20, 0.80, 0.50, 0.05)
            execute_btn = st.button("🚀 Calculate Net Pay", type="primary")

    with pcol2:
        if execute_btn:
            if tool_mode == "1D Multi-Track Log":
                with st.spinner("Generating multi-track log..."):
                    payload = {"well_id": well_choice, "top_depth": top_d, "bottom_depth": bot_d}
                    res = requests.post(f"{BACKEND_URL}/api/tools/plot_1d", json=payload).json()
                    fig = pio.from_json(res["figure_json"])
                    st.plotly_chart(fig, use_container_width=True)
                    
            elif tool_mode == "2D Lithology Crossplot":
                with st.spinner("Generating crossplot..."):
                    payload = {
                        "well_id": well_choice,
                        "x_curve": x_c,
                        "y_curve": y_c,
                        "z_curve": None if z_c == "None" else z_c
                    }
                    res = requests.post(f"{BACKEND_URL}/api/tools/crossplot", json=payload).json()
                    fig = pio.from_json(res["figure_json"])
                    st.plotly_chart(fig, use_container_width=True)
                    
            else:
                with st.spinner("Calculating cutoffs and volumetrics..."):
                    payload = {
                        "well_id": well_choice,
                        "top_depth": top_d,
                        "bottom_depth": bot_d,
                        "vsh_cutoff": vsh_cut,
                        "phi_cutoff": phi_cut,
                        "sw_cutoff": sw_cut
                    }
                    res = requests.post(f"{BACKEND_URL}/api/tools/net_pay", json=payload).json()
                    
                    st.success(f"Net Pay Computed for {well_choice} ({top_d}m - {bot_d}m)")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Gross Thickness", f"{res.get('gross_interval_m')} m")
                    m2.metric("Net Reservoir", f"{res.get('net_reservoir_m')} m")
                    m3.metric("Net Pay", f"{res.get('net_pay_m')} m")
                    m4.metric("Net-to-Gross (NTG)", f"{res.get('net_to_gross')*100:.1f}%")
                    
                    st.markdown("#### Pay Zone Properties")
                    p_avg = res.get("pay_zone_averages", {})
                    c1, c2, c3 = st.columns(3)
                    c1.info(f"**Avg Porosity (Pay)**: {p_avg.get('average_porosity')*100:.1f}%")
                    c2.info(f"**Avg Water Saturation**: {p_avg.get('average_water_saturation')*100:.1f}%")
                    c3.info(f"**Avg Shale Volume**: {p_avg.get('average_shale_volume')*100:.1f}%")
