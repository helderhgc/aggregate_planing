"""
app.py — Aggregate Planning Interactive Dashboard
Professor tool for teaching Production Planning & Control.
Run: streamlit run app.py
"""

import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from solvers import (
    solve_chase, solve_level, solve_mixed,
    solve_lp, solve_transportation, solve_trial,
    compare_all,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Aggregate Planning Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {background:#f0f4f8;border-radius:8px;padding:12px 18px;text-align:center;}
    .metric-label {font-size:0.78rem;color:#555;font-weight:600;text-transform:uppercase;}
    .metric-value {font-size:1.6rem;font-weight:700;color:#1f77b4;}
    .section-title {font-size:1.1rem;font-weight:700;color:#1a1a2e;
                    border-left:4px solid #1f77b4;padding-left:8px;margin-top:1rem;}
    .warning-box {background:#fff3cd;border-left:4px solid #ffc107;
                  padding:10px 14px;border-radius:4px;}
    .info-box   {background:#d1ecf1;border-left:4px solid #17a2b8;
                 padding:10px 14px;border-radius:4px;}
    .shadow-box {background:#e8f4f8;border-radius:8px;padding:14px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load default scenario
# ─────────────────────────────────────────────
@st.cache_data
def load_defaults():
    p = Path(__file__).parent / "data" / "default_scenario.json"
    with open(p) as f:
        return json.load(f)

DEFAULTS = load_defaults()

# ─────────────────────────────────────────────
# Sidebar — Parameter Control Panel
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📦 Aggregate Planning")
    st.caption("Production Planning & Control — Interactive Dashboard")
    st.caption("👨‍🏫 Prof. Helder Costa · heldergc@id.uff.br")
    st.divider()

    # Theory download button
    _theory_path = Path(__file__).parent / "theory.html"
    if _theory_path.exists():
        with open(_theory_path, "rb") as _f:
            st.download_button(
                label="📖 Download Theory Guide (HTML)",
                data=_f.read(),
                file_name="aggregate_planning_theory.html",
                mime="text/html",
                width='stretch',
            )
        st.caption("Open the downloaded file in any browser — works offline.")
        st.divider()

    # Forecast section
    st.markdown("### 📊 Forecast")
    preset_names = ["Custom"] + list(DEFAULTS["presets"].keys())
    preset_choice = st.selectbox("📂 Load Preset Scenario", preset_names)

    # Strategy selector — kept at top for easy access
    st.markdown("### 🎯 Strategy")
    strategy = st.selectbox("Select Strategy", [
        "Chase Demand",
        "Level Production",
        "Mixed / Hybrid",
        "Linear Programming (LP)",
        "Transportation Method",
        "Trial-and-Error",
    ])
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload demand series (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help=(
            "CSV: two columns — 'Period' and 'Demand'.\n"
            "Excel: same layout on the first sheet.\n"
            "Any number of periods is supported."
        ),
    )

    # Parse uploaded file
    uploaded_periods = None
    uploaded_demand  = None
    upload_error     = None

    if uploaded_file is not None:
        try:
            fname = uploaded_file.name.lower()
            if fname.endswith(".csv"):
                df_up = pd.read_csv(uploaded_file)
            else:
                df_up = pd.read_excel(uploaded_file, engine="openpyxl")

            # Normalise column names (case-insensitive, strip spaces)
            df_up.columns = [c.strip().lower() for c in df_up.columns]

            # Accept "period"/"periods"/"month"/"months" and "demand"/"demands"/"forecast"
            period_col = next(
                (c for c in df_up.columns if c in ("period","periods","month","months","time")),
                df_up.columns[0],
            )
            demand_col = next(
                (c for c in df_up.columns if c in ("demand","demands","forecast","qty","quantity")),
                df_up.columns[1] if len(df_up.columns) > 1 else None,
            )
            if demand_col is None:
                raise ValueError("Could not find a demand column. Expected header: 'Demand'.")

            df_up = df_up[[period_col, demand_col]].dropna()
            df_up[demand_col] = pd.to_numeric(df_up[demand_col], errors="coerce")
            if df_up[demand_col].isna().any():
                raise ValueError("Non-numeric values found in the Demand column.")

            uploaded_periods = [str(v) for v in df_up[period_col].tolist()]
            uploaded_demand  = [float(v) for v in df_up[demand_col].tolist()]

            if len(uploaded_periods) < 2:
                raise ValueError("At least 2 periods are required.")

        except Exception as e:
            upload_error = str(e)

    if upload_error:
        st.error(f"⚠️ Upload error: {upload_error}")

    # Decide active periods and demand source
    if uploaded_periods and not upload_error:
        active_periods = uploaded_periods
        st.success(f"✅ Uploaded: {len(active_periods)} periods detected.")
        st.dataframe(
            pd.DataFrame({"Period": uploaded_periods, "Demand": uploaded_demand}),
            width='stretch', hide_index=True, height=200,
        )
        # Still allow fine-tuning via number inputs
        st.caption("Fine-tune values below if needed:")
        demand_vals = []
        cols = st.columns(2)
        for i, (period, dval) in enumerate(zip(active_periods, uploaded_demand)):
            val = cols[i % 2].number_input(
                period, min_value=0, max_value=99999,
                value=int(dval), step=10, key=f"d_{i}"
            )
            demand_vals.append(float(val))
    else:
        # Preset / manual inputs
        if preset_choice != "Custom":
            preset_demand = DEFAULTS["presets"][preset_choice]["demand"]
        else:
            preset_demand = DEFAULTS["demand"]
        active_periods = DEFAULTS["periods"]
        demand_vals = []
        cols = st.columns(2)
        for i, period in enumerate(active_periods):
            val = cols[i % 2].number_input(
                period, min_value=0, max_value=9999,
                value=int(preset_demand[i]), step=10, key=f"d_{i}"
            )
            demand_vals.append(float(val))

    st.divider()

    # Initial conditions
    st.markdown("### 🏭 Initial Conditions")
    c1, c2 = st.columns(2)
    init_inv = c1.number_input("Starting Inventory", 0, 5000, DEFAULTS["initial_inventory"], 10)
    init_wf  = c2.number_input("Starting Workforce", 1, 200,  DEFAULTS["initial_workforce"], 1)
    productivity = st.number_input("Units / Worker / Period", 1, 500,
                                   DEFAULTS["productivity"], 1)
    st.divider()

    # Cost parameters
    st.markdown("### 💰 Cost Parameters ($ per unit / worker)")
    costs = {}
    cost_keys = [
        ("regular_time", "Regular-Time Labour ($/unit)"),
        ("hiring",       "Hiring Cost ($/worker)"),
        ("firing",       "Firing / Layoff Cost ($/worker)"),
        ("holding",      "Inventory Holding ($/unit·period)"),
        ("backorder",    "Backorder Penalty ($/unit·period)"),
        ("overtime",     "Overtime ($/unit)"),
        ("subcontracting","Subcontracting ($/unit)"),
    ]
    for key, label in cost_keys:
        costs[key] = st.number_input(label, 0, 99999,
                                     int(DEFAULTS["costs"][key]), 10,
                                     key=f"c_{key}")

    st.divider()

    # Capacity
    st.markdown("### ⚙️ Capacity Constraints")
    max_wf    = st.number_input("Max Workforce",             1,  500, int(DEFAULTS["capacity"]["max_workforce"]),  1)
    max_ot    = st.slider("Max Overtime Fraction",           0.0, 0.5, float(DEFAULTS["capacity"]["max_overtime_fraction"]), 0.05)
    max_sub   = st.number_input("Max Subcontract / Period",  0, 5000, int(DEFAULTS["capacity"]["max_subcontract_per_period"]), 50)

    capacity = {
        "max_workforce": max_wf,
        "max_overtime_fraction": max_ot,
        "max_subcontract_per_period": max_sub,
    }
    st.divider()

    # Variable Types
    st.markdown("### 🔢 Variable Types")
    st.caption("Force decision variables to be whole numbers (integers).")
    integer_wf   = st.checkbox(
        "Integer workforce — hired/fired/workers must be whole numbers",
        value=False,
        help="Applies to workforce level, hiring, and firing variables.",
    )
    integer_prod = st.checkbox(
        "Integer production — units produced / OT / subcontract must be whole numbers",
        value=False,
        help="Applies to regular-time production, overtime, and subcontracted units.",
    )
    st.divider()

    # Shortage Policy
    st.markdown("### 📋 Shortage Policy")
    shortage_policy = st.radio(
        "How to handle unmet demand:",
        options=["backorders", "no_shortages", "lost_sales"],
        format_func=lambda x: {
            "backorders":   "Backorders — demand fulfilled late (penalty cost)",
            "no_shortages": "No Shortages — demand must always be met",
            "lost_sales":   "Lost Sales — unmet demand is lost forever",
        }[x],
        index=0,
        help="Backorders allow negative inventory. No-shortages forces feasibility. Lost sales drop demand permanently.",
    )
    if shortage_policy == "lost_sales":
        costs["lost_sales"] = st.number_input(
            "Lost Sales Penalty ($/unit)",
            min_value=0, max_value=99999,
            value=int(DEFAULTS["costs"].get("lost_sales", 50)),
            step=10, key="c_lost_sales",
        )
    else:
        costs["lost_sales"] = 0.0
    st.divider()


# ─────────────────────────────────────────────
# Shared kwargs
# ─────────────────────────────────────────────
solver_kwargs = dict(
    periods=active_periods,
    demand=demand_vals,
    costs=costs,
    capacity=capacity,
    initial_workforce=float(init_wf),
    initial_inventory=float(init_inv),
    productivity=float(productivity),
)

# ─────────────────────────────────────────────
# Run solver
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_solver(strategy_name, periods_tuple, demand_tuple, costs_tuple, capacity_tuple,
               init_wf, init_inv, prod,
               trial_production=None, trial_workforce=None,
               trial_ot=None, trial_sub=None,
               integer_workforce=False, integer_production=False,
               shortage_policy="backorders"):
    kw = dict(
        periods=list(periods_tuple),
        demand=list(demand_tuple),
        costs=dict(costs_tuple),
        capacity=dict(capacity_tuple),
        initial_workforce=init_wf,
        initial_inventory=init_inv,
        productivity=prod,
        integer_workforce=integer_workforce,
        integer_production=integer_production,
        shortage_policy=shortage_policy,
    )
    if strategy_name == "Chase Demand":
        return solve_chase(**kw)
    if strategy_name == "Level Production":
        return solve_level(**kw)
    if strategy_name == "Mixed / Hybrid":
        return solve_mixed(**kw)
    if strategy_name == "Linear Programming (LP)":
        return solve_lp(**kw)
    if strategy_name == "Transportation Method":
        return solve_transportation(**kw)
    if strategy_name == "Trial-and-Error":
        return solve_trial(**kw,
                           user_production=list(trial_production) if trial_production else None,
                           user_workforce=list(trial_workforce)  if trial_workforce  else None,
                           user_overtime=list(trial_ot)          if trial_ot         else None,
                           user_subcontract=list(trial_sub)       if trial_sub        else None)
    return solve_chase(**kw)


# ─────────────────────────────────────────────
# Main content header
# ─────────────────────────────────────────────
st.title("Aggregate Planning Dashboard")
st.caption(f"Strategy in focus: **{strategy}** — adjust parameters in the sidebar and results update instantly.")

periods = active_periods
T = len(periods)

# ─────────────────────────────────────────────
# Trial-and-Error editable table (shown before solving)
# ─────────────────────────────────────────────
trial_prod = None
trial_wf   = None
trial_ot   = None
trial_sub  = None

if strategy == "Trial-and-Error":
    st.markdown("### ✏️ Enter Your Plan")
    st.info("Edit the table below to define your production plan. Click anywhere outside the cell to confirm. The dashboard will compute total costs automatically.")

    # Build editable dataframe
    chase_ref = solve_chase(**solver_kwargs)
    default_te = pd.DataFrame({
        "Period":      periods,
        "Demand":      [int(d) for d in demand_vals],
        "Production":  [int(p) for p in chase_ref["production"]],
        "Workforce":   [int(w) for w in chase_ref["workforce"]],
        "Overtime Units": [0] * T,
        "Subcontract Units": [0] * T,
    })
    edited = st.data_editor(
        default_te,
        width='stretch',
        num_rows="fixed",
        column_config={
            "Period":   st.column_config.TextColumn("Period", disabled=True),
            "Demand":   st.column_config.NumberColumn("Demand", disabled=True),
            "Production": st.column_config.NumberColumn("Production", min_value=0),
            "Workforce":  st.column_config.NumberColumn("Workforce",  min_value=0),
            "Overtime Units":    st.column_config.NumberColumn("Overtime", min_value=0),
            "Subcontract Units": st.column_config.NumberColumn("Subcontract", min_value=0),
        },
        key="trial_editor",
    )
    trial_prod = list(edited["Production"].astype(float))
    trial_wf   = list(edited["Workforce"].astype(float))
    trial_ot   = list(edited["Overtime Units"].astype(float))
    trial_sub  = list(edited["Subcontract Units"].astype(float))

# ─────────────────────────────────────────────
# Run solver (cached)
# ─────────────────────────────────────────────
with st.spinner("Computing plan…"):
    result = run_solver(
        strategy_name=strategy,
        periods_tuple=tuple(active_periods),
        demand_tuple=tuple(demand_vals),
        costs_tuple=tuple(sorted(costs.items())),
        capacity_tuple=tuple(sorted(capacity.items())),
        init_wf=float(init_wf),
        init_inv=float(init_inv),
        prod=float(productivity),
        trial_production=tuple(trial_prod) if trial_prod else None,
        trial_workforce=tuple(trial_wf)   if trial_wf   else None,
        trial_ot=tuple(trial_ot)           if trial_ot   else None,
        trial_sub=tuple(trial_sub)         if trial_sub  else None,
        integer_workforce=bool(integer_wf),
        integer_production=bool(integer_prod),
        shortage_policy=shortage_policy,
    )

if not result["feasible"]:
    st.markdown(f'<div class="warning-box">⚠️ <strong>Warning:</strong> {result["message"]}</div>',
                unsafe_allow_html=True)
elif result["message"]:
    st.markdown(f'<div class="info-box">ℹ️ {result["message"]}</div>',
                unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, label, value, fmt=",.0f", delta=None):
    col.metric(label=label, value=f"${value:{fmt}}" if "$" in label else f"{value:{fmt}}", delta=delta)

k1.metric("💵 Total Cost",      f"${result['grand_total']:,.0f}")
k2.metric("👷 Total Hired",     f"{sum(result['hired']):.1f}")
k3.metric("📉 Total Fired",     f"{sum(result['fired']):.1f}")
k4.metric("📦 Avg Inventory",   f"{sum(result['inventory'])/T:.1f}")
k5.metric("🔧 Total OT Units",  f"{sum(result['overtime']):.1f}")

total_ls = sum(result["lost_sales"])
if total_ls > 0:
    st.metric("🚫 Total Lost Sales", f"{total_ls:,.0f} units")

st.divider()

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab_labels = [
    "📊 Demand",
    "📋 Plan Table",
    "🏭 Production & Inventory",
    "👷 Workforce",
    "💰 Cost Breakdown",
    "⚖️ Compare All",
    "📖 Theory",
]
if strategy == "Transportation Method":
    tab_labels.append("🚚 Transport Tableau")
if strategy == "Linear Programming (LP)":
    tab_labels.append("🔍 LP Shadow Prices")

tabs = st.tabs(tab_labels)

# ─────────────────────────────────────────────
# Tab 0 — Demand Forecast
# ─────────────────────────────────────────────
with tabs[0]:
    st.markdown('<p class="section-title">Demand Forecast by Period</p>', unsafe_allow_html=True)
    cumulative = list(pd.Series(demand_vals).cumsum())
    fig_dem = go.Figure()
    fig_dem.add_bar(x=periods, y=demand_vals, name="Period Demand",
                    marker_color="#1f77b4", opacity=0.85)
    fig_dem.add_scatter(x=periods, y=cumulative, mode="lines+markers",
                        name="Cumulative Demand", line=dict(color="#ff7f0e", width=2),
                        yaxis="y2")
    fig_dem.update_layout(
        yaxis=dict(title="Units"),
        yaxis2=dict(title="Cumulative Units", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.15),
        height=420, margin=dict(t=40, b=60),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_dem, width='stretch')

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Demand", f"{sum(demand_vals):,.0f}")
    col_b.metric("Peak Demand",  f"{max(demand_vals):,.0f} ({periods[demand_vals.index(max(demand_vals))]})")
    col_c.metric("Min Demand",   f"{min(demand_vals):,.0f} ({periods[demand_vals.index(min(demand_vals))]})")

# ─────────────────────────────────────────────
# Tab 1 — Plan Table
# ─────────────────────────────────────────────
with tabs[1]:
    st.markdown('<p class="section-title">Plan Results — Period Detail</p>', unsafe_allow_html=True)

    df = pd.DataFrame({
        "Period":        periods,
        "Demand":        [round(v, 0) for v in result["demand"]],
        "Production":    [round(v, 0) for v in result["production"]],
        "Workforce":     [round(v, 1) for v in result["workforce"]],
        "Hired":         [round(v, 1) for v in result["hired"]],
        "Fired":         [round(v, 1) for v in result["fired"]],
        "Inventory":     [round(v, 1) for v in result["inventory"]],
        "Overtime":      [round(v, 0) for v in result["overtime"]],
        "Subcontract":   [round(v, 0) for v in result["subcontract"]],
        "Lost Sales":    [round(v, 0) for v in result["lost_sales"]],
        "Period Cost($)":[round(v, 0) for v in result["cost_total"]],
    })

    def color_inventory(val):
        if val < 0:
            return "background-color: #f8d7da; color: #721c24;"
        if val == 0:
            return "background-color: #fff3cd;"
        return ""

    def color_cost(val):
        if val > 0:
            q75 = df["Period Cost($)"].quantile(0.75)
            if val >= q75:
                return "background-color: #f8d7da;"
        return ""

    def color_lost_sales(val):
        if val > 0:
            return "background-color: #d1ecf1; color: #0c5460;"
        return ""

    styled = (
        df.style
          .map(color_inventory, subset=["Inventory"])
          .map(color_cost, subset=["Period Cost($)"])
          .map(color_lost_sales, subset=["Lost Sales"])
          .format({"Period Cost($)": "${:,.0f}",
                   "Demand": "{:,.0f}", "Production": "{:,.0f}",
                   "Overtime": "{:,.0f}", "Subcontract": "{:,.0f}",
                   "Lost Sales": "{:,.0f}"})
    )

    st.dataframe(styled, width='stretch', height=460)
    st.caption("🔴 Red inventory = backorder  |  🟡 Yellow = zero inventory  |  🔴 Red cost = above 75th percentile  |  💙 Blue = lost sales")

    # Grand total row
    st.markdown(f"**Grand Total Cost: ${result['grand_total']:,.0f}**")

# ─────────────────────────────────────────────
# Tab 2 — Production & Inventory
# ─────────────────────────────────────────────
with tabs[2]:
    st.markdown('<p class="section-title">Production Mix & Inventory Dynamics</p>', unsafe_allow_html=True)

    # RT production = total - OT - Sub
    rt_prod = [max(0.0, result["production"][t] - result["overtime"][t] - result["subcontract"][t])
               for t in range(T)]

    fig_prod = go.Figure()
    fig_prod.add_bar(x=periods, y=rt_prod,                 name="Regular-Time",  marker_color="#1f77b4")
    fig_prod.add_bar(x=periods, y=result["overtime"],      name="Overtime",      marker_color="#ff7f0e")
    fig_prod.add_bar(x=periods, y=result["subcontract"],   name="Subcontract",   marker_color="#2ca02c")
    fig_prod.add_scatter(x=periods, y=demand_vals,         name="Demand",
                         mode="lines+markers", line=dict(color="#d62728", width=2, dash="dash"))
    fig_prod.add_scatter(x=periods, y=result["inventory"], name="Ending Inventory",
                         mode="lines+markers", line=dict(color="#9467bd", width=2),
                         yaxis="y2")

    fig_prod.update_layout(
        barmode="stack",
        yaxis=dict(title="Units"),
        yaxis2=dict(title="Inventory (units)", overlaying="y", side="right",
                    zeroline=True, zerolinecolor="#999"),
        legend=dict(orientation="h", y=-0.18),
        height=460, margin=dict(t=40, b=80),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_prod, width='stretch')

# ─────────────────────────────────────────────
# Tab 3 — Workforce
# ─────────────────────────────────────────────
with tabs[3]:
    st.markdown('<p class="section-title">Workforce Dynamics</p>', unsafe_allow_html=True)

    fig_wf = go.Figure()
    fig_wf.add_scatter(
        x=periods, y=result["workforce"],
        mode="lines+markers", name="Workforce Level",
        line=dict(color="#1f77b4", width=3, shape="hv"),
        marker=dict(size=8),
    )
    fig_wf.add_bar(x=periods, y=result["hired"],  name="Hired",  marker_color="#2ca02c", opacity=0.7)
    fig_wf.add_bar(x=periods, y=[-v for v in result["fired"]], name="Fired",
                   marker_color="#d62728", opacity=0.7)

    fig_wf.add_hline(y=capacity["max_workforce"], line_dash="dot",
                     line_color="#ff7f0e", annotation_text="Max Workforce")

    fig_wf.update_layout(
        barmode="overlay",
        yaxis=dict(title="Workers"),
        legend=dict(orientation="h", y=-0.18),
        height=420, margin=dict(t=40, b=80),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_wf, width='stretch')

    wf_stats = pd.DataFrame({
        "Period":    periods,
        "Workforce": [round(v, 1) for v in result["workforce"]],
        "Hired":     [round(v, 1) for v in result["hired"]],
        "Fired":     [round(v, 1) for v in result["fired"]],
        "Hire Cost ($)": [round(v, 0) for v in result["cost_hire"]],
        "Fire Cost ($)": [round(v, 0) for v in result["cost_fire"]],
    })
    st.dataframe(wf_stats, width='stretch', hide_index=True)

# ─────────────────────────────────────────────
# Tab 4 — Cost Breakdown
# ─────────────────────────────────────────────
with tabs[4]:
    st.markdown('<p class="section-title">Cost Breakdown by Period & Category</p>', unsafe_allow_html=True)

    cost_cats = {
        "Regular-Time": result["cost_rt"],
        "Hiring":       result["cost_hire"],
        "Firing":       result["cost_fire"],
        "Holding":      result["cost_hold"],
        "Backorder":    result["cost_back"],
        "Overtime":     result["cost_ot"],
        "Subcontract":  result["cost_sub"],
    }
    colors = ["#1f77b4","#2ca02c","#d62728","#9467bd","#ff7f0e","#8c564b","#e377c2"]

    fig_cost = go.Figure()
    for (cat, vals), color in zip(cost_cats.items(), colors):
        if sum(vals) > 0:
            fig_cost.add_bar(x=periods, y=vals, name=cat, marker_color=color)

    fig_cost.update_layout(
        barmode="stack",
        yaxis=dict(title="Cost ($)"),
        legend=dict(orientation="h", y=-0.18),
        height=440, margin=dict(t=40, b=80),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_cost, width='stretch')

    # Summary donut
    totals = {cat: sum(vals) for cat, vals in cost_cats.items() if sum(vals) > 0}
    fig_pie = px.pie(
        values=list(totals.values()), names=list(totals.keys()),
        color_discrete_sequence=colors,
        hole=0.45, title="Total Cost Composition",
    )
    fig_pie.update_layout(height=380, margin=dict(t=60, b=20))
    col_pie, col_table = st.columns([1, 1])
    col_pie.plotly_chart(fig_pie, width='stretch')

    cost_summary = pd.DataFrame({
        "Category": list(totals.keys()),
        "Total ($)": [f"${v:,.0f}" for v in totals.values()],
        "Share (%)": [f"{v/result['grand_total']*100:.1f}%" for v in totals.values()],
    })
    col_table.dataframe(cost_summary, width='stretch', hide_index=True)

# ─────────────────────────────────────────────
# Tab 5 — Compare All Strategies
# ─────────────────────────────────────────────
with tabs[5]:
    st.markdown('<p class="section-title">Strategy Comparison — Same Parameters</p>', unsafe_allow_html=True)

    if st.button("🔄 Run Comparison (all 5 strategies)", type="primary"):
        with st.spinner("Solving all strategies…"):
            df_cmp = compare_all(
                periods=active_periods,
                demand=demand_vals,
                costs=costs,
                capacity=capacity,
                initial_workforce=float(init_wf),
                initial_inventory=float(init_inv),
                productivity=float(productivity),
                integer_workforce=bool(integer_wf),
                integer_production=bool(integer_prod),
                shortage_policy=shortage_policy,
            )
        st.session_state["comparison"] = df_cmp

    if "comparison" in st.session_state:
        df_cmp = st.session_state["comparison"]
        min_cost = df_cmp["Total Cost ($)"].min()

        def highlight_min(s):
            return ["background-color: #d4edda; font-weight:700;"
                    if v == min_cost else "" for v in s]

        st.dataframe(
            df_cmp.style.apply(highlight_min, subset=["Total Cost ($)"])
                        .format({"Total Cost ($)": "${:,.0f}",
                                 "Total Hired": "{:.1f}", "Total Fired": "{:.1f}",
                                 "Avg Inventory": "{:.1f}",
                                 "Total OT Units": "{:.1f}", "Total Sub Units": "{:.1f}"}),
            width='stretch', hide_index=True,
        )

        # Cost bar chart
        fig_cmp = px.bar(
            df_cmp, x="Strategy", y="Total Cost ($)",
            color="Strategy", text_auto=True,
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Total Cost by Strategy",
        )
        fig_cmp.update_layout(showlegend=False, height=380,
                               plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_cmp, width='stretch')

        # Radar / spider chart
        cats = ["Total Cost (norm)", "Workforce Stability", "Avg Inventory (norm)",
                "OT Usage (norm)", "Sub Usage (norm)"]
        fig_radar = go.Figure()
        max_cost = df_cmp["Total Cost ($)"].max() or 1
        max_inv  = df_cmp["Avg Inventory"].abs().max() or 1
        max_ot   = df_cmp["Total OT Units"].max() or 1
        max_sub  = df_cmp["Total Sub Units"].max() or 1
        max_chg  = (df_cmp["Total Hired"] + df_cmp["Total Fired"]).max() or 1

        for _, row in df_cmp.iterrows():
            values = [
                row["Total Cost ($)"]   / max_cost,
                (row["Total Hired"] + row["Total Fired"]) / max_chg,
                abs(row["Avg Inventory"]) / max_inv,
                row["Total OT Units"]   / max_ot  if max_ot  > 0 else 0,
                row["Total Sub Units"]  / max_sub if max_sub > 0 else 0,
            ]
            values += values[:1]
            fig_radar.add_trace(go.Scatterpolar(
                r=values, theta=cats + cats[:1],
                fill="toself", name=row["Strategy"], opacity=0.6,
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=450,
            title="Normalised Strategy Profile (lower = better)",
        )
        st.plotly_chart(fig_radar, width='stretch')
    else:
        st.info("Click the button above to compare all strategies with the current parameters.")

# ─────────────────────────────────────────────
# Tab 6 — Transportation Tableau (conditional)
# ─────────────────────────────────────────────
if strategy == "Transportation Method":
    with tabs[7]:
        st.markdown('<p class="section-title">Transportation Tableau</p>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Each cell shows units allocated (top) and unit cost in $ (bottom). Color intensity indicates allocation volume.</div>',
                    unsafe_allow_html=True)
        st.write("")

        if result.get("transport_tableau"):
            tb = result["transport_tableau"]
            sources      = tb["sources"]
            allocations  = np.array(tb["allocations"])
            cell_costs   = np.array(tb["cell_costs"])
            caps         = tb["capacities"]

            # Build display dataframe
            tdf = pd.DataFrame(allocations, index=sources, columns=periods)
            tdf["Capacity"] = caps

            def style_alloc(val):
                try:
                    v = float(val)
                except (ValueError, TypeError):
                    return ""
                if v > 0:
                    intensity = min(int(v / (allocations.max() or 1) * 200), 200)
                    return f"background-color: rgba(31,119,180,{intensity/255:.2f}); color: {'white' if intensity > 120 else 'black'};"
                return "color: #aaa;"

            st.dataframe(
                tdf.style.applymap(style_alloc, subset=periods)
                         .format("{:.0f}"),
                width='stretch',
                height=min(60 + len(sources) * 38, 600),
            )
            st.caption("Blue shading = higher allocation.  Last column = source capacity.")
        else:
            st.warning("Tableau data not available.")

# ─────────────────────────────────────────────
# Tab 6/7 — LP Shadow Prices (conditional)
# ─────────────────────────────────────────────
if strategy == "Linear Programming (LP)":
    shadow_tab_idx = 7
    with tabs[shadow_tab_idx]:
        st.markdown('<p class="section-title">LP Sensitivity Analysis — Shadow Prices</p>', unsafe_allow_html=True)

        st.markdown("""
<div class="shadow-box">
<strong>What is a Shadow Price?</strong><br>
A shadow price (dual value) represents the marginal cost of tightening a constraint by one unit.
For example, if the inventory balance constraint for June has a shadow price of <strong>$50</strong>,
it means that if demand in June increases by 1 unit, the optimal total cost increases by $50 —
this is the <em>true marginal cost of one unit of demand in that period</em>.
</div>
""", unsafe_allow_html=True)
        st.write("")

        sp = result.get("shadow_prices")
        if sp:
            sp_df = pd.DataFrame({
                "Period": periods,
                "Inventory Balance SP ($)": [round(v, 2) for v in sp.get("Inventory Balance", [0]*T)],
                "Workforce Balance SP ($)": [round(v, 2) for v in sp.get("Workforce Balance", [0]*T)],
            })
            st.dataframe(sp_df, width='stretch', hide_index=True)

            fig_sp = go.Figure()
            fig_sp.add_bar(x=periods, y=sp["Inventory Balance"], name="Inventory Balance",
                           marker_color="#1f77b4")
            fig_sp.add_bar(x=periods, y=sp["Workforce Balance"], name="Workforce Balance",
                           marker_color="#ff7f0e")
            fig_sp.update_layout(
                barmode="group",
                yaxis=dict(title="Shadow Price ($)"),
                legend=dict(orientation="h", y=-0.18),
                height=380, margin=dict(t=40, b=80),
                plot_bgcolor="white", paper_bgcolor="white",
                title="Shadow Prices by Period",
            )
            st.plotly_chart(fig_sp, width='stretch')
        else:
            st.info("Shadow price data not available from this solver run. "
                    "Ensure the LP solved to optimality.")

# ─────────────────────────────────────────────
# Theory tab (always index 6)
# ─────────────────────────────────────────────
with tabs[6]:
    st.markdown('<p class="section-title">Theory & Methodology Reference</p>', unsafe_allow_html=True)

    _theory_path = Path(__file__).parent / "theory.html"
    if _theory_path.exists():
        with open(_theory_path, "r", encoding="utf-8") as _f:
            theory_html = _f.read()
        # Render inside a scrollable iframe-like container
        components.html(theory_html, height=820, scrolling=True)
        st.divider()
        with open(_theory_path, "rb") as _fb:
            st.download_button(
                label="⬇️ Download as standalone HTML file",
                data=_fb.read(),
                file_name="aggregate_planning_theory.html",
                mime="text/html",
            )
        st.caption("The downloaded file works offline in any browser and can be printed or shared with students.")
    else:
        st.warning("theory.html not found. Please ensure it is in the same folder as app.py.")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "Aggregate Planning Dashboard · Production Planning & Control · "
    "Built with Streamlit + Plotly + SciPy"
)
st.caption("👨‍🏫 **Prof. Helder Costa** · heldergc@id.uff.br")
