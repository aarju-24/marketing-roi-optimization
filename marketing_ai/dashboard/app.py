"""
Streamlit Executive Dashboard: Marketing Decision Intelligence Platform.
Business-focused interactive UI bringing together all 9 system layers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from datetime import datetime

# Ensure marketing_ai is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from marketing_ai.warehouse.db_manager import WarehouseDBManager
from marketing_ai.features.feature_store import FeatureStore
from marketing_ai.models.campaign_response import CampaignResponseEngine
from marketing_ai.models.explainability import ModelExplainer
from marketing_ai.risk.risk_engine import RiskEngine
from marketing_ai.optimization.budget_optimizer import BudgetOptimizer
from marketing_ai.simulation.decision_simulator import DecisionSimulator
from marketing_ai.bandits.linucb import run_bandit_simulation
from marketing_ai.recommendations.decision_intelligence import DecisionIntelligence
from marketing_ai.history.decision_history import DecisionHistoryManager
from marketing_ai.mlops.experiment_tracker import ExperimentTracker
from marketing_ai.mlops.model_registry import ModelRegistry
from marketing_ai.mlops.drift_monitor import DriftMonitor

# Page Setup
st.set_page_config(
    page_title="Marketing Decision Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.0rem; color: #64748B; margin-bottom: 1.5rem; }
    .kpi-card { background: #F8FAFC; border-radius: 10px; padding: 1.2rem; border-left: 5px solid #3B82F6; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .kpi-title { font-size: 0.85rem; color: #64748B; text-transform: uppercase; font-weight: 600; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #0F172A; }
    .kpi-delta { font-size: 0.9rem; font-weight: 600; color: #10B981; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system_pipeline():
    """Initializes database, feature store, and trains response models."""
    db_mgr = WarehouseDBManager()
    db_mgr.initialize_warehouse(days=730, seed=42)
    df_raw = db_mgr.get_consolidated_warehouse_data()
    
    fs = FeatureStore()
    df_features = fs.build_feature_set(df_raw)
    
    resp_engine = CampaignResponseEngine()
    metrics = resp_engine.fit(df_features)
    
    risk_engine = RiskEngine()
    optimizer = BudgetOptimizer(resp_engine, risk_engine)
    simulator = DecisionSimulator(optimizer)
    intelligence = DecisionIntelligence(resp_engine, optimizer)
    history_mgr = DecisionHistoryManager()
    exp_tracker = ExperimentTracker()
    registry = ModelRegistry()
    drift_mon = DriftMonitor()

    # Log initial experiment & model registry
    exp_id = f"EXP_INIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_tracker.log_experiment(exp_id, "GradientBoosting+RandomForest", {}, metrics['r2_revenue'], metrics['mae_revenue'], metrics['r2_conversions'])
    registry.register_model("v1.0.0", "GBDT Ensemble", metrics['r2_revenue'], metrics['mae_revenue'], stage="Production")

    return {
        "db_mgr": db_mgr,
        "df_raw": df_raw,
        "df_features": df_features,
        "resp_engine": resp_engine,
        "risk_engine": risk_engine,
        "optimizer": optimizer,
        "simulator": simulator,
        "intelligence": intelligence,
        "history_mgr": history_mgr,
        "exp_tracker": exp_tracker,
        "registry": registry,
        "drift_mon": drift_mon,
        "metrics": metrics
    }


pipeline = load_system_pipeline()

# Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.caption("Marketing AI Decision Platform")

nav_choice = st.sidebar.radio(
    "Select Workflow View:",
    [
        "📊 Executive Overview",
        "📈 Campaign Analytics",
        "🎯 Response & Predictions",
        "⚡ Multi-Objective Budget Optimizer",
        "🧪 Decision Simulator (What-If)",
        "🤖 Adaptive Online Learning (Bandits)",
        "💡 Executive Recommendations",
        "📜 Decision History Audit",
        "🛠️ System Health & MLOps"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Model Version: **v1.0.0 (Production)**")
st.sidebar.caption(f"Feature Store: **v1.0.0**")

# ==================== VIEW 1: EXECUTIVE OVERVIEW ====================
if nav_choice == "📊 Executive Overview":
    st.markdown("<div class='main-header'>Executive ROI & Decision Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>High-level performance metrics, total revenue, and historical campaign efficiency</div>", unsafe_allow_html=True)

    df_raw = pipeline['df_raw']
    total_rev = df_raw['revenue'].sum()
    total_spend = df_raw['spend'].sum()
    total_conv = df_raw['conversions'].sum()
    overall_roi = (total_rev - total_spend) / total_spend
    avg_cpa = total_spend / total_conv

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Revenue</div><div class='kpi-value'>${total_rev:,.0f}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Spend</div><div class='kpi-value'>${total_spend:,.0f}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Overall ROI</div><div class='kpi-value'>{overall_roi*100:.1f}%</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Average CPA</div><div class='kpi-value'>${avg_cpa:.2f}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Revenue & Spend by Channel
    df_ch = df_raw.groupby('channel_name')[['spend', 'revenue', 'conversions']].sum().reset_index()
    df_ch['roi'] = (df_ch['revenue'] - df_ch['spend']) / df_ch['spend']
    
    fig = px.bar(df_ch, x='channel_name', y=['spend', 'revenue'], barmode='group', title="Spend vs. Revenue by Marketing Channel", labels={"value": "USD ($)", "channel_name": "Channel"})
    st.plotly_chart(fig, use_container_width=True)

# ==================== VIEW 2: CAMPAIGN ANALYTICS ====================
elif nav_choice == "📈 Campaign Analytics":
    st.markdown("<div class='main-header'>Campaign Performance & Channel Deep Dive</div>", unsafe_allow_html=True)
    
    df_raw = pipeline['df_raw']
    selected_ch = st.selectbox("Filter by Channel:", ["All"] + list(df_raw['channel_name'].unique()))
    
    if selected_ch != "All":
        df_filtered = df_raw[df_raw['channel_name'] == selected_ch]
    else:
        df_filtered = df_raw

    col1, col2 = st.columns(2)
    with col1:
        fig_cpa = px.histogram(df_filtered, x='cpa' if 'cpa' in df_filtered else df_filtered['spend'] / df_filtered['conversions'], nbins=30, title="CPA Distribution Across Campaigns", color_discrete_sequence=['#3B82F6'])
        st.plotly_chart(fig_cpa, use_container_width=True)
    with col2:
        fig_trend = px.line(df_filtered, x='date', y='revenue', color='channel_name', title="Daily Revenue Velocity")
        st.plotly_chart(fig_trend, use_container_width=True)

# ==================== VIEW 3: RESPONSE & PREDICTIONS ====================
elif nav_choice == "🎯 Response & Predictions":
    st.markdown("<div class='main-header'>Campaign Response & Forecast Engine</div>", unsafe_allow_html=True)
    st.markdown("Predict expected revenue and conversions for any given spend level across channels.")

    ch_select = st.selectbox("Select Target Channel:", [c['channel_name'] for c in pipeline['resp_engine'].feature_store.FEATURE_CATALOG['raw'] if False] or ["Google Search", "Facebook Ads", "Instagram Ads", "LinkedIn Ads", "Email Marketing", "TV & Brand Video"])
    test_spend = st.slider("Select Target Spend ($):", min_value=1000, max_value=50000, value=25000, step=1000)

    pred = pipeline['resp_engine'].predict_channel_response(ch_select, float(test_spend))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Expected Revenue", f"${pred['expected_revenue']:,.2f}")
    with col2:
        st.metric("Expected Conversions", f"{int(pred['expected_conversions']):,}")
    with col3:
        st.metric("Predicted CPA", f"${pred['expected_cpa']:.2f}")
    with col4:
        st.metric("Predicted ROI", f"{pred['expected_roi']*100:.1f}%")

    st.markdown("---")
    st.caption("95% Confidence Interval Forecast:")
    st.info(f"Predicted Revenue: **${pred['expected_revenue']:,.2f}** ± ${1.96 * pred['rev_std_err']:,.2f} (Confidence Range: **${max(0, pred['expected_revenue'] - 1.96*pred['rev_std_err']):,.2f}** to **${pred['expected_revenue'] + 1.96*pred['rev_std_err']:,.2f}**)")

# ==================== VIEW 4: BUDGET OPTIMIZER ====================
elif nav_choice == "⚡ Multi-Objective Budget Optimizer":
    st.markdown("<div class='main-header'>Multi-Objective Constrained Budget Optimizer</div>", unsafe_allow_html=True)
    st.markdown("Solve mathematical optimal allocations under real-world budget, CPA, risk, and channel constraints.")

    c1, c2, c3 = st.columns(3)
    with c1:
        budget_input = st.number_input("Total Budget ($):", min_value=10000, max_value=1000000, value=100000, step=10000)
    with c2:
        risk_weight = st.slider("Risk Penalty Weight:", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    with c3:
        max_cpa_input = st.number_input("Max Permissible CPA ($):", min_value=5.0, max_value=100.0, value=40.0)

    if st.button("Run Multi-Objective Optimization 🚀"):
        weights = {"revenue": 1.0, "conversions": 0.5, "cpa_penalty": 0.3, "risk_penalty": risk_weight}
        opt_res = pipeline['optimizer'].optimize_budget(total_budget=budget_input, weights=weights, max_cpa=max_cpa_input)
        
        # Log run in history
        import uuid
        run_id = f"RUN_{str(uuid.uuid4())[:8]}"
        pipeline['history_mgr'].log_run(run_id, budget_input, weights, opt_res['optimal_allocation'], opt_res['expected_revenue'], opt_res['expected_conversions'], opt_res['expected_cpa'])

        st.success(f"Optimization Complete! Projected Revenue Lift: **+{opt_res['revenue_lift_pct']}%** over equal allocation baseline.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Optimal Revenue", f"${opt_res['expected_revenue']:,.2f}")
        with col2:
            st.metric("Optimal Conversions", f"{int(opt_res['expected_conversions']):,}")
        with col3:
            st.metric("Expected CPA", f"${opt_res['expected_cpa']:.2f}")
        with col4:
            st.metric("Risk Profile", opt_res['risk_profile']['risk_tier'])

        df_alloc = pd.DataFrame([
            {"Channel": k, "Optimal Allocation ($)": v, "Baseline Allocation ($)": opt_res['equal_allocation_baseline'][k]}
            for k, v in opt_res['optimal_allocation'].items()
        ])
        fig_alloc = px.bar(df_alloc, x="Channel", y=["Optimal Allocation ($)", "Baseline Allocation ($)"], barmode="group", title="Optimal Budget Allocation vs. Baseline Equal Split")
        st.plotly_chart(fig_alloc, use_container_width=True)

# ==================== VIEW 5: DECISION SIMULATOR ====================
elif nav_choice == "🧪 Decision Simulator (What-If)":
    st.markdown("<div class='main-header'>Decision Simulator (What-If Stress Testing)</div>", unsafe_allow_html=True)
    st.markdown("Simulate market shocks and stress-test decisions before committing marketing dollars.")

    col1, col2 = st.columns(2)
    with col1:
        cpc_mult = st.slider("Competitor CPC Inflation Multiplier:", min_value=0.5, max_value=2.5, value=1.3, step=0.1)
    with col2:
        is_holiday_flag = st.selectbox("Promotional Holiday Season:", [0, 1], format_func=lambda x: "Yes (Q4 Peak)" if x == 1 else "No (Standard)")

    sim_res = pipeline['simulator'].run_scenario_simulation(base_budget=100000, cpc_multiplier=cpc_mult, holiday_override=is_holiday_flag)

    st.markdown("### Simulation Comparison Deltas")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Baseline Revenue", f"${sim_res['baseline_outcome']['revenue']:,.2f}")
    with c2:
        st.metric("Simulated Revenue", f"${sim_res['scenario_outcome']['revenue']:,.2f}", delta=f"{sim_res['deltas']['revenue_delta_pct']}%")
    with c3:
        st.metric("Simulated CPA", f"${sim_res['scenario_outcome']['cpa']:.2f}", delta=f"${sim_res['deltas']['cpa_delta']:.2f}")

# ==================== VIEW 6: ADAPTIVE ONLINE LEARNING ====================
elif nav_choice == "🤖 Adaptive Online Learning (Bandits)":
    st.markdown("<div class='main-header'>Adaptive Sequential Online Learning (Multi-Armed Bandits)</div>", unsafe_allow_html=True)
    st.markdown("Simulates daily dynamic channel allocation updating payoff beliefs (Exploration vs Exploitation).")

    days_input = st.slider("Simulation Horizon (Days):", min_value=14, max_value=120, value=60)
    if st.button("Run LinUCB Bandit Simulation 🔄"):
        ban_res = run_bandit_simulation(days=days_input)
        st.markdown(f"**Total Cumulative Reward:** ${ban_res['total_cumulative_reward']:,.2f} over {days_input} days")

        df_band = pd.DataFrame({
            "Day": range(1, days_input + 1),
            "Cumulative Reward": ban_res['cumulative_reward_curve'],
            "Daily Reward": ban_res['rewards_history']
        })
        fig_band = px.line(df_band, x="Day", y="Cumulative Reward", title="LinUCB Sequential Cumulative Reward Curve")
        st.plotly_chart(fig_band, use_container_width=True)

# ==================== VIEW 7: EXECUTIVE RECOMMENDATIONS ====================
elif nav_choice == "💡 Executive Recommendations":
    st.markdown("<div class='main-header'>Executive Decision Payload</div>", unsafe_allow_html=True)
    st.markdown("Synthesized natural-language decision support for marketing leadership.")

    payload = pipeline['intelligence'].generate_executive_payload(total_budget=100000)

    st.info(f"### 🎯 Strategic Recommendation\n{payload['executive_recommendation']}")

    st.markdown("### 📋 Next Best Actions")
    for act in payload['next_best_actions']:
        st.write(f"- {act}")

    st.markdown("### 🔍 Model Feature Drivers")
    df_drivers = pd.DataFrame(payload['business_explanation']['top_drivers'])
    st.table(df_drivers[['feature', 'importance_score', 'classification', 'business_meaning']])

# ==================== VIEW 8: DECISION HISTORY AUDIT ====================
elif nav_choice == "📜 Decision History Audit":
    st.markdown("<div class='main-header'>Decision History Audit Trail</div>", unsafe_allow_html=True)
    st.markdown("Audit log of past optimization runs and model recommendation history.")

    df_hist = pipeline['history_mgr'].get_run_history()
    if len(df_hist) > 0:
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.warning("No optimization runs logged yet. Run the Multi-Objective Optimizer to create audit records.")

# ==================== VIEW 9: SYSTEM HEALTH & MLOPS ====================
elif nav_choice == "🛠️ System Health & MLOps":
    st.markdown("<div class='main-header'>System Health & MLOps Control Center</div>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Model Registry", "Experiment Tracker", "Drift Monitoring"])
    
    with t1:
        st.subheader("Registered Models")
        st.dataframe(pipeline['registry'].get_registered_models(), use_container_width=True)
        
    with t2:
        st.subheader("Experiment Tracker Log")
        st.dataframe(pipeline['exp_tracker'].get_experiments(), use_container_width=True)
        
    with t3:
        st.subheader("Distribution & Feature Drift Monitor")
        drift_res = pipeline['drift_mon'].check_feature_drift(pipeline['df_raw'].iloc[:300], pipeline['df_raw'].iloc[300:])
        st.write(f"**Status:** {drift_res['overall_status']} - {drift_res['message']}")
        st.json(drift_res['feature_drift_details'])

# Technical Model Insights Drawer (Expandable)
with st.sidebar.expander("🔬 Model Technical Insights"):
    st.caption("Training Metrics:")
    st.write(f"Revenue R² Score: **{pipeline['metrics']['r2_revenue']:.4f}**")
    st.write(f"Revenue MAE: **${pipeline['metrics']['mae_revenue']:,.2f}**")
    st.write(f"Conversions R²: **{pipeline['metrics']['r2_conversions']:.4f}**")
