"""
Comprehensive Integration Tests for Models, Optimizer, Simulator, Bandits, Decision Intelligence, and MLOps.
"""

import pytest
import pandas as pd
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


@pytest.fixture
def trained_pipeline(tmp_path):
    test_db = tmp_path / "test_pipeline.db"
    db_mgr = WarehouseDBManager(db_path=test_db)
    db_mgr.initialize_warehouse(days=60, seed=42, force_recreate=True)
    df_raw = db_mgr.get_consolidated_warehouse_data()
    
    fs = FeatureStore()
    df_features = fs.build_feature_set(df_raw)
    
    resp_engine = CampaignResponseEngine()
    metrics = resp_engine.fit(df_features)
    
    risk_engine = RiskEngine()
    optimizer = BudgetOptimizer(resp_engine, risk_engine)
    simulator = DecisionSimulator(optimizer)
    intelligence = DecisionIntelligence(resp_engine, optimizer)
    history_mgr = DecisionHistoryManager(db_path=test_db)
    exp_tracker = ExperimentTracker(db_path=test_db)
    registry = ModelRegistry(db_path=test_db)
    drift_mon = DriftMonitor()

    return {
        "df_raw": df_raw,
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


def test_campaign_response_engine(trained_pipeline):
    engine = trained_pipeline['resp_engine']
    assert engine.is_trained
    
    # Test single channel response
    res = engine.predict_channel_response("Google Search", 10000.0)
    assert res['expected_revenue'] > 0.0
    assert res['expected_conversions'] > 0.0

    # Test multi-channel predict_response
    alloc = {"Google Search": 20000, "Facebook Ads": 15000, "Instagram Ads": 10000}
    multi_res = engine.predict_response(alloc)
    assert multi_res['expected_revenue'] > 0.0
    assert len(multi_res['channel_breakdown']) == 3


def test_budget_optimizer(trained_pipeline):
    optimizer = trained_pipeline['optimizer']
    opt_res = optimizer.optimize_budget(total_budget=100000.0)
    
    assert opt_res['total_budget'] == 100000.0
    assert opt_res['expected_revenue'] > 0.0
    assert sum(opt_res['optimal_allocation'].values()) <= 100005.0


def test_decision_simulator(trained_pipeline):
    simulator = trained_pipeline['simulator']
    sim_res = simulator.run_scenario_simulation(base_budget=100000.0, cpc_multiplier=1.4)
    
    assert "deltas" in sim_res
    assert "revenue_delta" in sim_res['deltas']


def test_linucb_bandits():
    ban_res = run_bandit_simulation(days=15)
    assert ban_res['simulation_days'] == 15
    assert len(ban_res['cumulative_reward_curve']) == 15


def test_decision_intelligence(trained_pipeline):
    intel = trained_pipeline['intelligence']
    payload = intel.generate_executive_payload(total_budget=100000.0)
    
    assert "executive_recommendation" in payload
    assert "next_best_actions" in payload
    assert len(payload['next_best_actions']) > 0


def test_decision_history_and_mlops(trained_pipeline):
    hist = trained_pipeline['history_mgr']
    hist.log_run("RUN_TEST_1", 100000.0, {}, {"Google": 50000}, 300000.0, 1500.0, 20.0)
    df_hist = hist.get_run_history()
    assert len(df_hist) >= 1

    tracker = trained_pipeline['exp_tracker']
    tracker.log_experiment("EXP_TEST", "XGBoost", {}, 0.85, 120.0, 0.80)
    df_exp = tracker.get_experiments()
    assert len(df_exp) >= 1

    reg = trained_pipeline['registry']
    reg.register_model("v1.0.0", "Ensemble", 0.85, 120.0, stage="Production")
    df_reg = reg.get_registered_models()
    assert len(df_reg) >= 1
