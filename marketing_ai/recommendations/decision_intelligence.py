"""
Decision Intelligence Module.
Translates ML predictions, MMM curves, optimization vectors, and risk bounds into executive business advice payloads.
"""

from typing import Dict, Any, List
from marketing_ai.models.campaign_response import CampaignResponseEngine
from marketing_ai.models.explainability import ModelExplainer
from marketing_ai.optimization.budget_optimizer import BudgetOptimizer
from marketing_ai.risk.risk_engine import RiskEngine


class DecisionIntelligence:
    """
    Synthesizes multi-module technical outputs into structured executive payloads.
    """
    def __init__(self, response_engine: CampaignResponseEngine, optimizer: BudgetOptimizer):
        self.response_engine = response_engine
        self.optimizer = optimizer
        self.explainer = ModelExplainer(response_engine)
        self.risk_engine = RiskEngine()

    def generate_executive_payload(
        self,
        total_budget: float = 100000.0,
        weights: Dict[str, float] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generates full Decision Intelligence payload for marketing executives.
        """
        context = context or {}
        
        # 1. Run Optimization
        opt_res = self.optimizer.optimize_budget(total_budget=total_budget, weights=weights, context=context)
        optimal_alloc = opt_res['optimal_allocation']
        equal_alloc = opt_res['equal_allocation_baseline']

        # 2. Risk Metrics
        risk_res = opt_res['risk_profile']

        # 3. Model Drivers
        drivers = self.explainer.get_driver_explanation(optimal_alloc, context)

        # 4. Synthesize Executive Recommendation
        top_channel = max(optimal_alloc.items(), key=lambda x: x[1])
        lowest_channel = min(optimal_alloc.items(), key=lambda x: x[1])
        
        rec_text = (
            f"Allocate ${optimal_alloc[top_channel[0]]:,.2f} ({top_channel[0]}) as your primary growth driver, "
            f"while constraining {lowest_channel[0]} to ${optimal_alloc[lowest_channel[0]]:,.2f}. "
            f"This optimization is projected to deliver ${opt_res['expected_revenue']:,.2f} in total revenue "
            f"(+{opt_res['revenue_lift_pct']}% lift over equal allocation) at a CPA of ${opt_res['expected_cpa']:.2f}."
        )

        # 5. Next Best Action
        next_actions = [
            f"Increase Email Marketing budget share to maximize high-ROI conversion velocity.",
            f"Maintain Google Search baseline (>20% share) to capture high-intent search traffic.",
            f"Monitor competitor CPC trends; set automated alert if CPC rises above +25%.",
            f"Conduct 5% allocation exploration via Contextual Bandits on new audience segments."
        ]

        return {
            "prediction_summary": {
                "expected_revenue": opt_res['expected_revenue'],
                "expected_conversions": opt_res['expected_conversions'],
                "expected_cpa": opt_res['expected_cpa'],
                "expected_roi": opt_res['expected_roi']
            },
            "optimization_summary": {
                "total_budget": total_budget,
                "optimal_allocation": optimal_alloc,
                "baseline_equal_allocation": equal_alloc,
                "revenue_lift_pct": opt_res['revenue_lift_pct']
            },
            "business_explanation": {
                "top_drivers": drivers
            },
            "executive_recommendation": rec_text,
            "confidence_and_risk": {
                "risk_tier": risk_res['risk_tier'],
                "confidence_interval_95": risk_res['confidence_interval_95'],
                "marketing_sharpe_ratio": risk_res['marketing_sharpe_ratio'],
                "diversification_score": risk_res['diversification_score']
            },
            "next_best_actions": next_actions
        }
