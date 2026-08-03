"""
Decision Simulation Engine ("What-If" Stress Testing).
Simulates market shocks and evaluates Decision -> Prediction -> Optimization -> Business Outcome chains.
"""

from typing import Dict, Any, List
from marketing_ai.optimization.budget_optimizer import BudgetOptimizer


class DecisionSimulator:
    """
    Simulates business outcomes under market shocks and scenario changes.
    """
    def __init__(self, optimizer: BudgetOptimizer):
        self.optimizer = optimizer

    def run_scenario_simulation(
        self,
        base_budget: float = 100000.0,
        cpc_multiplier: float = 1.0,
        holiday_override: int = 0,
        channel_efficiency_override: Dict[str, float] = None,
        weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Runs scenario simulation and calculates deltas against baseline.
        """
        # Baseline run
        base_context = {"competitor_cpc": 2.10, "holiday": 0, "month": 6}
        base_result = self.optimizer.optimize_budget(total_budget=base_budget, weights=weights, context=base_context)

        # Scenario run context
        scenario_cpc = 2.10 * cpc_multiplier
        scenario_context = {
            "competitor_cpc": scenario_cpc,
            "holiday": holiday_override,
            "month": 11 if holiday_override == 1 else 6
        }

        scenario_result = self.optimizer.optimize_budget(total_budget=base_budget, weights=weights, context=scenario_context)

        # Calculate Deltas
        rev_delta = scenario_result['expected_revenue'] - base_result['expected_revenue']
        rev_delta_pct = (rev_delta / (base_result['expected_revenue'] + 1e-6)) * 100
        
        cpa_delta = scenario_result['expected_cpa'] - base_result['expected_cpa']

        return {
            "scenario_parameters": {
                "base_budget": base_budget,
                "cpc_multiplier": cpc_multiplier,
                "holiday_override": holiday_override,
                "scenario_cpc": round(scenario_cpc, 2)
            },
            "baseline_outcome": {
                "revenue": base_result['expected_revenue'],
                "conversions": base_result['expected_conversions'],
                "cpa": base_result['expected_cpa'],
                "allocation": base_result['optimal_allocation']
            },
            "scenario_outcome": {
                "revenue": scenario_result['expected_revenue'],
                "conversions": scenario_result['expected_conversions'],
                "cpa": scenario_result['expected_cpa'],
                "allocation": scenario_result['optimal_allocation']
            },
            "deltas": {
                "revenue_delta": round(rev_delta, 2),
                "revenue_delta_pct": round(rev_delta_pct, 2),
                "cpa_delta": round(cpa_delta, 2)
            }
        }
