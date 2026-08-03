"""
Multi-Objective Constrained Budget Optimizer.
Uses Operations Research mathematical optimization (Scipy SLSQP) to solve optimal channel allocations.
Formulation: Maximize (w1 * Revenue + w2 * Conversions - w3 * CPA - w4 * Risk) under budget & channel bounds.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.optimize import minimize
from marketing_ai.models.campaign_response import CampaignResponseEngine
from marketing_ai.risk.risk_engine import RiskEngine
from marketing_ai.utils.config import CHANNELS, DEFAULT_OPTIMIZER_WEIGHTS


class BudgetOptimizer:
    """
    Constrained Operations Research solver for marketing budget allocation.
    """
    def __init__(self, response_engine: CampaignResponseEngine, risk_engine: Optional[RiskEngine] = None):
        self.response_engine = response_engine
        self.risk_engine = risk_engine or RiskEngine()

    def optimize_budget(
        self,
        total_budget: float = 100000.0,
        weights: Optional[Dict[str, float]] = None,
        max_cpa: Optional[float] = None,
        channel_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Solves multi-objective constrained optimization problem.
        
        Args:
            total_budget: Total spending limit ($).
            weights: Weighting dict {'revenue': w1, 'conversions': w2, 'cpa_penalty': w3, 'risk_penalty': w4}.
            max_cpa: Maximum permissible Cost per Acquisition ($).
            channel_bounds: Dict mapping channel name to (min_share, max_share) bounds e.g. (0.10, 0.40).
            context: Market context parameters.
        """
        if not self.response_engine.is_trained:
            raise RuntimeError("Response engine must be trained before optimization.")

        weights = weights or DEFAULT_OPTIMIZER_WEIGHTS
        context = context or {}

        channel_names = [c['channel_name'] for c in CHANNELS]
        n_channels = len(channel_names)

        # Default bounds: min 5%, max 45% per channel unless specified
        default_bounds_dict = {
            "Google Search": (0.20, 0.50),   # min 20%
            "Facebook Ads": (0.10, 0.40),
            "Instagram Ads": (0.10, 0.40),
            "LinkedIn Ads": (0.05, 0.30),
            "Email Marketing": (0.05, 0.40), # max 40%
            "TV & Brand Video": (0.00, 0.30)
        }
        
        bounds_map = channel_bounds or default_bounds_dict
        
        # Scipy bounds format: (min_spend, max_spend)
        scipy_bounds = []
        for name in channel_names:
            min_pct, max_pct = bounds_map.get(name, (0.05, 0.45))
            scipy_bounds.append((total_budget * min_pct, total_budget * max_pct))

        # Equal allocation initial guess
        initial_guess = np.array([total_budget / n_channels] * n_channels)

        # Objective Function to MINIMIZE (negative of utility)
        def objective(x):
            alloc_dict = {name: float(val) for name, val in zip(channel_names, x)}
            res = self.response_engine.predict_response(alloc_dict, context)
            
            rev = res['expected_revenue']
            conv = res['expected_conversions']
            cpa = res['expected_cpa']
            std_dev = res['revenue_std_dev']
            
            # Multi-objective utility score
            # Normalization factors for balanced gradient search
            utility = (
                weights.get('revenue', 1.0) * (rev / 1000.0) +
                weights.get('conversions', 0.5) * (conv * 2.0) -
                weights.get('cpa_penalty', 0.3) * (cpa * 5.0) -
                weights.get('risk_penalty', 0.2) * (std_dev / 1000.0)
            )
            return -utility

        # Constraints
        # 1. Total budget sum constraint: sum(x) <= total_budget
        constraints = [
            {'type': 'ineq', 'fun': lambda x: total_budget - np.sum(x)}
        ]

        # 2. Max CPA constraint if specified: max_cpa - cpa >= 0
        if max_cpa is not None:
            def cpa_constraint(x):
                alloc_dict = {name: float(val) for name, val in zip(channel_names, x)}
                res = self.response_engine.predict_response(alloc_dict, context)
                return max_cpa - res['expected_cpa']
            constraints.append({'type': 'ineq', 'fun': cpa_constraint})

        # Run SLSQP solver
        opt_res = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=scipy_bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-6}
        )

        optimized_spends = np.clip(opt_res.x, [b[0] for b in scipy_bounds], [b[1] for b in scipy_bounds])
        
        # Scale to ensure exact total budget if sum exceeds
        if np.sum(optimized_spends) > total_budget:
            optimized_spends = (optimized_spends / np.sum(optimized_spends)) * total_budget

        optimal_alloc_dict = {name: round(float(val), 2) for name, val in zip(channel_names, optimized_spends)}
        
        # Query final performance
        eval_metrics = self.response_engine.predict_response(optimal_alloc_dict, context)
        risk_metrics = self.risk_engine.evaluate_risk_profile(
            eval_metrics['expected_revenue'],
            eval_metrics['revenue_std_dev'],
            optimal_alloc_dict
        )

        # Baseline equal allocation comparison
        equal_alloc_dict = {name: round(total_budget / n_channels, 2) for name in channel_names}
        baseline_metrics = self.response_engine.predict_response(equal_alloc_dict, context)

        rev_lift_pct = round(((eval_metrics['expected_revenue'] - baseline_metrics['expected_revenue']) / (baseline_metrics['expected_revenue'] + 1e-6)) * 100, 2)

        return {
            "total_budget": float(total_budget),
            "optimal_allocation": optimal_alloc_dict,
            "equal_allocation_baseline": equal_alloc_dict,
            "expected_revenue": eval_metrics['expected_revenue'],
            "expected_conversions": eval_metrics['expected_conversions'],
            "expected_cpa": eval_metrics['expected_cpa'],
            "expected_roi": eval_metrics['expected_roi'],
            "revenue_lift_pct": rev_lift_pct,
            "channel_breakdown": eval_metrics['channel_breakdown'],
            "risk_profile": risk_metrics,
            "optimization_success": bool(opt_res.success)
        }
