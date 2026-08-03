"""
Risk & Uncertainty Engine.
Quantifies forecast variance, confidence intervals, scenario volatility, and risk-adjusted ROI ratios.
"""

import numpy as np
from typing import Dict, Any, Tuple


class RiskEngine:
    """
    Evaluates risk profile, variance bounds, and volatility metrics for proposed allocation plans.
    """
    def __init__(self):
        pass

    def evaluate_risk_profile(self, expected_revenue: float, revenue_std_dev: float, channel_spend_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Computes comprehensive risk metrics for a budget allocation plan.
        """
        total_spend = max(1.0, sum(channel_spend_dict.values()))
        expected_roi = (expected_revenue - total_spend) / total_spend
        
        # Coefficient of Variation (Relative Volatility)
        cv = revenue_std_dev / (expected_revenue + 1e-6)
        
        # 95% Confidence Bounds
        ci_lower = max(0.0, expected_revenue - 1.96 * revenue_std_dev)
        ci_upper = expected_revenue + 1.96 * revenue_std_dev
        
        # Channel Diversification Index (Herfindahl-Hirschman Index)
        shares = [s / total_spend for s in channel_spend_dict.values()]
        hhi = sum(s ** 2 for s in shares)
        diversification_score = round(max(0.0, 1.0 - hhi), 2)
        
        # Risk-Adjusted ROI (Marketing Sharpe Ratio)
        # Higher is better
        marketing_sharpe = round(expected_roi / (cv + 0.05), 2)
        
        # Volatility Grade
        if cv < 0.08:
            risk_tier = "Low Volatility (High Certainty)"
        elif cv < 0.18:
            risk_tier = "Moderate Risk"
        else:
            risk_tier = "High Volatility (Uncertain)"

        return {
            "expected_revenue": float(expected_revenue),
            "revenue_std_dev": float(revenue_std_dev),
            "confidence_interval_95": (float(ci_lower), float(ci_upper)),
            "coefficient_of_variation": float(cv),
            "diversification_score": float(diversification_score),
            "marketing_sharpe_ratio": float(marketing_sharpe),
            "risk_tier": risk_tier
        }
