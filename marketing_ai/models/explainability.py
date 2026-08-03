"""
Model Explainability Module.
Computes feature importances and driver attributions for executive decision support.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from marketing_ai.models.campaign_response import CampaignResponseEngine


class ModelExplainer:
    """
    Extracts feature importances and driver rationales from trained CampaignResponseEngine.
    """
    def __init__(self, response_engine: CampaignResponseEngine):
        self.engine = response_engine

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Returns normalized feature importance dictionary.
        """
        if not self.engine.is_trained:
            return {}
            
        importances = self.engine.rev_model.feature_importances_
        feature_names = self.engine.feature_cols
        
        imp_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
        # Sort descending
        return dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))

    def get_driver_explanation(self, channel_spend_dict: Dict[str, float], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Generates business driver impact explanations (Positive Drivers vs Negative Drivers).
        """
        importances = self.get_feature_importances()
        top_features = list(importances.keys())[:5]
        
        explanations = []
        
        for feat in top_features:
            impact_type = "Positive Driver" if feat in ['adstock_spend', 'hill_saturation_index', 'rolling_7d_revenue', 'holiday'] else "Cost & Volatility Factor"
            explanations.append({
                "feature": feat,
                "importance_score": round(importances[feat], 4),
                "classification": impact_type,
                "business_meaning": self._map_business_meaning(feat)
            })
            
        return explanations

    def _map_business_meaning(self, feature_name: str) -> str:
        meanings = {
            "adstock_spend": "Historical carryover spend building brand memory and persistent awareness",
            "hill_saturation_index": "Diminishing marginal returns threshold on spend volume",
            "rolling_7d_revenue": "Recent 7-day revenue momentum and campaign velocity",
            "holiday": "Seasonal demand surge during Q4 and promotional holidays",
            "competitor_cpc": "Market competition pressure raising cost-per-click",
            "channel_momentum": "Acceleration of daily spend relative to 14-day average",
            "cpa": "Cost per acquisition efficiency metric",
            "ctr": "Ad copy relevance and click-through engagement rate"
        }
        return meanings.get(feature_name, "Key operational input metric affecting revenue forecast")
