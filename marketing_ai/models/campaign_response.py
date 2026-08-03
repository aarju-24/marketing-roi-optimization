"""
Unified Campaign Response Engine.
Combines ML Ensemble (XGBoost/RandomForest) with Marketing Mix (Adstock & Hill Saturation) adjustments.
Exposes a simple predict_response(spend_vector, context) interface for downstream optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from marketing_ai.features.feature_store import FeatureStore
from marketing_ai.utils.config import CHANNELS


class CampaignResponseEngine:
    """
    Unified Response Model predicting Expected Revenue, Conversions, and CPA
    from budget allocation and market context.
    """
    def __init__(self):
        self.feature_store = FeatureStore()
        self.rev_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        self.conv_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        self.is_trained = False
        self.feature_cols = []
        self.rev_std_err = 0.0
        self.conv_std_err = 0.0

    def fit(self, df_features: pd.DataFrame) -> Dict[str, float]:
        """
        Trains response regressors on engineered features dataset.
        """
        self.feature_cols = [col for col in self.feature_store.get_model_features_list() if col in df_features.columns]
        
        X = df_features[self.feature_cols].fillna(0.0)
        y_rev = df_features['revenue'].values
        y_conv = df_features['conversions'].values
        
        # Fit Revenue model
        self.rev_model.fit(X, y_rev)
        pred_rev = self.rev_model.predict(X)
        self.rev_std_err = float(np.std(y_rev - pred_rev))
        r2_rev = float(r2_score(y_rev, pred_rev))
        mae_rev = float(mean_absolute_error(y_rev, pred_rev))

        # Fit Conversions model
        self.conv_model.fit(X, y_conv)
        pred_conv = self.conv_model.predict(X)
        self.conv_std_err = float(np.std(y_conv - pred_conv))
        r2_conv = float(r2_score(y_conv, pred_conv))

        self.is_trained = True
        
        return {
            "r2_revenue": r2_rev,
            "mae_revenue": mae_rev,
            "r2_conversions": r2_conv,
            "rev_std_error": self.rev_std_err
        }

    def predict_channel_response(self, channel_name: str, spend: float, context: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Predicts expected revenue and conversions for a single channel at a specific spend level.
        Applies Marketing Mix (Hill saturation adjustment).
        """
        if not self.is_trained:
            raise RuntimeError("CampaignResponseEngine must be trained before predicting.")

        context = context or {}
        holiday = context.get("holiday", 0)
        is_weekend = context.get("is_weekend", 0)
        month = context.get("month", 6)
        competitor_cpc = context.get("competitor_cpc", 2.10)
        industry_index = context.get("industry_index", 100.0)
        inflation = context.get("inflation", 2.5)

        # Lookup channel params
        ch_meta = next((c for c in CHANNELS if c['channel_name'] == channel_name), CHANNELS[0])
        ch_id = ch_meta['channel_id']
        base_cpc = ch_meta['base_cpc']
        base_cvr = ch_meta['base_cvr']

        # Estimate impressions & clicks
        cpc = base_cpc * (competitor_cpc / 2.10)
        clicks = spend / (cpc + 1e-6)
        impressions = clicks / 0.025
        ctr = 0.025

        # Marketing science features
        adstock = spend * 1.3  # Short-term carryover factor
        hill_sat = (adstock ** 0.85) / (2500.0**0.85 + adstock**0.85 + 1e-8)

        # Construct single feature row matching training schema
        feature_dict = {
            "spend": spend,
            "ctr": ctr,
            "cpc": cpc,
            "cvr": base_cvr,
            "cpa": spend / (max(1.0, clicks * base_cvr)),
            "roi": (spend * ch_meta['roi_multiplier'] - spend) / (spend + 1e-6),
            "profit": spend * (ch_meta['roi_multiplier'] - 1.0),
            "rpc": cpc * ch_meta['roi_multiplier'],
            "rpi": (cpc * ch_meta['roi_multiplier']) * ctr,
            "channel_spend_share": 0.2,
            "channel_revenue_share": 0.2,
            "customer_ltv_proxy": cpc * 25.0,
            "market_share_proxy": 0.05,
            "rolling_7d_spend": spend,
            "rolling_14d_spend": spend,
            "rolling_7d_revenue": spend * ch_meta['roi_multiplier'],
            "rolling_7d_cpa": spend / (max(1.0, clicks * base_cvr)),
            "lag_1d_spend": spend,
            "lag_1d_revenue": spend * ch_meta['roi_multiplier'],
            "channel_momentum": 1.0,
            "adstock_spend": adstock,
            "hill_saturation_index": hill_sat,
            "holiday": holiday,
            "is_weekend": is_weekend,
            "month": month,
            "competitor_cpc": competitor_cpc,
            "industry_index": industry_index,
            "inflation": inflation
        }

        row_df = pd.DataFrame([feature_dict])[self.feature_cols].fillna(0.0)
        
        # Raw ML base prediction
        base_rev = float(self.rev_model.predict(row_df)[0])
        base_conv = float(self.conv_model.predict(row_df)[0])

        # MMM saturation scaling adjustment
        adjusted_rev = max(0.0, base_rev * (0.5 + 0.5 * hill_sat))
        adjusted_conv = max(0.0, base_conv * (0.5 + 0.5 * hill_sat))

        return {
            "expected_revenue": float(adjusted_rev),
            "expected_conversions": float(adjusted_conv),
            "expected_cpa": float(spend / (adjusted_conv + 1e-6)),
            "expected_roi": float((adjusted_rev - spend) / (spend + 1e-6)),
            "rev_std_err": self.rev_std_err
        }

    def predict_response(self, channel_spend_dict: Dict[str, float], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Unified response function querying total expected revenue, conversions, CPA, and channel breakdown.
        Used directly by the Decision Optimizer.
        """
        total_spend = sum(channel_spend_dict.values())
        total_rev = 0.0
        total_conv = 0.0
        channel_breakdown = {}

        for channel_name, spend in channel_spend_dict.items():
            if spend <= 0:
                channel_breakdown[channel_name] = {
                    "spend": 0.0, "expected_revenue": 0.0, "expected_conversions": 0.0, "cpa": 0.0, "roi": 0.0
                }
                continue
                
            resp = self.predict_channel_response(channel_name, spend, context)
            total_rev += resp['expected_revenue']
            total_conv += resp['expected_conversions']
            
            channel_breakdown[channel_name] = {
                "spend": float(spend),
                "expected_revenue": resp['expected_revenue'],
                "expected_conversions": resp['expected_conversions'],
                "cpa": resp['expected_cpa'],
                "roi": resp['expected_roi']
            }

        cpa = total_spend / (total_conv + 1e-6)
        roi = (total_rev - total_spend) / (total_spend + 1e-6)
        
        # Total variance estimate across channels
        combined_std = float(np.sqrt(len(channel_spend_dict) * (self.rev_std_err ** 2)))

        return {
            "total_spend": float(total_spend),
            "expected_revenue": float(total_rev),
            "expected_conversions": float(total_conv),
            "expected_cpa": float(cpa),
            "expected_roi": float(roi),
            "revenue_std_dev": combined_std,
            "confidence_interval_95": (max(0.0, total_rev - 1.96 * combined_std), total_rev + 1.96 * combined_std),
            "channel_breakdown": channel_breakdown
        }
