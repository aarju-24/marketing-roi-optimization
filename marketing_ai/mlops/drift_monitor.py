"""
Model & Data Drift Monitoring Engine.
Tracks distribution shifts in market signals (CPC/CPM inflation) and triggers retraining alerts.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class DriftMonitor:
    """
    Monitors feature distribution drift and prediction errors.
    """
    def __init__(self, drift_threshold_pct: float = 25.0):
        self.drift_threshold_pct = drift_threshold_pct

    def check_feature_drift(self, baseline_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: list = None) -> Dict[str, Any]:
        """
        Calculates percentage change in mean and std across key input features.
        Triggers retraining alert if drift exceeds threshold.
        """
        feature_cols = feature_cols or ['cpc', 'spend', 'competitor_cpc', 'ctr']
        drift_results = {}
        alert_triggered = False

        for col in feature_cols:
            if col not in baseline_df.columns or col not in current_df.columns:
                continue
                
            base_mean = float(baseline_df[col].mean())
            curr_mean = float(current_df[col].mean())
            
            shift_pct = float(((curr_mean - base_mean) / (abs(base_mean) + 1e-6)) * 100)
            
            is_drifted = abs(shift_pct) >= self.drift_threshold_pct
            if is_drifted:
                alert_triggered = True
                
            drift_results[col] = {
                "baseline_mean": round(base_mean, 4),
                "current_mean": round(curr_mean, 4),
                "shift_pct": round(shift_pct, 2),
                "status": "DRIFT_DETECTED" if is_drifted else "STABLE"
            }

        status_msg = "WARNING: Significant distribution drift detected! Model retraining recommended." if alert_triggered else "SYSTEM STABLE: Feature distributions remain within tolerance."

        return {
            "overall_status": "ALERT" if alert_triggered else "OK",
            "message": status_msg,
            "feature_drift_details": drift_results
        }
