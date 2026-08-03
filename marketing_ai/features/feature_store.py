"""
Central Feature Store Engine & Metadata Registry.
Orchestrates raw, engineered, rolling, and marketing science features.
Ensures train/inference consistency and tracks feature versioning.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from marketing_ai.features.raw_features import compute_raw_features
from marketing_ai.features.engineered_features import compute_engineered_features
from marketing_ai.features.rolling_features import compute_rolling_features
from marketing_ai.features.marketing_science import compute_marketing_science_features


class FeatureStore:
    """
    Central Feature Store managing feature computation, schema validation, and versioning.
    """
    FEATURE_STORE_VERSION = "v1.0.0"
    
    FEATURE_CATALOG = {
        "raw": ["ctr", "cpc", "cvr", "cpa", "roi", "profit", "rpc", "rpi"],
        "engineered": ["channel_spend_share", "channel_revenue_share", "customer_ltv_proxy", "market_share_proxy"],
        "rolling": ["rolling_7d_spend", "rolling_14d_spend", "rolling_7d_revenue", "rolling_7d_cpa", "lag_1d_spend", "lag_1d_revenue", "channel_momentum"],
        "marketing_science": ["adstock_spend", "hill_saturation_index"],
        "context": ["holiday", "is_weekend", "month", "competitor_cpc", "industry_index", "inflation"]
    }

    def __init__(self):
        self.version = self.FEATURE_STORE_VERSION

    def build_feature_set(self, df_warehouse: pd.DataFrame) -> pd.DataFrame:
        """
        Runs full feature engineering pipeline over star schema warehouse dataframe.
        """
        df = df_warehouse.copy()
        
        # 1. Raw features
        df = compute_raw_features(df)
        
        # 2. Engineered business features
        df = compute_engineered_features(df)
        
        # 3. Rolling & lag features
        df = compute_rolling_features(df)
        
        # 4. Marketing science features
        df = compute_marketing_science_features(df)
        
        return df

    def get_model_features_list(self) -> List[str]:
        """
        Returns flat list of feature column names used for training response models.
        """
        features = []
        for cat, cols in self.FEATURE_CATALOG.items():
            features.extend(cols)
        # Add budget / spend itself
        if "spend" not in features:
            features.append("spend")
        return features

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validates whether required feature columns are present in DataFrame.
        """
        required = self.get_model_features_list()
        missing = [col for col in required if col not in df.columns]
        return (len(missing) == 0, missing)
