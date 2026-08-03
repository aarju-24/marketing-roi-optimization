"""
Raw Feature Engineering Module.
Computes fundamental marketing metrics and efficiency ratios.
"""

import numpy as np
import pandas as pd


def compute_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes baseline ratio features from raw campaign facts.
    """
    df = df.copy()
    
    # Avoid division by zero with small epsilon
    eps = 1e-6
    
    df['ctr'] = df['clicks'] / (df['impressions'] + eps)
    df['cpc'] = df['spend'] / (df['clicks'] + eps)
    df['cvr'] = df['conversions'] / (df['clicks'] + eps)
    df['cpa'] = df['spend'] / (df['conversions'] + eps)
    df['roi'] = (df['revenue'] - df['spend']) / (df['spend'] + eps)
    df['profit'] = df['revenue'] - df['spend']
    df['rpc'] = df['revenue'] / (df['clicks'] + eps)
    df['rpi'] = df['revenue'] / (df['impressions'] + eps)
    
    # Clean inf/nan
    ratio_cols = ['ctr', 'cpc', 'cvr', 'cpa', 'roi', 'profit', 'rpc', 'rpi']
    for col in ratio_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
    return df
