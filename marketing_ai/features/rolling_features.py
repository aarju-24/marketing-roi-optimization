"""
Rolling & Lagged Time-Series Feature Module.
Computes 7d/14d moving averages, variances, and historical lags per channel.
"""

import pandas as pd
import numpy as np


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling statistics and lag features grouped by channel_id.
    """
    df = df.sort_values(['channel_id', 'date']).reset_index(drop=True)
    
    # 7-day & 14-day rolling revenue and spend
    df['rolling_7d_spend'] = df.groupby('channel_id')['spend'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['rolling_14d_spend'] = df.groupby('channel_id')['spend'].transform(lambda x: x.rolling(14, min_periods=1).mean())
    
    df['rolling_7d_revenue'] = df.groupby('channel_id')['revenue'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['rolling_7d_cpa'] = df.groupby('channel_id')['cpa'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    
    # Spend & Revenue Lags
    df['lag_1d_spend'] = df.groupby('channel_id')['spend'].shift(1).fillna(df['spend'])
    df['lag_1d_revenue'] = df.groupby('channel_id')['revenue'].shift(1).fillna(df['revenue'])
    
    # Moving variance (Volatility signal)
    df['rolling_7d_spend_std'] = df.groupby('channel_id')['spend'].transform(lambda x: x.rolling(7, min_periods=1).std()).fillna(0.0)
    
    # Channel Momentum: Ratio of 7d average spend to 14d average spend
    df['channel_momentum'] = (df['rolling_7d_spend'] + 1e-6) / (df['rolling_14d_spend'] + 1e-6)
    
    return df
