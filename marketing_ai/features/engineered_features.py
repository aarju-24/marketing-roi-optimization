"""
Engineered Business & Customer Feature Module.
Calculates Customer LTV proxy, Market Share proxy, and Channel Share metrics.
"""

import pandas as pd
import numpy as np


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific engineered features.
    """
    df = df.copy()
    
    # Daily total spend across all channels
    daily_spend = df.groupby('date')['spend'].transform('sum')
    df['channel_spend_share'] = df['spend'] / (daily_spend + 1e-6)
    
    # Daily total revenue
    daily_revenue = df.groupby('date')['revenue'].transform('sum')
    df['channel_revenue_share'] = df['revenue'] / (daily_revenue + 1e-6)
    
    # Customer LTV Proxy (Higher for Desktop and B2B LinkedIn/Search)
    ltv_base = np.where(df['device'] == 'Desktop', 1.3, 1.0) * np.where(df['channel_name'] == 'LinkedIn Ads', 2.0, 1.0)
    df['customer_ltv_proxy'] = round(df['rpc'] * 12.0 * ltv_base, 2)
    
    # Market Share Proxy (Channel volume relative to competitor index)
    df['market_share_proxy'] = np.clip((df['impressions'] / (df['industry_index'] * 1000 + 1e-6)), 0.0, 1.0)
    
    return df
