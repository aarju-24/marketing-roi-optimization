"""
Marketing Science Transformation Module.
Implements Adstock carryover (memory decay) and Hill function (diminishing returns).
"""

import pandas as pd
import numpy as np


def apply_adstock_transformation(spend_series: pd.Series, decay_rate: float = 0.5) -> pd.Series:
    """
    Applies geometric adstock decay transformation.
    A_t = S_t + decay_rate * A_{t-1}
    """
    adstock = np.zeros(len(spend_series))
    spend_vals = spend_series.values
    for t in range(len(spend_vals)):
        if t == 0:
            adstock[t] = spend_vals[t]
        else:
            adstock[t] = spend_vals[t] + decay_rate * adstock[t-1]
    return pd.Series(adstock, index=spend_series.index)


def apply_hill_saturation(spend_series: pd.Series, half_saturation: float = 3000.0, slope: float = 0.8) -> pd.Series:
    """
    Applies Hill saturation function modeling diminishing marginal returns.
    S(x) = x^slope / (half_saturation^slope + x^slope)
    """
    x = np.maximum(0.0, spend_series.values)
    num = x ** slope
    den = (half_saturation ** slope) + num
    saturation = num / (den + 1e-8)
    return pd.Series(saturation, index=spend_series.index)


def compute_marketing_science_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates channel-specific Adstock spend and Hill Saturation index.
    """
    df = df.sort_values(['channel_id', 'date']).reset_index(drop=True)
    
    # Apply Adstock per channel
    adstock_list = []
    saturation_list = []
    
    for ch_id, group in df.groupby('channel_id'):
        # Decay rates: TV=0.7, Social=0.4, Search=0.2, Email=0.3
        decay = 0.7 if ch_id == 6 else (0.4 if ch_id in [2, 3] else (0.5 if ch_id == 4 else 0.2))
        adstock = apply_adstock_transformation(group['spend'], decay_rate=decay)
        saturation = apply_hill_saturation(adstock, half_saturation=2500.0, slope=0.85)
        
        adstock_list.append(adstock)
        saturation_list.append(saturation)
        
    df['adstock_spend'] = pd.concat(adstock_list).sort_index()
    df['hill_saturation_index'] = pd.concat(saturation_list).sort_index()
    
    return df
