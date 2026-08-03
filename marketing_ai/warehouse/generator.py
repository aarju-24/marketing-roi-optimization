"""
Star Schema Synthetic Data Generator.
Generates realistic multi-table campaign, customer, calendar, and market dimension data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from marketing_ai.utils.config import CHANNELS, CUSTOMER_SEGMENTS


def generate_star_schema_data(days: int = 730, seed: int = 42) -> dict:
    """
    Generates DataFrames representing a Star Schema Data Warehouse.
    
    Returns:
        dict containing 'Fact_Campaign_Performance', 'Dim_Channel',
        'Dim_Customer', 'Dim_Calendar', and 'Dim_Market'.
    """
    np.random.seed(seed)
    start_date = datetime(2024, 1, 1)

    # 1. Dim_Channel
    df_dim_channel = pd.DataFrame(CHANNELS)[['channel_id', 'channel_name', 'platform_type']]

    # 2. Dim_Customer
    df_dim_customer = pd.DataFrame(CUSTOMER_SEGMENTS)

    # 3. Dim_Calendar
    calendar_rows = []
    for i in range(days):
        dt = start_date + timedelta(days=i)
        date_str = dt.strftime("%Y-%m-%d")
        month = dt.month
        quarter = (month - 1) // 3 + 1
        is_weekend = 1 if dt.weekday() >= 5 else 0
        
        # Holidays: Black Friday / Cyber Monday (Nov late), Christmas (Dec late), New Year (Jan 1)
        is_holiday = 0
        if (month == 11 and 23 <= dt.day <= 30) or (month == 12 and 15 <= dt.day <= 31) or (month == 1 and dt.day == 1):
            is_holiday = 1
            
        season = "Winter" if month in [12, 1, 2] else ("Spring" if month in [3, 4, 5] else ("Summer" if month in [6, 7, 8] else "Autumn"))
        
        calendar_rows.append({
            "date": date_str,
            "holiday": is_holiday,
            "week": dt.isocalendar()[1],
            "month": month,
            "quarter": quarter,
            "season": season,
            "is_weekend": is_weekend
        })
    df_dim_calendar = pd.DataFrame(calendar_rows)

    # 4. Dim_Market
    market_rows = []
    base_cpc_trend = 1.0
    for i in range(days):
        dt = start_date + timedelta(days=i)
        date_str = dt.strftime("%Y-%m-%d")
        # Inflation & CPC trend gradually increases by 8% per year + random walk
        base_cpc_trend += np.random.normal(0.0001, 0.002)
        base_cpc_trend = max(0.85, base_cpc_trend)
        
        competitor_cpc = round(2.10 * base_cpc_trend + np.random.normal(0, 0.15), 2)
        industry_index = round(100.0 * (1 + 0.05 * (i / 365) + np.random.normal(0, 0.02)), 2)
        inflation = round(2.5 + 0.5 * (i / 365) + np.random.normal(0, 0.1), 2)
        
        market_rows.append({
            "date": date_str,
            "competitor_cpc": max(0.50, competitor_cpc),
            "industry_index": industry_index,
            "inflation": inflation
        })
    df_dim_market = pd.DataFrame(market_rows)

    # 5. Fact_Campaign_Performance
    fact_rows = []
    campaign_id_counter = 1001

    # Carryover state per channel for adstock simulation
    channel_adstock_state = {c['channel_id']: 0.0 for c in CHANNELS}

    for i in range(days):
        cal_row = df_dim_calendar.iloc[i]
        mkt_row = df_dim_market.iloc[i]
        date_str = cal_row['date']
        
        seasonality_mult = 1.4 if cal_row['holiday'] == 1 else (1.1 if cal_row['season'] == 'Autumn' else 1.0)
        
        for ch in CHANNELS:
            ch_id = ch['channel_id']
            # Pick 1-2 target customer segments per campaign
            seg_id = np.random.choice(df_dim_customer['segment_id'].values)
            
            # Base budget per channel per day ($200 - $5,000)
            base_budget = np.random.uniform(500, 3500) if ch['platform_type'] in ['Search', 'Social'] else np.random.uniform(1500, 8000)
            budget = round(base_budget * seasonality_mult * np.random.uniform(0.85, 1.2), 2)
            
            # Adstock memory effect (decay lambda = 0.5)
            channel_adstock_state[ch_id] = 0.5 * channel_adstock_state[ch_id] + budget
            effective_spend = channel_adstock_state[ch_id]
            
            # CPC and CPM modeling
            cpc = round(ch['base_cpc'] * (mkt_row['competitor_cpc'] / 2.10) * np.random.uniform(0.9, 1.1), 2)
            cpc = max(0.10, cpc)
            
            clicks = max(10, int(budget / cpc))
            impressions = int(clicks / np.random.uniform(0.015, 0.045))
            
            # Diminishing returns (Hill-like curve on effective spend)
            # Saturation threshold around $4,000 daily spend
            saturation_factor = (effective_spend ** 0.8) / (3000**0.8 + effective_spend**0.8)
            
            # Conversions
            cvr = ch['base_cvr'] * seasonality_mult * np.random.uniform(0.85, 1.15)
            conversions = max(1, int(clicks * cvr * saturation_factor * 2.0))
            
            # Revenue calculation
            avg_order_value = np.random.uniform(65.0, 140.0) if ch_id != 4 else np.random.uniform(250.0, 600.0) # B2B higher AOV
            revenue = round(conversions * avg_order_value * (ch['roi_multiplier'] / 3.0), 2)
            
            fact_rows.append({
                "campaign_id": campaign_id_counter,
                "date": date_str,
                "channel_id": ch_id,
                "segment_id": seg_id,
                "budget": budget,
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "revenue": revenue
            })
            campaign_id_counter += 1

    df_fact_campaign = pd.DataFrame(fact_rows)

    return {
        "Fact_Campaign_Performance": df_fact_campaign,
        "Dim_Channel": df_dim_channel,
        "Dim_Customer": df_dim_customer,
        "Dim_Calendar": df_dim_calendar,
        "Dim_Market": df_dim_market
    }
