"""
System configuration and constants for Marketing Decision Intelligence Platform.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "marketing_warehouse.db"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Marketing Channels Configuration
CHANNELS = [
    {"channel_id": 1, "channel_name": "Google Search", "platform_type": "Search", "base_cpc": 2.50, "base_cvr": 0.045, "roi_multiplier": 3.8},
    {"channel_id": 2, "channel_name": "Facebook Ads", "platform_type": "Social", "base_cpc": 1.80, "base_cvr": 0.035, "roi_multiplier": 3.2},
    {"channel_id": 3, "channel_name": "Instagram Ads", "platform_type": "Social", "base_cpc": 2.10, "base_cvr": 0.038, "roi_multiplier": 3.4},
    {"channel_id": 4, "channel_name": "LinkedIn Ads", "platform_type": "B2B Social", "base_cpc": 5.20, "base_cvr": 0.028, "roi_multiplier": 2.9},
    {"channel_id": 5, "channel_name": "Email Marketing", "platform_type": "Direct", "base_cpc": 0.30, "base_cvr": 0.065, "roi_multiplier": 5.1},
    {"channel_id": 6, "channel_name": "TV & Brand Video", "platform_type": "Offline/Display", "base_cpc": 4.00, "base_cvr": 0.015, "roi_multiplier": 2.2},
]

# Customer Segments
CUSTOMER_SEGMENTS = [
    {"segment_id": 1, "age_group": "18-24", "device": "Mobile", "country": "US"},
    {"segment_id": 2, "age_group": "25-34", "device": "Mobile", "country": "US"},
    {"segment_id": 3, "age_group": "25-34", "device": "Desktop", "country": "US"},
    {"segment_id": 4, "age_group": "35-54", "device": "Desktop", "country": "US"},
    {"segment_id": 5, "age_group": "35-54", "device": "Mobile", "country": "UK"},
    {"segment_id": 6, "age_group": "55+", "device": "Desktop", "country": "US"},
]

# Default Multi-Objective Optimizer Weights
DEFAULT_OPTIMIZER_WEIGHTS = {
    "revenue": 1.0,
    "conversions": 0.5,
    "cpa_penalty": 0.3,
    "risk_penalty": 0.2
}
