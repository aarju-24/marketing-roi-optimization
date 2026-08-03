"""
Unit tests for Layer 2: Feature Store.
"""

import pytest
import pandas as pd
from marketing_ai.warehouse.generator import generate_star_schema_data
from marketing_ai.warehouse.db_manager import WarehouseDBManager
from marketing_ai.features.feature_store import FeatureStore


def test_feature_store_pipeline(tmp_path):
    test_db = tmp_path / "test_features.db"
    db_mgr = WarehouseDBManager(db_path=test_db)
    db_mgr.initialize_warehouse(days=30, seed=42, force_recreate=True)
    df_raw = db_mgr.get_consolidated_warehouse_data()

    fs = FeatureStore()
    df_features = fs.build_feature_set(df_raw)

    valid, missing = fs.validate_schema(df_features)
    assert valid, f"Missing features: {missing}"
    assert "ctr" in df_features.columns
    assert "adstock_spend" in df_features.columns
    assert "hill_saturation_index" in df_features.columns
    assert "channel_momentum" in df_features.columns
    assert len(df_features) == len(df_raw)
