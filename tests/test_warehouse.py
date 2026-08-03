"""
Unit tests for Layer 1: Data Warehouse.
"""

import os
import pytest
import pandas as pd
from marketing_ai.warehouse.generator import generate_star_schema_data
from marketing_ai.warehouse.db_manager import WarehouseDBManager
from marketing_ai.utils.config import DB_PATH


def test_star_schema_generator():
    data_dict = generate_star_schema_data(days=30, seed=123)
    assert "Fact_Campaign_Performance" in data_dict
    assert "Dim_Channel" in data_dict
    assert "Dim_Customer" in data_dict
    assert "Dim_Calendar" in data_dict
    assert "Dim_Market" in data_dict

    df_fact = data_dict["Fact_Campaign_Performance"]
    assert len(df_fact) > 0
    assert "revenue" in df_fact.columns
    assert "spend" in df_fact.columns or "budget" in df_fact.columns


def test_db_manager_initialization(tmp_path):
    test_db = tmp_path / "test_warehouse.db"
    db_mgr = WarehouseDBManager(db_path=test_db)
    db_mgr.initialize_warehouse(days=30, seed=42, force_recreate=True)
    
    assert test_db.exists()
    
    df_consolidated = db_mgr.get_consolidated_warehouse_data()
    assert isinstance(df_consolidated, pd.DataFrame)
    assert len(df_consolidated) > 0
    assert "spend" in df_consolidated.columns
    assert "revenue" in df_consolidated.columns
    assert "channel_name" in df_consolidated.columns
    assert "platform_type" in df_consolidated.columns
