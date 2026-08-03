"""
Database Manager for Star Schema Data Warehouse.
Manages SQLite tables, relational integrity, data ingestion, and analytical joins.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from marketing_ai.utils.config import DB_PATH
from marketing_ai.warehouse.generator import generate_star_schema_data


class WarehouseDBManager:
    """
    Manages SQLite Data Warehouse operations and Star Schema queries.
    """
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def get_connection(self):
        """Returns a SQLite connection object."""
        return sqlite3.connect(self.db_path)

    def initialize_warehouse(self, days: int = 730, seed: int = 42, force_recreate: bool = False):
        """
        Creates SQLite tables and populates them with initial Star Schema dataset.
        """
        if self.db_path.exists() and not force_recreate:
            # Check if tables exist
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Fact_Campaign_Performance'")
                if cursor.fetchone():
                    return  # Already initialized

        data_dict = generate_star_schema_data(days=days, seed=seed)
        
        with self.get_connection() as conn:
            for table_name, df in data_dict.items():
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                
            # Create Indexes for fast querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fact_date ON Fact_Campaign_Performance(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fact_channel ON Fact_Campaign_Performance(channel_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fact_segment ON Fact_Campaign_Performance(segment_id)")

    def get_consolidated_warehouse_data(self) -> pd.DataFrame:
        """
        Executes a Star Schema JOIN to produce a consolidated flat dataset
        joining Fact_Campaign_Performance with Dim_Channel, Dim_Customer, Dim_Calendar, and Dim_Market.
        """
        query = """
        SELECT 
            f.campaign_id,
            f.date,
            f.channel_id,
            c.channel_name,
            c.platform_type,
            f.segment_id,
            cust.age_group,
            cust.device,
            cust.country,
            cal.holiday,
            cal.week,
            cal.month,
            cal.quarter,
            cal.season,
            cal.is_weekend,
            mkt.competitor_cpc,
            mkt.industry_index,
            mkt.inflation,
            f.budget AS spend,
            f.impressions,
            f.clicks,
            f.conversions,
            f.revenue
        FROM Fact_Campaign_Performance f
        JOIN Dim_Channel c ON f.channel_id = c.channel_id
        JOIN Dim_Customer cust ON f.segment_id = cust.segment_id
        JOIN Dim_Calendar cal ON f.date = cal.date
        JOIN Dim_Market mkt ON f.date = mkt.date
        ORDER BY f.date ASC, f.channel_id ASC
        """
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn)
        return df

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """Executes a custom SQL query and returns result as DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql_query(sql, conn, params=params)
