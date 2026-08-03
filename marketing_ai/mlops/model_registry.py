"""
MLOps Model Registry.
Manages production model versions, staging tags, and artifact metadata.
"""

import json
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from marketing_ai.utils.config import DB_PATH


class ModelRegistry:
    """
    Manages local Model Registry tags and production version metadata.
    """
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._initialize_table()

    def _initialize_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS Model_Registry (
                version_tag TEXT PRIMARY KEY,
                registered_at TEXT,
                model_type TEXT,
                stage TEXT,
                r2_revenue REAL,
                mae_revenue REAL,
                artifact_path TEXT
            )
            """)

    def register_model(self, version_tag: str, model_type: str, r2_rev: float, mae_rev: float, stage: str = "Staging", artifact_path: str = "models/v1"):
        """
        Registers a new model version.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            INSERT OR REPLACE INTO Model_Registry (version_tag, registered_at, model_type, stage, r2_revenue, mae_revenue, artifact_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (version_tag, now_str, model_type, stage, r2_rev, mae_rev, artifact_path))

    def promote_to_production(self, version_tag: str):
        """
        Promotes a specific model version to Production and demotes others.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE Model_Registry SET stage = 'Archived' WHERE stage = 'Production'")
            conn.execute("UPDATE Model_Registry SET stage = 'Production' WHERE version_tag = ?", (version_tag,))

    def get_registered_models(self) -> pd.DataFrame:
        """
        Retrieves all registered models.
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM Model_Registry ORDER BY registered_at DESC", conn)
