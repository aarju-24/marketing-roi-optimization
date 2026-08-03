"""
MLOps Experiment Tracker.
Logs training runs, hyperparameter sweeps, and validation metrics (R2, MAE, RMSE).
"""

import json
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from marketing_ai.utils.config import DB_PATH


class ExperimentTracker:
    """
    Tracks ML experiment runs, hyperparameters, and evaluation metrics.
    """
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._initialize_table()

    def _initialize_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS ML_Experiments (
                exp_id TEXT PRIMARY KEY,
                timestamp TEXT,
                model_name TEXT,
                hyperparameters_json TEXT,
                r2_revenue REAL,
                mae_revenue REAL,
                r2_conversions REAL,
                status TEXT
            )
            """)

    def log_experiment(self, exp_id: str, model_name: str, params: Dict[str, Any], r2_rev: float, mae_rev: float, r2_conv: float, status: str = "COMPLETED"):
        """
        Logs experiment trial details.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                INSERT OR REPLACE INTO ML_Experiments (exp_id, timestamp, model_name, hyperparameters_json, r2_revenue, mae_revenue, r2_conversions, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    exp_id, now_str, model_name, json.dumps(params), r2_rev, mae_rev, r2_conv, status
                ))
        except sqlite3.IntegrityError:
            pass  # Ignore duplicate primary key if already recorded

    def get_experiments(self) -> pd.DataFrame:
        """
        Retrieves experiment log DataFrame.
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM ML_Experiments ORDER BY timestamp DESC", conn)
