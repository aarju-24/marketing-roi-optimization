"""
Decision History & Audit Trail Engine.
Logs every optimization run to SQLite for historical tracking and recommendation comparison.
"""

import json
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from marketing_ai.utils.config import DB_PATH


class DecisionHistoryManager:
    """
    Manages SQLite audit trail for optimization runs.
    """
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._initialize_table()

    def _initialize_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS Decision_History_Audit (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                total_budget REAL,
                weights_json TEXT,
                optimal_allocation_json TEXT,
                predicted_revenue REAL,
                predicted_conversions REAL,
                predicted_cpa REAL,
                model_version TEXT
            )
            """)

    def log_run(self, run_id: str, total_budget: float, weights: Dict[str, float], optimal_allocation: Dict[str, float], predicted_revenue: float, predicted_conversions: float, predicted_cpa: float, model_version: str = "v1.0.0"):
        """
        Logs an optimization run into SQLite audit table.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            INSERT INTO Decision_History_Audit (run_id, timestamp, total_budget, weights_json, optimal_allocation_json, predicted_revenue, predicted_conversions, predicted_cpa, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                now_str,
                total_budget,
                json.dumps(weights),
                json.dumps(optimal_allocation),
                predicted_revenue,
                predicted_conversions,
                predicted_cpa,
                model_version
            ))

    def get_run_history(self, limit: int = 50) -> pd.DataFrame:
        """
        Retrieves recent optimization runs DataFrame.
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM Decision_History_Audit ORDER BY timestamp DESC LIMIT ?", conn, params=(limit,))
        return df
