from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily (
 date TEXT PRIMARY KEY,
 taiex REAL, tsmc REAL,
 margin_balance_billion REAL,
 maintenance_est REAL,
 foreign_net_100m REAL,
 usdtwd REAL, dxy REAL, vix REAL,
 updated_at TEXT NOT NULL
);
"""

class Store:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con: con.executescript(SCHEMA)
    def connect(self): return sqlite3.connect(self.path)
    def upsert(self, row: dict):
        cols = list(row)
        sql = f"INSERT INTO daily ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)}) ON CONFLICT(date) DO UPDATE SET " + ",".join(f"{c}=excluded.{c}" for c in cols if c != 'date')
        with self.connect() as con: con.execute(sql, [row[c] for c in cols])
    def frame(self) -> pd.DataFrame:
        with self.connect() as con:
            df = pd.read_sql_query("SELECT * FROM daily ORDER BY date", con)
        if not df.empty: df["date"] = pd.to_datetime(df["date"])
        return df
    def dates(self) -> set[str]:
        with self.connect() as con:
            return {r[0] for r in con.execute("SELECT date FROM daily")}
