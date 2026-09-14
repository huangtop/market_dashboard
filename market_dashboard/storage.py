from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS daily (
 date TEXT PRIMARY KEY,
 taiex REAL,
 tsmc REAL,
 margin_balance_billion REAL,
 maintenance_est REAL,
 foreign_net_100m REAL,
 usdtwd REAL,
 dxy REAL,
 vix REAL,
 sp500 REAL,
 nasdaq REAL,
 dow REAL,
 sox REAL,
 us10y REAL,
 updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS global_daily (
 date TEXT PRIMARY KEY,
 usdtwd REAL,
 dxy REAL,
 vix REAL,
 sp500 REAL,
 nasdaq REAL,
 dow REAL,
 sox REAL,
 us10y REAL
);
"""


# Existing databases from Step 1 do not have the new US index columns.
# CREATE TABLE IF NOT EXISTS does not alter an existing table, so we migrate
# missing columns automatically at startup.
REQUIRED_COLUMNS = {
    "sp500": "REAL",
    "nasdaq": "REAL",
    "dow": "REAL",
    "sox": "REAL",
    "us10y": "REAL",
}


class Store:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.connect() as con:
            con.executescript(SCHEMA)
            self._migrate(con)
            self._seed_global_daily(con)

    def connect(self):
        return sqlite3.connect(self.path)

    def _seed_global_daily(self, con: sqlite3.Connection) -> None:
        """Seed the new global calendar from legacy Taiwan-keyed rows.

        This is intentionally INSERT OR IGNORE: once a real global-market row
        has been downloaded, it remains authoritative. The seed exists so an
        upgrade can immediately use --derive-only without another Yahoo call.
        """
        con.execute(
            """
            INSERT OR IGNORE INTO global_daily (
                date, usdtwd, dxy, vix, sp500, nasdaq, dow, sox, us10y
            )
            SELECT
                date, usdtwd, dxy, vix, sp500, nasdaq, dow, sox, us10y
            FROM daily
            WHERE usdtwd IS NOT NULL
               OR dxy IS NOT NULL
               OR vix IS NOT NULL
               OR sp500 IS NOT NULL
               OR nasdaq IS NOT NULL
               OR dow IS NOT NULL
               OR sox IS NOT NULL
               OR us10y IS NOT NULL
            """
        )

    def _migrate(self, con: sqlite3.Connection) -> None:
        existing = {
            row[1]
            for row in con.execute("PRAGMA table_info(daily)").fetchall()
        }

        for column, sql_type in REQUIRED_COLUMNS.items():
            if column not in existing:
                con.execute(
                    f"ALTER TABLE daily ADD COLUMN {column} {sql_type}"
                )

    def upsert(self, row: dict):
        cols = list(row)

        sql = (
            f"INSERT INTO daily ({','.join(cols)}) "
            f"VALUES ({','.join('?' for _ in cols)}) "
            "ON CONFLICT(date) DO UPDATE SET "
            + ",".join(
                f"{c}=excluded.{c}"
                for c in cols
                if c != "date"
            )
        )

        with self.connect() as con:
            con.execute(sql, [row[c] for c in cols])

    def update_external(self, external: pd.DataFrame) -> int:
        """Store the full global-market calendar, including US-only dates.

        The old implementation updated only Taiwan trading dates. That loses
        US sessions that occur during Taiwan holidays and can also tempt the
        collector to use an unfinished same-calendar-day US quote. Keeping a
        separate global calendar lets the output layer perform strict as-of
        alignment against the most recently completed external session.
        """
        if external.empty:
            return 0

        fields = [
            "usdtwd",
            "dxy",
            "vix",
            "sp500",
            "nasdaq",
            "dow",
            "sox",
            "us10y",
        ]

        sql = """
        INSERT INTO global_daily (
            date, usdtwd, dxy, vix, sp500, nasdaq, dow, sox, us10y
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            usdtwd = excluded.usdtwd,
            dxy = excluded.dxy,
            vix = excluded.vix,
            sp500 = excluded.sp500,
            nasdaq = excluded.nasdaq,
            dow = excluded.dow,
            sox = excluded.sox,
            us10y = excluded.us10y
        """

        rows = []
        for _, row in external.iterrows():
            day = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            values = [
                None if pd.isna(row.get(field)) else float(row.get(field))
                for field in fields
            ]
            rows.append((day, *values))

        with self.connect() as con:
            con.executemany(sql, rows)

        return len(rows)


    def replace_global(self, external: pd.DataFrame) -> int:
        """Replace the global calendar with a fresh authoritative bulk download."""
        if external.empty:
            return 0

        fields = [
            "usdtwd",
            "dxy",
            "vix",
            "sp500",
            "nasdaq",
            "dow",
            "sox",
            "us10y",
        ]

        rows = []
        for _, row in external.iterrows():
            day = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            values = [
                None if pd.isna(row.get(field)) else float(row.get(field))
                for field in fields
            ]
            rows.append((day, *values))

        sql = """
        INSERT INTO global_daily (
            date, usdtwd, dxy, vix, sp500, nasdaq, dow, sox, us10y
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        with self.connect() as con:
            con.execute("DELETE FROM global_daily")
            con.executemany(sql, rows)

        return len(rows)

    def global_frame(self) -> pd.DataFrame:
        with self.connect() as con:
            df = pd.read_sql_query(
                "SELECT * FROM global_daily ORDER BY date",
                con,
            )

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")

        return df

    def frame(self) -> pd.DataFrame:
        with self.connect() as con:
            df = pd.read_sql_query(
                "SELECT * FROM daily ORDER BY date",
                con,
            )

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])

        return df

    def dates(self) -> set[str]:
        with self.connect() as con:
            return {
                r[0]
                for r in con.execute("SELECT date FROM daily")
            }

    def complete_dates(self) -> set[str]:
        """Dates whose Taiwan daily fields are complete enough to reuse."""
        sql = """
        SELECT date
        FROM daily
        WHERE taiex IS NOT NULL
          AND tsmc IS NOT NULL
          AND margin_balance_billion IS NOT NULL
          AND maintenance_est IS NOT NULL
          AND foreign_net_100m IS NOT NULL
        """

        with self.connect() as con:
            return {r[0] for r in con.execute(sql)}
