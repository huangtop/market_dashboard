from __future__ import annotations
import math
import time
from datetime import date
from typing import Any
import numpy as np
import pandas as pd
import requests

TWSE_BASE = "https://www.twse.com.tw/rwd/zh"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.twse.com.tw/",
}

def number(value: Any) -> float:
    try:
        s = str(value).strip().replace(",", "").replace('"', "").replace("+", "")
        if s in {"", "--", "---", "X", "None", "nan"}: return np.nan
        return float(s)
    except Exception:
        return np.nan

def roc_date(value: Any) -> pd.Timestamp:
    try:
        y,m,d = map(int, str(value).replace("-", "/").split("/"))
        return pd.Timestamp(y + 1911 if y < 1911 else y, m, d)
    except Exception:
        return pd.NaT

def safe_float(value: Any, digits: int = 3):
    try:
        x = float(value)
        return None if not math.isfinite(x) else round(x, digits)
    except Exception:
        return None

def get_json(session: requests.Session, path: str, params: dict, timeout: int, retries: int) -> dict:
    url = f"{TWSE_BASE}/{path}"
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if data.get("stat") not in (None, "OK"):
                raise RuntimeError(f"TWSE stat={data.get('stat')}")
            return data
        except Exception as exc:
            last = exc
            if attempt < retries:
                time.sleep(attempt * 1.5)
    raise RuntimeError(f"TWSE 下載失敗 {path}: {last}")

def tables(payload: dict):
    if payload.get("fields") and payload.get("data"):
        yield payload["fields"], payload["data"], payload.get("title", "")
    for t in payload.get("tables", []):
        if t.get("fields") and t.get("data"):
            yield t["fields"], t["data"], t.get("title", "")

def find_table(payload: dict, required: list[str]) -> pd.DataFrame:
    for fields, rows, _ in tables(payload):
        normalized = [str(x).replace(" ", "").replace("\n", "") for x in fields]
        if all(any(req in col for col in normalized) for req in required):
            return pd.DataFrame(rows, columns=fields)
    available = [list(f) for f,_,_ in tables(payload)]
    raise KeyError(f"找不到欄位 {required}; 可用欄位={available}")

def month_starts(start: date, end: date):
    cur = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cur <= last:
        yield cur
        cur = date(cur.year + (cur.month == 12), 1 if cur.month == 12 else cur.month + 1, 1)
