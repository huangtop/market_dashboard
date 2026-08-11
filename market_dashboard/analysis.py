from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(s: pd.Series, window=252, minimum=60):
    return (s - s.rolling(window, min_periods=minimum).mean()) / s.rolling(
        window, min_periods=minimum
    ).std()


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    for c in ["usdtwd", "dxy", "vix"]:
        if c in df:
            df[c] = df[c].ffill(limit=4)

    df["foreign_20d_sum_100m"] = df["foreign_net_100m"].rolling(20, min_periods=5).sum()
    df["usdtwd_20d_change_pct"] = df["usdtwd"].pct_change(20) * 100
    df["taiex_volume_billion"] = np.nan

    pressure = (
        zscore(df["usdtwd_20d_change_pct"])
        + zscore(df["dxy"])
        - zscore(df["foreign_20d_sum_100m"])
    )
    df["outflow_pressure_score"] = pressure

    # maintenance_est 是公開資料推導的「市場融資擔保比估算」，不是券商整戶維持率。
    # 因此融資熱度使用自身歷史分位，不套用 145/170 等券商維持率語意門檻。
    df["maintenance_percentile"] = df["maintenance_est"].expanding(min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
    )
    maint = df["maintenance_percentile"].clip(0, 100)
    foreign = (zscore(df["foreign_20d_sum_100m"]).clip(-2, 2) + 2) / 4 * 100
    calm = 100 - (df["vix"].clip(10, 45) - 10) / 35 * 100
    fx = 100 - (pressure.clip(-2, 2) + 2) / 4 * 100
    df["market_temperature"] = (
        0.35 * maint + 0.30 * foreign + 0.20 * calm + 0.15 * fx
    ).clip(0, 100)
    return df


def backtests(df: pd.DataFrame, thresholds: list[float]):
    """Backtest drops in the proxy's historical percentile.

    ``thresholds`` are percentile levels (for example 10/25/75/90), not
    account-maintenance-ratio percentages.
    """
    out = []
    x = df.reset_index(drop=True)
    metric = x["maintenance_percentile"]
    for threshold in thresholds:
        idx = x.index[(metric < threshold) & (metric.shift(1) >= threshold)].tolist()
        horizons = []
        for days in (20, 60, 120):
            vals = [
                x.loc[i + days, "taiex"] / x.loc[i, "taiex"] - 1
                for i in idx
                if i + days < len(x) and x.loc[i, "taiex"] > 0
            ]
            horizons.append(
                {
                    "trading_days": days,
                    "samples": len(vals),
                    "average_return_pct": round(float(np.mean(vals) * 100), 2) if vals else None,
                    "median_return_pct": round(float(np.median(vals) * 100), 2) if vals else None,
                    "win_rate_pct": round(float(np.mean(np.array(vals) > 0) * 100), 1) if vals else None,
                }
            )
        out.append({"threshold": threshold, "events": len(idx), "horizons": horizons})
    return out


def commentary(row: pd.Series) -> str:
    parts = []
    mp = row.get("maintenance_percentile")
    f = row.get("foreign_20d_sum_100m")
    v = row.get("vix")
    p = row.get("outflow_pressure_score")

    if pd.notna(mp):
        if mp >= 90:
            parts.append("市場融資擔保比估算處歷史高檔")
        elif mp >= 70:
            parts.append("市場融資擔保比估算處歷史偏高區")
        elif mp >= 30:
            parts.append("市場融資擔保比估算處歷史中性區")
        elif mp >= 10:
            parts.append("市場融資擔保比估算處歷史偏低區")
        else:
            parts.append("市場融資擔保比估算處歷史低檔")
    if pd.notna(f):
        parts.append("外資近20日偏買方" if f > 0 else "外資近20日偏賣方")
    if pd.notna(v):
        parts.append("市場波動顯著升高" if v >= 30 else "波動風險偏高" if v >= 20 else "波動情緒相對平穩")
    if pd.notna(p):
        parts.append("匯率與美元形成資金外流壓力" if p > 1 else "外部資金壓力有限" if p < 0 else "外部資金壓力中性")
    return "；".join(parts) + "。此內容為量化資料摘要，不構成投資建議。"
