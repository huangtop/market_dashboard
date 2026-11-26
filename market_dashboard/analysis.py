from __future__ import annotations

import numpy as np
import pandas as pd



GLOBAL_MARKET_FIELDS = (
    "usdtwd",
    "dxy",
    "vix",
    "sp500",
    "nasdaq",
    "dow",
    "sox",
    "us10y",
)

GLOBAL_CHANGE_FIELDS = {
    "sp500": "sp500_change_pct",
    "nasdaq": "nasdaq_change_pct",
    "dow": "dow_change_pct",
    "sox": "sox_change_pct",
    "vix": "vix_change_pct",
    "us10y": "us10y_change_bp",
}


def add_global_daily_changes(global_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate moves between consecutive valid sessions for each market.

    global_daily uses the union of calendars from Yahoo symbols, so a row may
    exist because one market traded while another field is NaN. Daily changes
    therefore must be calculated on each field's own valid-session series,
    rather than across the union-calendar rows.
    """
    if global_df is None or global_df.empty:
        return global_df

    x = global_df.sort_values("date").copy()

    for field in ("sp500", "nasdaq", "dow", "sox", "vix"):
        if field not in x.columns:
            continue

        values = pd.to_numeric(x[field], errors="coerce")

        # Calculate return against the previous VALID session for this field.
        valid = values.dropna()
        changes = valid.pct_change(fill_method=None) * 100

        # Keep the union-calendar index. Dates where this market has no quote
        # remain NaN, while valid sessions receive the correct daily return.
        x[f"{field}_change_pct"] = changes.reindex(x.index)

    if "us10y" in x.columns:
        values = pd.to_numeric(x["us10y"], errors="coerce")

        # ^TNX is stored as percentage yield.
        # Example: 4.975 - 4.944 = 0.031 percentage point = +3.1 bp.
        valid = values.dropna()
        changes = valid.diff() * 100

        x["us10y_change_bp"] = changes.reindex(x.index)

    return x


def align_global_asof(
    taiwan_df: pd.DataFrame,
    global_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align Taiwan date to the latest strictly-prior completed global session.

    Daily move fields are calculated on raw global sessions first and aligned
    together with their base field.  This guarantees US10Y change_bp and stock
    index daily returns refer to the exact same source session as the displayed
    level.
    """
    left = taiwan_df.sort_values("date").copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce")

    for field in GLOBAL_MARKET_FIELDS:
        left[field] = np.nan
        left[f"{field}_date"] = pd.NaT
        change_field = GLOBAL_CHANGE_FIELDS.get(field)
        if change_field:
            left[change_field] = np.nan

    if global_df is None or global_df.empty:
        return left

    global_work = add_global_daily_changes(global_df.copy())
    global_work["date"] = pd.to_datetime(global_work["date"], errors="coerce")
    global_work = (
        global_work
        .dropna(subset=["date"])
        .sort_values("date")
    )

    base_dates = left[["date"]].sort_values("date")

    for field in GLOBAL_MARKET_FIELDS:
        if field not in global_work.columns:
            continue

        change_field = GLOBAL_CHANGE_FIELDS.get(field)
        keep = ["date", field]
        if change_field and change_field in global_work.columns:
            keep.append(change_field)

        required = [field]
        if change_field and change_field in global_work.columns:
            required.append(change_field)

        right = (
            global_work[keep]
            .dropna(subset=required)
            .sort_values("date")
            .rename(columns={"date": f"{field}_date"})
        )
        
        if right.empty:
            continue

        aligned = pd.merge_asof(
            base_dates,
            right,
            left_on="date",
            right_on=f"{field}_date",
            direction="backward",
            allow_exact_matches=(field == "usdtwd"),
        )

        left[field] = aligned[field].to_numpy()
        left[f"{field}_date"] = aligned[f"{field}_date"].to_numpy()
        if change_field and change_field in aligned.columns:
            left[change_field] = aligned[change_field].to_numpy()

    return left




def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    # External markets can be closed on a Taiwan trading day.
    # A short forward-fill is useful for dashboard continuity, while limiting
    # the fill prevents stale values from surviving long data gaps.
    external_cols = [
        "usdtwd",
        "dxy",
        "vix",
        "sp500",
        "nasdaq",
        "dow",
        "sox",
        "us10y",
    ]
    for col in external_cols:
        if col in df.columns:
            df[col] = df[col].ffill(limit=4)

    df["foreign_20d_sum_100m"] = (
        df["foreign_net_100m"]
        .rolling(20, min_periods=5)
        .sum()
    )

    # Explicit fill_method=None removes pandas' deprecated implicit padding
    # and ensures the 20-day FX change is based on actual prepared values.
    df["usdtwd_20d_change_pct"] = (
        df["usdtwd"].pct_change(20, fill_method=None) * 100
    )

    # Retained for backward compatibility.
    df["taiex_volume_billion"] = np.nan

    # maintenance_est is a public-data market proxy, not a brokerage account
    # maintenance ratio. Use its own historical percentile rather than broker
    # thresholds such as 130/145/170%.
    maintenance = pd.to_numeric(
        df["maintenance_est"],
        errors="coerce",
    )

    maintenance = maintenance.where(
        np.isfinite(maintenance)
        & (maintenance > 0)
    )

    df["maintenance_est"] = maintenance

    df["maintenance_percentile"] = (
        maintenance
        .expanding(min_periods=20)
        .rank(method="average", pct=True)
        .mul(100)
        .where(maintenance.notna())
    )

    df["market_temperature"] = np.nan
    df["outflow_pressure_score"] = np.nan

    df = add_rolling_correlations(df)
    return df



CORRELATION_WINDOWS = (20, 60, 120)

def add_rolling_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling Pearson correlations using daily returns/changes, not levels."""
    df = df.copy()

    for col in ["taiex", "sp500", "nasdaq", "dow", "sox", "usdtwd", "dxy", "vix", "us10y"]:
        if col in df.columns:
            ret = (
                pd.to_numeric(df[col], errors="coerce")
                .pct_change(fill_method=None)
            )

            # If an as-of aligned external observation is repeated because the
            # external market was closed, do not count the repeated value as a
            # genuine zero-return session.
            source_col = f"{col}_date"
            if source_col in df.columns and col != "taiex":
                source_dates = pd.to_datetime(
                    df[source_col],
                    errors="coerce",
                )
                repeated = source_dates.eq(source_dates.shift(1))
                ret = ret.mask(repeated)

            df[f"{col}_return"] = ret

    if "taiex_return" not in df.columns:
        return df

    targets = ["sp500", "nasdaq", "dow", "sox", "usdtwd", "dxy", "vix", "us10y"]

    for target in targets:
        ret_col = f"{target}_return"
        if ret_col not in df.columns:
            continue
        for window in CORRELATION_WINDOWS:
            df[f"corr_{window}d_taiex_{target}"] = (
                df["taiex_return"]
                .rolling(window, min_periods=max(10, window // 2))
                .corr(df[ret_col])
            )

    if "foreign_net_100m" in df.columns:
        foreign = pd.to_numeric(df["foreign_net_100m"], errors="coerce")
        for window in CORRELATION_WINDOWS:
            df[f"corr_{window}d_taiex_foreign"] = (
                df["taiex_return"]
                .rolling(window, min_periods=max(10, window // 2))
                .corr(foreign)
            )

    return df


def latest_correlations(row: pd.Series) -> dict:
    labels = {
        "sp500": "S&P 500",
        "nasdaq": "Nasdaq",
        "dow": "Dow",
        "sox": "SOX",
        "usdtwd": "USD/TWD",
        "dxy": "DXY",
        "vix": "VIX",
        "us10y": "US 10Y Treasury",
        "foreign": "Foreign flow",
    }

    result = {}
    for window in CORRELATION_WINDOWS:
        bucket = {}
        for key, label in labels.items():
            value = row.get(f"corr_{window}d_taiex_{key}")
            bucket[key] = {
                "label": label,
                "value": round(float(value), 3) if pd.notna(value) else None,
            }
        result[f"{window}d"] = bucket
    return result


def _pct_change_over(series: pd.Series, periods: int) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= periods:
        return None
    old = s.iloc[-periods - 1]
    new = s.iloc[-1]
    if old == 0 or pd.isna(old) or pd.isna(new):
        return None
    return float((new / old - 1) * 100)


def _signal(
    signal_id: str,
    title: str,
    message: str,
    level: str,
    category: str,
    evidence: dict,
) -> dict:
    return {
        "id": signal_id,
        "title": title,
        "message": message,
        "level": level,
        "category": category,
        "evidence": evidence,
    }


def detect_signals(
    df: pd.DataFrame,
    max_signals: int = 5,
    current: pd.Series | None = None,
    current_changes: dict | None = None,
) -> list[dict]:
    """Generate a small set of explainable, non-duplicative market signals.

    The goal is not to predict prices. It surfaces unusual conditions and
    divergences that are worth a user's attention.
    """
    if df.empty:
        return []

    x = df.sort_values("date").copy()
    last = current if current is not None else x.iloc[-1]
    candidates: list[tuple[int, dict]] = []

    taiex_20d = _pct_change_over(x["taiex"], 20) if "taiex" in x else None
    if current_changes is not None:
        sp500_20d = current_changes.get("sp500_20d_change_pct")
        sox_20d = current_changes.get("sox_20d_change_pct")
    else:
        sp500_20d = _pct_change_over(x["sp500"], 20) if "sp500" in x else None
        sox_20d = _pct_change_over(x["sox"], 20) if "sox" in x else None

    foreign20 = last.get("foreign_20d_sum_100m")
    vix = last.get("vix")
    maintenance_pct = last.get("maintenance_percentile")
    us10y = last.get("us10y")
    corr_foreign = last.get("corr_60d_taiex_foreign")
    corr_sox = last.get("corr_60d_taiex_sox")
    corr_sp500 = last.get("corr_60d_taiex_sp500")

    # 1) TAIEX vs foreign flow divergence
    if taiex_20d is not None and pd.notna(foreign20):
        if taiex_20d >= 3 and foreign20 < 0:
            candidates.append((
                100,
                _signal(
                    "taiex_up_foreign_sell",
                    "指數上漲，但外資仍站在賣方",
                    f"台股近20日約上漲 {taiex_20d:.1f}%，但外資20日累計仍為賣超。這屬於價格與資金流背離，漲勢需要留意是否缺乏外資確認。",
                    "warning",
                    "divergence",
                    {
                        "taiex_20d_change_pct": round(taiex_20d, 2),
                        "foreign_20d_sum_100m": round(float(foreign20), 2),
                    },
                )
            ))
        elif taiex_20d <= -3 and foreign20 > 0:
            candidates.append((
                95,
                _signal(
                    "taiex_down_foreign_buy",
                    "指數下跌，但外資正在承接",
                    f"台股近20日約下跌 {abs(taiex_20d):.1f}%，但外資20日累計仍為買超。價格弱、資金流卻改善，可能是值得觀察的反向背離。",
                    "watch",
                    "divergence",
                    {
                        "taiex_20d_change_pct": round(taiex_20d, 2),
                        "foreign_20d_sum_100m": round(float(foreign20), 2),
                    },
                )
            ))


    # 2) Volatility condition
    if pd.notna(vix):
        if vix >= 30:
            candidates.append((
                92,
                _signal(
                    "vix_extreme_high",
                    "市場恐慌程度偏高",
                    f"VIX 恐慌指數為 {float(vix):.2f}，"
                    "市場波動風險偏高，需留意急漲急跌。",
                    "risk",
                    "risk",
                    {"vix": round(float(vix), 2)},
                )
            ))
        elif vix >= 20:
            candidates.append((
                65,
                _signal(
                    "vix_elevated",
                    "市場恐慌情緒升溫",
                    f"VIX 恐慌指數為 {float(vix):.2f}，"
                    "市場波動風險有所升高。",
                    "warning",
                    "risk",
                    {"vix": round(float(vix), 2)},
                )
            ))

    # 3) Foreign flow
    # Retain actual foreign flow and price divergence.
    # Do not introduce arbitrary abnormal-flow thresholds.

    # 4) FX and dollar condition
    if current_changes is not None:
        usd20 = current_changes.get("usdtwd_20d_change_pct")
        dxy20 = current_changes.get("dxy_20d_change_pct")
    else:
        usd20 = _pct_change_over(x["usdtwd"], 20) if "usdtwd" in x else None
        dxy20 = _pct_change_over(x["dxy"], 20) if "dxy" in x else None

    if usd20 is not None and dxy20 is not None:
        if usd20 >= 2 and dxy20 >= 2:
            candidates.append((
                84,
                _signal(
                    "usd_fx_pressure",
                    "美元走強且台幣走弱",
                    f"近20筆有效觀測期間，美元指數上漲 {dxy20:.2f}%，"
                    f"美元兌台幣上漲 {usd20:.2f}%，"
                    "需留意外資資金動向。",
                    "warning",
                    "macro",
                    {
                        "usdtwd_20d_change_pct": round(usd20, 2),
                        "dxy_20d_change_pct": round(dxy20, 2),
                    },
                )
            ))

    # 5) US 10Y Treasury yield condition
    if pd.notna(us10y):
        if us10y >= 4.5:
            candidates.append((
                86,
                _signal(
                    "us10y_high",
                    "美國長天期利率偏高",
                    f"美國10年期公債殖利率為 {float(us10y):.2f}%，"
                    "較高的利率可能增加股票估值壓力。",
                    "warning",
                    "rates",
                    {"us10y": round(float(us10y), 3)},
                )
            ))

    # 6) Leverage condition
    if pd.notna(maintenance_pct):
        if maintenance_pct >= 90:
            candidates.append((
                70,
                _signal(
                    "maintenance_high",
                    "融資擔保狀態處在歷史偏高區",
                    f"市場融資擔保比估算位於歷史第 "
                    f"{float(maintenance_pct):.1f} 百分位，"
                    "目前相對歷史處於偏高水準。"
                    "此指標不等同券商帳戶維持率。",
                    "info",
                    "leverage",
                    {
                        "maintenance_percentile":
                            round(float(maintenance_pct), 2)
                    },
                )
            ))
        elif maintenance_pct <= 10:
            candidates.append((
                80,
                _signal(
                    "maintenance_low",
                    "融資擔保狀態處在歷史低檔",
                    f"市場融資擔保比估算位於歷史第 "
                    f"{float(maintenance_pct):.1f} 百分位，"
                    "市場槓桿安全墊相對歷史偏弱，"
                    "需要留意融資壓力。",
                    "risk",
                    "leverage",
                    {
                        "maintenance_percentile":
                            round(float(maintenance_pct), 2)
                    },
                )
            ))


    # 7) Taiwan vs US divergence
    if taiex_20d is not None and sp500_20d is not None:
        gap = taiex_20d - sp500_20d
        if abs(gap) >= 7:
            stronger = "台股" if gap > 0 else "S&P 500"
            candidates.append((
                72,
                _signal(
                    "taiex_sp500_divergence",
                    "台美股近一個月走勢明顯分化",
                    f"台股近20日 {taiex_20d:+.1f}%，S&P 500 {sp500_20d:+.1f}%，目前由{stronger}明顯領先。這表示跨市場同步性正在下降。",
                    "watch",
                    "cross_market",
                    {
                        "taiex_20d_change_pct": round(taiex_20d, 2),
                        "sp500_20d_change_pct": round(sp500_20d, 2),
                        "gap_pct_points": round(gap, 2),
                    },
                )
            ))

    # 8) Semiconductor divergence
    if taiex_20d is not None and sox_20d is not None:
        gap = taiex_20d - sox_20d
        if abs(gap) >= 8:
            candidates.append((
                74,
                _signal(
                    "taiex_sox_divergence",
                    "台股與半導體指數走勢出現分化",
                    f"台股近20日 {taiex_20d:+.1f}%，SOX {sox_20d:+.1f}%。台股科技權重高，兩者明顯分化時值得特別追蹤。",
                    "watch",
                    "cross_market",
                    {
                        "taiex_20d_change_pct": round(taiex_20d, 2),
                        "sox_20d_change_pct": round(sox_20d, 2),
                        "gap_pct_points": round(gap, 2),
                    },
                )
            ))

    # 9) Dependency summary: whichever currently matters most
    corr_candidates = []
    for key, label, value in [
        ("foreign", "外資流向", corr_foreign),
        ("sox", "SOX", corr_sox),
        ("sp500", "S&P 500", corr_sp500),
    ]:
        if pd.notna(value):
            corr_candidates.append((abs(float(value)), key, label, float(value)))

    if corr_candidates:
        _, key, label, value = max(corr_candidates)
        if abs(value) >= 0.55:
            direction = "同向" if value > 0 else "反向"
            candidates.append((
                66,
                _signal(
                    f"dominant_corr_{key}",
                    f"近期台股與{label}連動最明顯",
                    f"60日相關係數約 {value:.2f}，屬於較明顯的{direction}關係。這描述近期同步程度，不代表因果或未來一定維持。",
                    "info",
                    "relationship",
                    {"correlation_60d": round(value, 3)},
                )
            ))

    # De-duplicate by category when possible, then keep the strongest 3–5.
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    used_ids = set()

    for _, item in candidates:
        if item["id"] in used_ids:
            continue
        selected.append(item)
        used_ids.add(item["id"])
        if len(selected) >= max_signals:
            break

    if not selected:
        selected.append(
            _signal(
                "no_major_anomaly",
                "目前沒有明顯異常訊號",
                "主要風險、資金與跨市場指標目前沒有觸發明顯背離或極端條件。這不代表市場沒有風險，只表示量化條件暫時偏中性。",
                "neutral",
                "summary",
                {},
            )
        )

    return selected


def interpretation_guide() -> dict:
    """Beginner-facing explanations that the frontend can render as help text."""
    return {
        "indexed_100": {
            "title": "起點100怎麼看？",
            "what": "把每條線在目前選取區間的第一個有效值都設成100。",
            "how": "看誰從100上升得更多，就代表誰在同一期間漲得更快；低於100則代表相對下跌。",
            "caution": "適合比較相對走勢，不代表原始價格高低。",
        },
        "rolling_correlation": {
            "title": "相關係數怎麼看？",
            "what": "觀察台股和另一個指標最近是否常一起漲跌。",
            "how": "+1接近完全同向、0代表線性關係弱、-1接近完全反向。20日看短期，60日看中期，120日看較穩定關係。",
            "caution": "相關不代表因果，而且市場關係會隨時間改變。",
        },
        "foreign_flow": {
            "title": "外資流向怎麼看？",
            "what": "觀察外資單日與20日累計買賣超。",
            "how": "價格上漲且外資持續買超，趨勢通常較有資金確認；價格上漲但外資持續賣超，則形成背離。",
            "caution": "外資不是唯一影響市場的資金來源。",
        },
        "us10y": {
            "title": "美國10年期公債殖利率怎麼看？",
            "what": "它是全球資產定價最重要的長天期無風險利率參考之一。",
            "how": "殖利率上升通常代表折現率壓力增加，對高估值與成長股較不利；下降則通常減輕估值壓力。",
            "caution": "殖利率上升可能來自景氣強、通膨高或風險溢酬改變，不能單獨當成股市方向訊號。",
        },
        "vix": {
            "title": "VIX怎麼看？",
            "what": "反映美股選擇權市場對未來波動的定價。",
            "how": "指數升高時，通常表示市場預期波動加大，風險情緒升溫。",
            "caution": "VIX高不代表市場一定繼續跌，VIX低也不代表沒有風險。",
        },
    }

def backtests(df: pd.DataFrame, thresholds: list[float]):
    """Backtest downward crossings of historical percentiles."""

    out = []

    x = (
        df.sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    metric = pd.to_numeric(
        x["maintenance_percentile"],
        errors="coerce",
    )

    metric = metric.where(
        np.isfinite(metric)
        & metric.between(0, 100)
    )

    taiex = pd.to_numeric(
        x["taiex"],
        errors="coerce",
    )

    taiex = taiex.where(
        np.isfinite(taiex)
        & (taiex > 0)
    )

    previous = metric.shift(1)

    for threshold in thresholds:

        if not np.isfinite(threshold) or not (
            0 <= threshold <= 100
        ):
            raise ValueError(
                f"回測百分位門檻必須介於 0～100：{threshold}"
            )

        idx = x.index[
            metric.notna()
            & previous.notna()
            & (metric < threshold)
            & (previous >= threshold)
        ].tolist()

        horizons = []

        for days in (20, 60, 120):

            vals = [
                taiex.iloc[i + days] / taiex.iloc[i] - 1
                for i in idx
                if i + days < len(x)
                and pd.notna(taiex.iloc[i])
                and pd.notna(taiex.iloc[i + days])
            ]

            horizons.append({
                "trading_days": days,
                "samples": len(vals),
                "average_return_pct": (
                    round(float(np.mean(vals) * 100), 2)
                    if vals else None
                ),
                "median_return_pct": (
                    round(float(np.median(vals) * 100), 2)
                    if vals else None
                ),
                "win_rate_pct": (
                    round(
                        float(np.mean(np.array(vals) > 0) * 100),
                        1,
                    )
                    if vals else None
                ),
            })

        out.append({
            "threshold": threshold,
            "events": len(idx),
            "horizons": horizons,
        })

    return out


def commentary(row: pd.Series) -> str:
    parts = []

    mp = row.get("maintenance_percentile")
    foreign = row.get("foreign_20d_sum_100m")
    vix = row.get("vix")

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

    if pd.notna(foreign):
        parts.append(
            "外資近20日偏買方"
            if foreign > 0
            else "外資近20日偏賣方"
        )

    if pd.notna(vix):
        parts.append(
            "市場波動顯著升高"
            if vix >= 30
            else "波動風險偏高"
            if vix >= 20
            else "波動情緒相對平穩"
        )

    return (
        "；".join(parts)
        + "。此內容為量化資料摘要，不構成投資建議。"
    )
