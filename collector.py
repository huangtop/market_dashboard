#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from market_dashboard.analysis import add_global_daily_changes, align_global_asof, backtests, commentary, enrich, latest_correlations, detect_signals, interpretation_guide
from market_dashboard.config import Settings
from market_dashboard.sources import Sources
from market_dashboard.storage import Store
from market_dashboard.utils import safe_float


SCHEMA_VERSION = "4.8-global-latest"

# Step 3:
# 保留前端「區間起點 = 100」的動態比較概念，
# 後端新增 rolling Z-score，讓風險 / 資金 / 總經指標可以用同一尺度
# 判斷「相對自身歷史是否異常」。
# 舊版 flat latest / series 仍保留，WordPress 暫時不需要修改。
VIEW_FIELDS = {
    "taiwan": [
        "taiex",
        "taiex_change_pct",
        "tsmc",
        "tsmc_change_pct",
        "maintenance_est",
        "maintenance_percentile",
        "maintenance_zscore",
        "margin_balance_billion",
        "foreign_net_100m",
        "foreign_20d_sum_100m",
        "foreign_20d_zscore",
        "market_temperature",
    ],
    "us": [
        "sp500",
        "sp500_change_pct",
        "nasdaq",
        "nasdaq_change_pct",
        "dow",
        "dow_change_pct",
        "sox",
        "sox_change_pct",
        "vix",
        "vix_change_pct",
        "vix_zscore",
        "us10y",
        "us10y_change_bp",
        "us10y_zscore",
        "corr_20d_taiex_sp500",
        "corr_20d_taiex_nasdaq",
        "corr_20d_taiex_dow",
        "corr_20d_taiex_sox",
        "corr_20d_taiex_usdtwd",
        "corr_20d_taiex_dxy",
        "corr_20d_taiex_vix",
        "corr_20d_taiex_us10y",
        "corr_20d_taiex_foreign",
        "corr_60d_taiex_sp500",
        "corr_60d_taiex_nasdaq",
        "corr_60d_taiex_dow",
        "corr_60d_taiex_sox",
        "corr_60d_taiex_usdtwd",
        "corr_60d_taiex_dxy",
        "corr_60d_taiex_vix",
        "corr_60d_taiex_us10y",
        "corr_60d_taiex_foreign",
        "corr_120d_taiex_sp500",
        "corr_120d_taiex_nasdaq",
        "corr_120d_taiex_dow",
        "corr_120d_taiex_sox",
        "corr_120d_taiex_usdtwd",
        "corr_120d_taiex_dxy",
        "corr_120d_taiex_vix",
        "corr_120d_taiex_us10y",
        "corr_120d_taiex_foreign",
    ],
    "macro": [
        "usdtwd",
        "usdtwd_zscore",
        "dxy",
        "dxy_zscore",
        "us10y",
        "us10y_change_bp",
        "us10y_zscore",
        "outflow_pressure_score",
    ],
    # Cross-market 不是新的資料，而是把之後需要一起比較的欄位整理成同一視角。
    "cross_market": [
        "taiex",
        "tsmc",
        "sp500",
        "nasdaq",
        "sox",
        "vix",
        "vix_zscore",
        "usdtwd",
        "usdtwd_zscore",
        "dxy",
        "dxy_zscore",
        "us10y",
        "us10y_zscore",
        "foreign_20d_sum_100m",
        "foreign_20d_zscore",
        "maintenance_zscore",
        "outflow_pressure_score",
        "corr_60d_taiex_sp500",
        "corr_60d_taiex_nasdaq",
        "corr_60d_taiex_dow",
        "corr_60d_taiex_sox",
        "corr_60d_taiex_usdtwd",
        "corr_60d_taiex_dxy",
        "corr_60d_taiex_vix",
        "corr_60d_taiex_us10y",
        "corr_60d_taiex_foreign",
    ],
}


def setup_log() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/collector.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--years", type=int)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--latest", action="store_true", help="只更新最新交易日（預設）")
    mode.add_argument("--backfill", action="store_true", help="只補 SQLite 缺少或不完整的日期")
    mode.add_argument("--full", action="store_true", help="強制重新抓取完整期間")
    mode.add_argument("--derive-only", action="store_true", help="不連網，只讀 SQLite 重算 JSON/CSV")
    mode.add_argument(
        "--taiwan-latest",
        action="store_true",
        help="只更新台股：TAIEX、2330、外資、融資全部直連 TWSE；不連 Yahoo",
    )
    mode.add_argument(
        "--global-latest",
        action="store_true",
        help="只更新全球市場：單次 Yahoo bulk request；不連 TWSE",
    )
    mode.add_argument(
        "--refresh-global-only",
        action="store_true",
        help="只用一次 Yahoo bulk request 重建 global_daily；不連 TWSE",
    )
    ap.add_argument("--refresh-days", type=int, default=None, help=argparse.SUPPRESS)
    return ap.parse_args()


def select_days(
    args: argparse.Namespace,
    all_days: list[pd.Timestamp],
    store: Store,
) -> tuple[list[pd.Timestamp], str]:
    if not all_days:
        return [], "沒有交易日"

    if args.full:
        return all_days, "完整重抓"

    if args.backfill:
        complete = store.complete_dates()
        missing = [d for d in all_days if d.strftime("%Y-%m-%d") not in complete]
        return missing, "增量回補"

    return [max(all_days)], "最新交易日"


def previous_valid_margin(store: Store, day: pd.Timestamp) -> dict:
    """Return the most recent valid margin values before *day*.

    This is only a display-safety fallback for a transient TWSE outage.
    It prevents a missing value from being serialized as null and later shown
    by the UI as 0%. The log clearly marks the value as carried forward.
    """
    df = store.frame()
    if df.empty:
        return {}

    prior = df[
        (df["date"] < pd.Timestamp(day))
        & df["margin_balance_billion"].notna()
        & df["maintenance_est"].notna()
    ]
    if prior.empty:
        return {}

    last = prior.iloc[-1]
    return {
        "margin_balance_billion": safe_float(last["margin_balance_billion"]),
        "maintenance_est": safe_float(last["maintenance_est"]),
    }


def select_fields(row: dict, fields: list[str]) -> dict:
    """只留下某個 market view 需要的欄位。

    欄位不存在時仍回傳 None，讓 JSON schema 穩定，
    後續新增美股欄位時前端不需要重新猜測欄位是否存在。
    """
    return {field: row.get(field) for field in fields}


def build_market_views(records: list[dict], latest: dict) -> dict:
    """建立分市場視角，同時維持舊版 flat payload 相容性。"""
    views = {}

    for view_name, fields in VIEW_FIELDS.items():
        views[view_name] = {
            "fields": fields,
            "latest": select_fields(latest, fields),
            "series": [
                {
                    "date": row["date"],
                    **select_fields(row, fields),
                }
                for row in records
            ],
        }

    return views


def build_global_latest(global_df: pd.DataFrame) -> dict:
    """Return the newest completed raw global session, independent of Taiwan date.

    `latest` remains the Taiwan-row strict-prior-session view used for historical
    comparison. `global_latest` is for "what is the latest completed US/global
    market session now?" KPIs and therefore must come directly from global_daily.
    """
    if global_df is None or global_df.empty:
        return {}

    work = add_global_daily_changes(global_df.copy()).sort_values("date")
    base_fields = ["usdtwd", "dxy", "vix", "sp500", "nasdaq", "dow", "sox", "us10y"]

    # Prefer the newest row that contains the core US market set. Yahoo symbols
    # can occasionally have isolated gaps, so per-field dates below still make
    # provenance explicit.
    core = ["sp500", "nasdaq", "dow", "sox", "vix", "us10y"]
    available_sessions = work.dropna(subset=core, how="all")
    if available_sessions.empty:
        return {}

    latest_date = pd.Timestamp(available_sessions.iloc[-1]["date"])
    result = {"date": latest_date.strftime("%Y-%m-%d")}

    change_fields = {
        "vix": "vix_change_pct",
        "sp500": "sp500_change_pct",
        "nasdaq": "nasdaq_change_pct",
        "dow": "dow_change_pct",
        "sox": "sox_change_pct",
        "us10y": "us10y_change_bp",
    }

    # Use each field's newest available completed session and carry its own date.
    # This avoids pretending that FX/DXY necessarily share the exact same session.
    for field in base_fields:
        change_field = change_fields.get(field)

        if change_field:
            valid = work.dropna(subset=[field, change_field])
        else:
            valid = work.dropna(subset=[field])

        if valid.empty:
            result[field] = None
            result[f"{field}_date"] = None
            if change_field:
                result[change_field] = None
            continue

        row = valid.iloc[-1]

        result[field] = safe_float(row[field])
        result[f"{field}_date"] = pd.Timestamp(
            row["date"]
        ).strftime("%Y-%m-%d")

        if change_field:
            result[change_field] = safe_float(row[change_field])

    return result


def main() -> None:
    args = parse_args()
    setup_log()

    cfg = Settings.load(args.config)
    if args.years:
        cfg.years = args.years

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = Store(cfg.database)

    end = date.today()
    historical_mode = args.backfill or args.full
    interrupted = False

    if args.derive_only:
        logging.info("模式：derive-only；不連線 TWSE / Yahoo Finance")

    elif args.global_latest:
        logging.info("模式：global-latest；只更新全球市場，不連 TWSE")
        src = Sources(cfg)
        external_start = end - timedelta(days=14)
        external = src.external(external_start, end)
        if external.empty:
            raise RuntimeError("Yahoo Finance 近期全球市場資料為空；保留原 global_daily")
        stored_external_rows = store.update_external(external)
        logging.info("global_daily 增量 UPSERT：%d 筆近期全球交易日", stored_external_rows)

    elif args.taiwan_latest:
        logging.info("模式：taiwan-latest；台股資料一律直連 TWSE，不連 Yahoo Finance")
        src = Sources(cfg)
        download_start = end - timedelta(days=14)

        taiex_df = src.taiex(download_start, end)
        tsmc_df = src.tsmc(download_start, end)
        core = (
            taiex_df
            .merge(tsmc_df, on="date", how="outer")
            .sort_values("date")
        )
        all_days = list(core["date"].dropna().drop_duplicates().sort_values())
        if not all_days:
            raise RuntimeError("TWSE 最近期間沒有可用交易日")

        day = max(all_days)
        key = day.strftime("%Y-%m-%d")
        row = {
            "date": key,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }

        core_row = core[core["date"].eq(day)]
        if not core_row.empty:
            last_core = core_row.iloc[-1]
            row["taiex"] = safe_float(last_core.get("taiex"))
            row["taiex_change_pct"] = safe_float(last_core.get("taiex_change_pct"))
            row["tsmc"] = safe_float(last_core.get("tsmc"))
            row["tsmc_change_pct"] = safe_float(last_core.get("tsmc_change_pct"))

        try:
            row.update({k: safe_float(v) for k, v in src.margin(day).items()})
        except Exception as exc:
            logging.warning("%s margin: %s", key, exc)
            fallback = previous_valid_margin(store, day)
            if fallback:
                row.update(fallback)
                logging.warning("%s margin 使用前一有效交易日數值暫代", key)

        try:
            row["foreign_net_100m"] = safe_float(src.foreign(day))
        except Exception as exc:
            logging.warning("%s foreign: %s", key, exc)

        store.upsert(row)
        logging.info(
            "%s TWSE saved：TAIEX=%s change=%s%%；2330=%s change=%s%%；foreign=%s 億",
            key,
            row.get("taiex"),
            row.get("taiex_change_pct"),
            row.get("tsmc"),
            row.get("tsmc_change_pct"),
            row.get("foreign_net_100m"),
        )

    elif args.refresh_global_only:
        logging.info("模式：refresh-global-only；不連 TWSE，只執行一次 Yahoo bulk download")
        src = Sources(cfg)
        external_start = end - timedelta(days=365 * cfg.years + 45)
        logging.info(
            "重建全球市場交易日曆 %s 至 %s（USD/TWD, DXY, VIX, S&P500, Nasdaq, Dow, SOX, US10Y）",
            external_start,
            end,
        )
        external = src.external(external_start, end)

        if external.empty:
            raise RuntimeError("Yahoo Finance 全球市場資料為空；保留原 global_daily，不做覆蓋")

        stored_external_rows = store.replace_global(external)
        logging.info(
            "global_daily 已用原始 Yahoo bulk 資料完整重建：%d 筆全球交易日",
            stored_external_rows,
        )

    else:
        src = Sources(cfg)
        download_start = (
            end - timedelta(days=365 * cfg.years + 45)
            if historical_mode
            else end - timedelta(days=14)
        )

        logging.info("下載 %s 至 %s 的核心行情", download_start, end)
        core = (
            src.taiex(download_start, end)
            .merge(src.tsmc(download_start, end), on="date", how="outer")
            .sort_values("date")
        )

        # 全球市場更新策略：
        # - --latest：只抓最近 10 個日曆日，UPSERT 進 global_daily。
        #   這足以跨過一般週末與連假，也避免每天重抓五年 Yahoo 歷史。
        # - --backfill / --full：才抓完整設定期間，並重建 global_daily。
        if historical_mode:
            external_start = end - timedelta(days=365 * cfg.years + 45)
            logging.info(
                "歷史模式：下載 %s 至 %s 的完整全球市場行情（USD/TWD, DXY, VIX, S&P500, Nasdaq, Dow, SOX, US10Y）",
                external_start,
                end,
            )
            external = src.external(external_start, end)
            if external.empty:
                raise RuntimeError("Yahoo Finance 全球市場資料為空；保留原 global_daily，不做覆蓋")
            stored_external_rows = store.replace_global(external)
            logging.info(
                "global_daily 已完整重建：%d 筆全球交易日",
                stored_external_rows,
            )
        else:
            external_start = end - timedelta(days=14)
            logging.info(
                "增量模式：只下載 %s 至 %s 的近期全球市場行情（單次 Yahoo bulk request）",
                external_start,
                end,
            )
            external = src.external(external_start, end)
            if external.empty:
                logging.warning("Yahoo Finance 近期全球市場資料為空；沿用既有 global_daily")
                stored_external_rows = 0
            else:
                stored_external_rows = store.update_external(external)
            logging.info(
                "global_daily 增量 UPSERT：%d 筆近期全球交易日",
                stored_external_rows,
            )

        all_days = list(core["date"].dropna().drop_duplicates().sort_values())
        days, mode_name = select_days(args, all_days, store)
        logging.info("本次待抓 %d 個交易日（模式：%s）", len(days), mode_name)

        try:
            for i, day in enumerate(days, 1):
                key = day.strftime("%Y-%m-%d")
                row = {
                    "date": key,
                    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                }

                core_row = core[core["date"].eq(day)]
                if not core_row.empty:
                    last_core = core_row.iloc[-1]
                    row["taiex"] = safe_float(last_core.get("taiex"))
                    row["taiex_change_pct"] = safe_float(last_core.get("taiex_change_pct"))
                    row["tsmc"] = safe_float(last_core.get("tsmc"))
                    row["tsmc_change_pct"] = safe_float(last_core.get("tsmc_change_pct"))

                try:
                    row.update({k: safe_float(v) for k, v in src.margin(day).items()})
                except Exception as exc:
                    logging.warning("%s margin: %s", key, exc)
                    if not historical_mode:
                        fallback = previous_valid_margin(store, day)
                        if fallback:
                            row.update(fallback)
                            logging.warning(
                                "%s margin 使用前一有效交易日數值暫代：maintenance_est=%s, margin_balance_billion=%s",
                                key,
                                fallback["maintenance_est"],
                                fallback["margin_balance_billion"],
                            )

                try:
                    row["foreign_net_100m"] = safe_float(src.foreign(day))
                except Exception as exc:
                    logging.warning("%s foreign: %s", key, exc)

                store.upsert(row)
                logging.info("%d/%d %s saved", i, len(days), key)
                time.sleep(cfg.request_delay_seconds)

        except KeyboardInterrupt:
            interrupted = True
            logging.warning("收到 Ctrl+C，停止下載並輸出目前已保存的資料")

    df = store.frame()
    if df.empty:
        raise RuntimeError("SQLite 沒有可輸出的資料")

    global_df = store.global_frame()
    if global_df.empty:
        logging.warning("global_daily 沒有資料；全球市場欄位將維持缺值")
    else:
        logging.info(
            "使用 global_daily 做 completed-session as-of 對齊：%d 筆全球交易日",
            len(global_df),
        )

    # 4.8: raw newest global session for current US/global KPIs.
    # This is intentionally independent of Taiwan latest_date.
    global_latest = build_global_latest(global_df)

    df = align_global_asof(
        df.sort_values("date"),
        global_df,
    )
    df = enrich(df)
    recent_start = end - timedelta(days=365 * cfg.years + 5)
    recent = df[df["date"].dt.date >= recent_start].copy()
    if recent.empty:
        raise RuntimeError("沒有可輸出的資料")

    cols = [
        "taiex",
        "taiex_change_pct",
        "tsmc",
        "tsmc_change_pct",
        "maintenance_est",
        "margin_balance_billion",
        "foreign_net_100m",
        "foreign_20d_sum_100m",
        "usdtwd",
        "dxy",
        "vix",
        "vix_change_pct",
        "sp500",
        "sp500_change_pct",
        "nasdaq",
        "nasdaq_change_pct",
        "dow",
        "dow_change_pct",
        "sox",
        "sox_change_pct",
        "outflow_pressure_score",
        "market_temperature",
        "maintenance_percentile",
        "maintenance_zscore",
        "foreign_20d_zscore",
        "usdtwd_zscore",
        "dxy_zscore",
        "vix_zscore",
        "us10y",
        "us10y_change_bp",
        "us10y_zscore",
        "usdtwd_date",
        "dxy_date",
        "vix_date",
        "sp500_date",
        "nasdaq_date",
        "dow_date",
        "sox_date",
        "us10y_date",
    ]

    date_cols = {col for col in cols if col.endswith("_date")}
    records = []
    for _, row in recent.iterrows():
        item = {"date": row.date.strftime("%Y-%m-%d")}
        for col in cols:
            value = row.get(col)
            if col in date_cols:
                item[col] = (
                    pd.Timestamp(value).strftime("%Y-%m-%d")
                    if pd.notna(value)
                    else None
                )
            else:
                item[col] = safe_float(value)
        records.append(item)

    valid = recent.dropna(subset=["taiex"])
    if valid.empty:
        raise RuntimeError("沒有有效的 TAIEX 資料")
    last = valid.iloc[-1]

    latest = {}
    for col in cols:
        value = last.get(col)
        if col.endswith("_date"):
            latest[col] = (
                pd.Timestamp(value).strftime("%Y-%m-%d")
                if pd.notna(value)
                else None
            )
        else:
            latest[col] = safe_float(value)

    latest_source_dates = {
        field: latest.get(f"{field}_date")
        for field in [
            "usdtwd",
            "dxy",
            "vix",
            "sp500",
            "nasdaq",
            "dow",
            "sox",
            "us10y",
        ]
    }

    market_views = build_market_views(records, latest)

    payload = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "timezone": "Asia/Taipei",
            "market_views": list(VIEW_FIELDS.keys()),
            "transforms": {
                "indexed_100": {
                    "location": "frontend",
                    "description": "依使用者目前選取區間的第一個有效值設為100；適合比較不同價格指數的相對走勢。",
                },
                "zscore": {
                    "location": "backend",
                    "window_trading_days": 252,
                    "minimum_observations": 60,
                    "description": "衡量目前數值相對自身近一年歷史的異常程度；0約等於近期平均，+2代表明顯偏高，-2代表明顯偏低。",
                },
                "rolling_correlation": {
                    "location": "backend",
                    "windows_trading_days": [20, 60, 120],
                    "method": "Pearson correlation on daily returns/changes",
                    "description": "衡量台股日報酬與海外市場、匯率、VIX、外資流向的近期連動。",
                },
                "global_alignment": {
                    "method": "strict prior-session as-of",
                    "allow_exact_calendar_date": False,
                    "description": "台灣交易日只對齊當時已完成的最近全球市場日資料，避免使用同日尚未收盤的美股、美債或波動率。",
                },
                "taiwan_source_policy": {
                    "provider": "TWSE official",
                    "taiex": "indicesReport/MI_5MINS_HIST",
                    "tsmc": "exchangeReport/STOCK_DAY",
                    "foreign_flow": "TWSE",
                    "margin": "TWSE",
                    "description": "台股資料一律使用臺灣證券交易所官方資料；taiwan-latest 不連 Yahoo Finance。"
                },
                "global_latest": {
                    "source": "raw global_daily newest completed session",
                    "description": "提供美股/全球市場即時 KPI；不受台灣 latest_date 的 strict prior-session 對齊限制。歷史台美比較仍使用 latest/series 的 strict prior-session 資料。",
                },
                "global_update": {
                    "latest_mode": "10-calendar-day incremental Yahoo bulk download + SQLite UPSERT",
                    "historical_mode": "full configured-period rebuild",
                    "description": "平日只補近期全球交易日；只有 backfill/full/refresh-global-only 才處理完整歷史。",
                },
            },
            "maintenance_ratio_note": "市場融資擔保比估算＝可配對之上市融資股票市值 ÷ 融資金額；為市場層級 proxy，非券商整戶融資維持率。",
            "disclaimer": "資料僅供研究與資訊用途，不構成投資建議。",
        },
        "latest_date": last.date.strftime("%Y-%m-%d"),
        "latest_source_dates": latest_source_dates,

        # 4.8：全球市場自己的最新 completed session，供美股/全球 KPI 使用。
        # 不改動 latest/series 的 strict prior-session 歷史對齊語意。
        "global_latest": global_latest,

        # 新版結構：之後 WordPress 會改讀這裡。
        "markets": market_views,

        # 舊版結構：暫時保留，確保現有 WordPress 不壞。
        "latest": latest,
        "summary": commentary(last),
        "backtests": backtests(recent, cfg.maintenance_thresholds),
        "analytics": {
            "rolling_correlations": latest_correlations(last),
            "signals": detect_signals(recent, max_signals=5),
        },
        "education": interpretation_guide(),
        "series": records,
    }

    json_path = out / "market-dashboard.json"
    csv_path = out / "market-dashboard.csv"

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    recent.to_csv(csv_path, index=False, encoding="utf-8-sig")

    logging.info(
        "%s：%s",
        "中斷後已輸出" if interrupted else "完成",
        json_path,
    )
    logging.info(
        "JSON schema=%s；market views=%s",
        SCHEMA_VERSION,
        ", ".join(VIEW_FIELDS.keys()),
    )


if __name__ == "__main__":
    main()