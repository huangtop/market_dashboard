#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from market_dashboard.analysis import backtests, commentary, enrich
from market_dashboard.config import Settings
from market_dashboard.sources import Sources
from market_dashboard.storage import Store
from market_dashboard.utils import safe_float


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


def main() -> None:
    args = parse_args()
    setup_log()

    cfg = Settings.load(args.config)
    if args.years:
        cfg.years = args.years

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = Store(cfg.database)
    src = Sources(cfg)

    end = date.today()
    historical_mode = args.backfill or args.full
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
    external = src.external(download_start, end)

    all_days = list(core["date"].dropna().drop_duplicates().sort_values())
    days, mode_name = select_days(args, all_days, store)
    logging.info("本次待抓 %d 個交易日（模式：%s）", len(days), mode_name)

    interrupted = False
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
                row["tsmc"] = safe_float(last_core.get("tsmc"))

            try:
                row.update({k: safe_float(v) for k, v in src.margin(day).items()})
            except Exception as exc:
                logging.warning("%s margin: %s", key, exc)

                # Latest-day runs should not publish null margin values because
                # the current dashboard UI can render null as 0%. Carry forward
                # the last known valid values instead, and make it explicit in logs.
                if not historical_mode:
                    fallback = previous_valid_margin(store, day)
                    if fallback:
                        row.update(fallback)
                        logging.warning(
                            "%s margin 使用前一有效交易日數值暫代：maintenance_est=%s, "
                            "margin_balance_billion=%s",
                            key,
                            fallback["maintenance_est"],
                            fallback["margin_balance_billion"],
                        )
                    else:
                        logging.warning("%s margin 無可用歷史值，維持缺值", key)

            try:
                row["foreign_net_100m"] = safe_float(src.foreign(day))
            except Exception as exc:
                logging.warning("%s foreign: %s", key, exc)

            ext = external[external["date"].eq(day)]
            if not ext.empty:
                for col in ["usdtwd", "dxy", "vix"]:
                    row[col] = safe_float(ext.iloc[-1].get(col))

            store.upsert(row)
            logging.info("%d/%d %s saved", i, len(days), key)
            time.sleep(cfg.request_delay_seconds)
    except KeyboardInterrupt:
        interrupted = True
        logging.warning("收到 Ctrl+C，停止下載並輸出目前已保存的資料")

    df = store.frame()
    if df.empty:
        raise RuntimeError("SQLite 沒有可輸出的資料")

    df = enrich(df.sort_values("date"))
    recent_start = end - timedelta(days=365 * cfg.years + 5)
    recent = df[df["date"].dt.date >= recent_start].copy()
    if recent.empty:
        raise RuntimeError("沒有可輸出的資料")

    cols = [
        "taiex",
        "tsmc",
        "maintenance_est",
        "margin_balance_billion",
        "foreign_net_100m",
        "foreign_20d_sum_100m",
        "usdtwd",
        "dxy",
        "vix",
        "outflow_pressure_score",
        "market_temperature",
        "maintenance_percentile",
    ]
    records = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            **{col: safe_float(row.get(col)) for col in cols},
        }
        for _, row in recent.iterrows()
    ]

    valid = recent.dropna(subset=["taiex"])
    if valid.empty:
        raise RuntimeError("沒有有效的 TAIEX 資料")
    last = valid.iloc[-1]

    payload = {
        "meta": {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "timezone": "Asia/Taipei",
            "maintenance_ratio_note": "市場融資擔保比估算＝可配對之上市融資股票市值 ÷ 融資金額；為市場層級 proxy，非券商整戶融資維持率。",
            "disclaimer": "資料僅供研究與資訊用途，不構成投資建議。",
        },
        "latest_date": last.date.strftime("%Y-%m-%d"),
        "latest": {col: safe_float(last.get(col)) for col in cols},
        "summary": commentary(last),
        "backtests": backtests(recent, cfg.maintenance_thresholds),
        "series": records,
    }

    (out / "market-dashboard.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    recent.to_csv(out / "market-dashboard.csv", index=False, encoding="utf-8-sig")
    logging.info(
        "%s：%s",
        "中斷後已輸出" if interrupted else "完成",
        out / "market-dashboard.json",
    )


if __name__ == "__main__":
    main()
