from __future__ import annotations

from datetime import date, timedelta
import logging
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .config import Settings
from .utils import get_json, find_table, month_starts, number, roc_date, tables


# Yahoo Finance symbols used by the dashboard.
# Step 2 adds broad US equity indexes and the semiconductor index.
GLOBAL_TICKERS = {
    "usdtwd": "TWD=X",
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "sox": "^SOX",
    "us10y": "^TNX",
}


class Sources:
    def __init__(self, settings: Settings):
        self.cfg = settings
        self.session = requests.Session()

    def _monthly_core(self, start: date, end: date, *, kind: str) -> pd.DataFrame:
        chunks = []
        failed_months = []

        for m in month_starts(start, end):
            try:
                if kind == "taiex":
                    payload = get_json(
                        self.session,
                        "TAIEX/MI_5MINS_HIST",
                        {"date": m.strftime("%Y%m01"), "response": "json"},
                        self.cfg.request_timeout_seconds,
                        self.cfg.max_retries,
                    )
                    data = find_table(payload, ["日期", "收盤指數"])
                    date_col = next(c for c in data if "日期" in str(c))
                    value_col = next(c for c in data if "收盤指數" in str(c))
                    chunk = pd.DataFrame(
                        {
                            "date": data[date_col].map(roc_date),
                            "taiex": data[value_col].map(number),
                        }
                    )
                else:
                    payload = get_json(
                        self.session,
                        "afterTrading/STOCK_DAY",
                        {
                            "date": m.strftime("%Y%m01"),
                            "stockNo": "2330",
                            "response": "json",
                        },
                        self.cfg.request_timeout_seconds,
                        self.cfg.max_retries,
                    )
                    data = find_table(payload, ["日期", "收盤價"])
                    date_col = next(c for c in data if "日期" in str(c))
                    value_col = next(c for c in data if "收盤價" in str(c))
                    chunk = pd.DataFrame(
                        {
                            "date": data[date_col].map(roc_date),
                            "tsmc": data[value_col].map(number),
                        }
                    )

                chunk = chunk.dropna().drop_duplicates("date")
                if not chunk.empty:
                    chunks.append(chunk)

            except Exception as exc:
                failed_months.append(m.strftime("%Y-%m"))
                logging.warning(
                    "%s 月資料 %s 失敗：%s",
                    kind.upper(),
                    m.strftime("%Y-%m"),
                    exc,
                )
            finally:
                time.sleep(self.cfg.request_delay_seconds)

        if not chunks:
            raise RuntimeError(f"{kind.upper()} 所有月份下載皆失敗")

        if failed_months:
            logging.warning(
                "%s 本輪有 %d 個月份失敗，將由後續 backfill 再補：%s",
                kind.upper(),
                len(failed_months),
                ", ".join(failed_months),
            )

        return (
            pd.concat(chunks, ignore_index=True)
            .dropna()
            .drop_duplicates("date")
        )

    def taiex(self, start: date, end: date) -> pd.DataFrame:
        return self._monthly_core(start, end, kind="taiex")

    def tsmc(self, start: date, end: date) -> pd.DataFrame:
        return self._monthly_core(start, end, kind="tsmc")

    def close_prices(self, day: pd.Timestamp) -> pd.DataFrame:
        payload = get_json(
            self.session,
            "afterTrading/MI_INDEX",
            {
                "date": day.strftime("%Y%m%d"),
                "type": "ALLBUT0999",
                "response": "json",
            },
            self.cfg.request_timeout_seconds,
            self.cfg.max_retries,
        )
        data = find_table(payload, ["證券代號", "收盤價"])
        code_col = next(x for x in data if "證券代號" in str(x))
        close_col = next(x for x in data if "收盤價" in str(x))

        return pd.DataFrame(
            {
                "code": data[code_col].astype(str).str.strip(),
                "close": data[close_col].map(number),
            }
        ).dropna()

    def margin(self, day: pd.Timestamp) -> dict:
        """Estimate market-wide maintenance ratio from TWSE margin data.

        TWSE may return duplicated column labels after merged headers are
        flattened. This parser therefore selects columns by position when
        necessary.
        """
        common = {"date": day.strftime("%Y%m%d"), "response": "json"}

        detail_payload = get_json(
            self.session,
            "marginTrading/MI_MARGN",
            {**common, "selectType": "STOCK"},
            self.cfg.request_timeout_seconds,
            self.cfg.max_retries,
        )
        summary_payload = get_json(
            self.session,
            "marginTrading/MI_MARGN",
            {**common, "selectType": "MS"},
            self.cfg.request_timeout_seconds,
            self.cfg.max_retries,
        )

        def norm(value) -> str:
            return str(value).replace(" ", "").replace("\n", "")

        detail_rows = None
        code_idx = None
        balance_idx = None
        available_detail_fields = []

        for fields, rows, title in tables(detail_payload):
            names = [norm(x) for x in fields]
            available_detail_fields.append({"title": title, "fields": names})

            candidate_code = next(
                (i for i, x in enumerate(names) if "代號" in x),
                None,
            )
            candidate_balance = next(
                (
                    i
                    for i, x in enumerate(names)
                    if "融資" in x and "今日餘額" in x
                ),
                None,
            )

            if (
                candidate_balance is None
                and candidate_code is not None
                and len(names) >= 7
            ):
                candidate_balance = 6

            if (
                candidate_code is not None
                and candidate_balance is not None
                and rows
            ):
                detail_rows = rows
                code_idx = candidate_code
                balance_idx = candidate_balance
                break

        if detail_rows is None or code_idx is None or balance_idx is None:
            raise KeyError(
                f"找不到個股融資明細，TWSE 欄位={available_detail_fields}"
            )

        parsed_shares = []
        for row in detail_rows:
            if (
                not isinstance(row, (list, tuple))
                or max(code_idx, balance_idx) >= len(row)
            ):
                continue

            code = str(row[code_idx]).strip()
            lots = number(row[balance_idx])

            if (
                code.isdigit()
                and 4 <= len(code) <= 6
                and np.isfinite(lots)
            ):
                parsed_shares.append((code, lots))

        shares = pd.DataFrame(parsed_shares, columns=["code", "lots"])
        if shares.empty:
            raise ValueError("個股融資明細解析後為空")

        shares = shares.groupby("code", as_index=False)["lots"].sum()

        financing_twd = None
        available_summary_fields = []

        for fields, rows, title in tables(summary_payload):
            names = [norm(x) for x in fields]
            available_summary_fields.append({"title": title, "fields": names})

            item_idx = next(
                (i for i, x in enumerate(names) if "項目" in x),
                None,
            )
            today_candidates = [
                i for i, x in enumerate(names) if "今日餘額" in x
            ]

            if item_idx is None or not today_candidates:
                continue

            today_idx = today_candidates[0]

            for row in rows:
                if (
                    not isinstance(row, (list, tuple))
                    or max(item_idx, today_idx) >= len(row)
                ):
                    continue

                if "融資金額" not in norm(row[item_idx]):
                    continue

                value = number(row[today_idx])
                if np.isfinite(value) and value > 0:
                    financing_twd = value * 1000
                    break

            if financing_twd is not None:
                break

        if financing_twd is None:
            raise KeyError(
                f"找不到融資金額，TWSE 欄位={available_summary_fields}"
            )

        prices = self.close_prices(day)
        merged = (
            shares.merge(prices, on="code", how="inner")
            .dropna(subset=["lots", "close"])
        )

        if merged.empty:
            raise ValueError("融資明細與當日收盤價無法配對")

        collateral_twd = float(
            (merged["lots"] * 1000 * merged["close"]).sum()
        )

        if collateral_twd <= 0:
            raise ValueError("估算擔保品市值不是正數")

        return {
            "margin_balance_billion": financing_twd / 1e9,
            "maintenance_est": collateral_twd / financing_twd * 100,
        }

    def foreign(self, day: pd.Timestamp) -> float:
        payload = get_json(
            self.session,
            "fund/BFI82U",
            {
                "dayDate": day.strftime("%Y%m%d"),
                "type": "day",
                "response": "json",
            },
            self.cfg.request_timeout_seconds,
            self.cfg.max_retries,
        )

        data = find_table(payload, ["單位名稱", "買賣差額"])
        unit_col = next(c for c in data if "單位名稱" in str(c))
        net_col = next(c for c in data if "買賣差額" in str(c))

        rows = data[
            data[unit_col].astype(str).str.contains("外資及陸資", na=False)
        ]

        return (
            float(rows[net_col].map(number).sum() / 1e8)
            if not rows.empty
            else np.nan
        )

    def external(self, start: date, end: date) -> pd.DataFrame:
        """Download global-market data from Yahoo Finance.

        The output uses stable internal column names so the rest of the
        dashboard never needs to know Yahoo ticker symbols.
        """
        tickers = list(GLOBAL_TICKERS.values())

        raw = yf.download(
            tickers,
            start=start.isoformat(),
            end=(end + timedelta(days=2)).isoformat(),
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        columns = ["date", *GLOBAL_TICKERS.keys()]

        if raw.empty:
            logging.warning("Yahoo Finance 外部市場資料為空")
            return pd.DataFrame(columns=columns)

        if isinstance(raw.columns, pd.MultiIndex):
            try:
                close = raw["Close"]
            except KeyError:
                logging.warning("Yahoo Finance 回傳資料缺少 Close 欄位")
                return pd.DataFrame(columns=columns)
        else:
            close = raw

        out = pd.DataFrame(index=close.index)

        for internal_name, ticker in GLOBAL_TICKERS.items():
            if ticker in close.columns:
                out[internal_name] = close[ticker]
            elif len(tickers) == 1 and "Close" in close.columns:
                out[internal_name] = close["Close"]
            else:
                out[internal_name] = np.nan
                logging.warning(
                    "Yahoo Finance 缺少 %s (%s)",
                    internal_name,
                    ticker,
                )

        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out.reset_index().rename(
            columns={"Date": "date", "index": "date"}
        )

        out["date"] = pd.to_datetime(out["date"]).dt.normalize()

        return out[columns].sort_values("date").drop_duplicates("date")
