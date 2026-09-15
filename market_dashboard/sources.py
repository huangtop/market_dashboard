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

    def _twse_report_json(self, path: str, params: dict) -> dict:
        """Fetch an official TWSE report endpoint outside the /rwd/zh tree.

        TWSE's current historical TAIEX and STOCK_DAY reports are exposed under
        /indicesReport and /exchangeReport.  A HTTP 200 that redirects to
        page-not-found.html is treated as failure instead of being parsed.
        """
        bases = (
            "https://www.twse.com.tw",
            "https://wwwc.twse.com.tw",
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 Chrome/126 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.twse.com.tw/",
        }
        last = None

        for attempt in range(1, max(1, self.cfg.max_retries) + 1):
            for base in bases:
                try:
                    r = self.session.get(
                        f"{base}/{path}",
                        params=params,
                        headers=headers,
                        timeout=self.cfg.request_timeout_seconds,
                        allow_redirects=True,
                    )
                    r.raise_for_status()

                    if "page-not-found" in r.url:
                        raise RuntimeError(f"TWSE endpoint redirected to 404 page: {r.url}")

                    text = r.text.strip()
                    ctype = (r.headers.get("content-type") or "").lower()
                    if not text:
                        raise RuntimeError("TWSE 回傳空內容")
                    if "json" not in ctype and text[:1] not in {"{", "["}:
                        preview = " ".join(text[:160].split())
                        raise RuntimeError(
                            f"TWSE 回傳非 JSON content-type={ctype or 'unknown'} "
                            f"body={preview!r}"
                        )

                    payload = r.json()
                    if not isinstance(payload, dict):
                        raise RuntimeError("TWSE JSON 不是物件")
                    if payload.get("stat") not in (None, "OK"):
                        raise RuntimeError(f"TWSE stat={payload.get('stat')}")
                    return payload
                except Exception as exc:
                    last = exc

            if attempt < max(1, self.cfg.max_retries):
                time.sleep(min(8.0, 2.0 * attempt))

        raise RuntimeError(f"TWSE report 下載失敗 {path}: {last}")

    @staticmethod
    def _legacy_table(payload: dict, required: list[str]) -> pd.DataFrame:
        """Find a table in either modern tables[] or legacy fieldsN/dataN JSON."""
        try:
            return find_table(payload, required)
        except Exception:
            pass

        candidates = []
        for key, fields in payload.items():
            if not str(key).startswith("fields") or not isinstance(fields, list):
                continue
            suffix = str(key)[6:]
            rows = payload.get(f"data{suffix}")
            if not isinstance(rows, list):
                continue
            normalized = [str(x).replace(" ", "").replace("\n", "") for x in fields]
            if all(any(req in col for col in normalized) for req in required):
                return pd.DataFrame(rows, columns=fields)
            candidates.append(fields)

        raise KeyError(f"找不到欄位 {required}; legacy 可用欄位={candidates}")

    def _monthly_core(self, start: date, end: date, *, kind: str) -> pd.DataFrame:
        chunks = []
        failed_months = []

        # The monthly report is requested by month.  Keep the preceding month
        # when needed so TAIEX daily return on the first trading day can still
        # be computed from the prior official close.
        fetch_start = start
        if kind == "taiex":
            fetch_start = (pd.Timestamp(start) - pd.offsets.MonthBegin(1)).date()

        for m in month_starts(fetch_start, end):
            try:
                if kind == "taiex":
                    payload = self._twse_report_json(
                        "indicesReport/MI_5MINS_HIST",
                        {
                            "date": m.strftime("%Y%m01"),
                            "response": "json",
                        },
                    )
                    data = self._legacy_table(payload, ["日期", "收盤指數"])
                    date_col = next(c for c in data if "日期" in str(c))
                    close_col = next(c for c in data if "收盤指數" in str(c))
                    chunk = pd.DataFrame(
                        {
                            "date": data[date_col].map(roc_date),
                            "taiex": data[close_col].map(number),
                        }
                    )
                else:
                    payload = self._twse_report_json(
                        "exchangeReport/STOCK_DAY",
                        {
                            "date": m.strftime("%Y%m01"),
                            "stockNo": "2330",
                            "response": "json",
                        },
                    )
                    data = self._legacy_table(payload, ["日期", "收盤價"])
                    date_col = next(c for c in data if "日期" in str(c))
                    close_col = next(c for c in data if "收盤價" in str(c))
                    change_col = next(
                        (c for c in data if "漲跌價差" in str(c)),
                        None,
                    )
                    close = data[close_col].map(number)
                    change = (
                        data[change_col].map(number)
                        if change_col is not None
                        else pd.Series(np.nan, index=data.index)
                    )
                    previous_close = close - change
                    chunk = pd.DataFrame(
                        {
                            "date": data[date_col].map(roc_date),
                            "tsmc": close,
                            "tsmc_change_pct": (change / previous_close) * 100,
                        }
                    )

                chunk = (
                    chunk
                    .dropna(subset=["date", kind])
                    .drop_duplicates("date")
                )
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

        result = (
            pd.concat(chunks, ignore_index=True)
            .sort_values("date")
            .drop_duplicates("date")
        )

        if kind == "taiex":
            # TWSE historical index report publishes official closes.  Daily
            # return is therefore current official close vs previous official
            # trading-session close.
            result["taiex_change_pct"] = (
                pd.to_numeric(result["taiex"], errors="coerce")
                .pct_change(fill_method=None)
                * 100
            )

        # Return only the caller's requested range after change calculation.
        result = result[
            (result["date"].dt.date >= start)
            & (result["date"].dt.date <= end)
        ].copy()

        if failed_months:
            logging.warning(
                "%s 本輪有 %d 個月份失敗：%s",
                kind.upper(),
                len(failed_months),
                ", ".join(failed_months),
            )

        return result

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