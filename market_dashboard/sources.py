from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import logging
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .price_fallback import apply_reviewed_prices
from .config import Settings
from .utils import get_json, find_table, month_starts, number, roc_date, tables
from .tpex import (
    fetch_tpex_margin as fetch_secondary_margin,
    combine_market_margin,
)
from .fx import fetch_usdtwd


# Yahoo Finance symbols used by the dashboard.
# Step 2 adds broad US equity indexes and the semiconductor index.
GLOBAL_TICKERS = {
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
        /indicesReport and /exchangeReport. A HTTP 200 that redirects to
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
                        raise RuntimeError(
                            f"TWSE endpoint redirected to 404 page: {r.url}"
                        )

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
            normalized = [
                str(x).replace(" ", "").replace("\n", "") for x in fields
            ]
            if all(any(req in col for col in normalized) for req in required):
                return pd.DataFrame(rows, columns=fields)
            candidates.append(fields)

        raise KeyError(f"找不到欄位 {required}; legacy 可用欄位={candidates}")

    def _monthly_core(self, start: date, end: date, *, kind: str) -> pd.DataFrame:
        chunks = []
        failed_months = []

        # Both TAIEX and TSMC returns are calculated from consecutive official
        # closes. Fetch the preceding month so the first requested trading day
        # can still be compared with the previous official trading-session close.
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
                elif kind == "tsmc":
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

                    # Do not depend on TWSE's "漲跌價差" field. Its formatting
                    # can change (or be temporarily unparsable), which previously
                    # produced NaN -> None -> workflow validation failure.
                    chunk = pd.DataFrame(
                        {
                            "date": data[date_col].map(roc_date),
                            "tsmc": data[close_col].map(number),
                        }
                    )
                else:
                    raise ValueError(f"未知 monthly core kind: {kind}")

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

        # Calculate daily returns from consecutive official closes instead of
        # relying on a separate change field supplied by TWSE.
        if kind == "taiex":
            result["taiex_change_pct"] = (
                pd.to_numeric(result["taiex"], errors="coerce")
                .pct_change(fill_method=None)
                * 100
            )
        elif kind == "tsmc":
            result["tsmc_change_pct"] = (
                pd.to_numeric(result["tsmc"], errors="coerce")
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
    
    @staticmethod
    def _validate_daily_payload(
        payload: dict,
        day: pd.Timestamp,
        source: str,
    ) -> None:
        expected = day.strftime("%Y%m%d")

        if not isinstance(payload, dict):
            raise ValueError(f"{source} 回傳格式不是 JSON 物件")

        if payload.get("stat") != "OK":
            raise ValueError(
                f"{source} 回傳異常：{payload.get('stat')}"
            )

        actual = str(payload.get("date", "")).strip()

        if actual != expected:
            raise ValueError(
                f"{source} 日期不符："
                f"預期={expected}，實際={actual}"
            )

    
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

        self._validate_daily_payload(
            payload, day, "MI_INDEX"
        )

        data = find_table(
            payload, ["證券代號", "收盤價"]
        )

        code_col = next(
            x for x in data if "證券代號" in str(x)
        )
        close_col = next(
            x for x in data if "收盤價" in str(x)
        )

        prices = pd.DataFrame({
            "code": data[code_col].astype(str).str.strip(),
            "close": data[close_col].map(number),
        })

        duplicated = prices[
            prices["code"].duplicated(keep=False)
        ]

        if not duplicated.empty:
            codes = duplicated["code"].unique().tolist()
            raise ValueError(
                f"收盤價證券代號重複：{codes[:20]}"
            )

        valid = (
            np.isfinite(prices["close"])
            & (prices["close"] > 0)
        )

        prices = prices.loc[valid].copy()

        if prices.empty:
            raise ValueError("沒有有效的當日收盤價")

        return prices


    
    def _margin_primary(self, day: pd.Timestamp) -> dict:
        """Estimate market-wide maintenance ratio (V3)."""

        key = day.strftime("%Y%m%d")
        common = {"date": key, "response": "json"}

        # 收盤價最低張數配對率
        min_coverage = 0.99

        # 1. 取得融資明細及市場彙總
        detail_payload = get_json(
            self.session,
            "marginTrading/MI_MARGN",
            {**common, "selectType": "ALL"},
            self.cfg.request_timeout_seconds,
            self.cfg.max_retries,
        )

        self._validate_daily_payload(
            detail_payload, day, "MI_MARGN ALL"
        )

        summary_payload = get_json(
            self.session,
            "marginTrading/MI_MARGN",
            {**common, "selectType": "MS"},
            self.cfg.request_timeout_seconds,
            self.cfg.max_retries,
        )

        self._validate_daily_payload(
            summary_payload, day, "MI_MARGN MS"
        )

        def norm(value) -> str:
            return "".join(str(value).split())

        # 2. 尋找融資明細欄位
        detail_rows = None
        code_idx = None
        balance_idx = None
        available_detail_fields = []

        for fields, rows, title in tables(detail_payload):
            names = [norm(x) for x in fields]

            available_detail_fields.append({
                "title": title,
                "fields": names,
            })

            candidate_code = next(
                (i for i, x in enumerate(names)
                 if "代號" in x),
                None,
            )

            candidate_balance = next(
                (i for i, x in enumerate(names)
                 if "融資" in x and "今日餘額" in x),
                None,
            )

            # 相容證交所重複欄位名稱格式
            if (
                candidate_balance is None
                and candidate_code is not None
                and len(names) >= 7
                and names[5] == "前日餘額"
                and names[6] == "今日餘額"
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

        if (
            detail_rows is None
            or code_idx is None
            or balance_idx is None
        ):
            raise KeyError(
                f"找不到個股融資明細："
                f"{available_detail_fields}"
            )

        # 3. 解析並驗證證券代號與融資張數
        parsed_shares = []

        for row in detail_rows:
            if (
                not isinstance(row, (list, tuple))
                or max(code_idx, balance_idx) >= len(row)
            ):
                raise ValueError("融資明細列格式異常")

            code = str(row[code_idx]).strip()

            if not (
                code.isascii()
                and code.isalnum()
                and code[0].isdigit()
                and 4 <= len(code) <= 6
            ):
                if norm(code) in ("合計", "總計", ""):
                    continue

                raise ValueError(
                    f"無法辨識的融資證券代號：{code}"
                )

            lots = number(row[balance_idx])

            if (
                not np.isfinite(lots)
                or lots < 0
                or lots != int(lots)
            ):
                raise ValueError(
                    f"{code} 融資張數異常：{lots}"
                )

            parsed_shares.append((code, lots))

        shares = pd.DataFrame(
            parsed_shares,
            columns=["code", "lots"],
        )

        if shares.empty:
            raise ValueError("個股融資明細解析後為空")

        duplicated = shares[
            shares["code"].duplicated(keep=False)
        ]

        if not duplicated.empty:
            codes = duplicated["code"].unique().tolist()
            raise ValueError(
                f"融資證券代號重複：{codes[:20]}"
            )

        shares = shares[shares["lots"] > 0].copy()

        detail_total_lots = int(shares["lots"].sum())

        if detail_total_lots <= 0:
            raise ValueError("融資明細總張數不是正數")

        # 4. 解析市場融資金額及總張數
        financing_twd = None
        summary_total_lots = None
        available_summary_fields = []
        available_items = []

        for fields, rows, title in tables(summary_payload):
            names = [norm(x) for x in fields]

            available_summary_fields.append({
                "title": title,
                "fields": names,
            })

            item_idx = next(
                (i for i, x in enumerate(names)
                 if "項目" in x),
                None,
            )

            today_candidates = [
                i for i, x in enumerate(names)
                if "今日餘額" in x
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

                item = norm(row[item_idx])
                available_items.append(item)

                value = number(row[today_idx])

                if "融資金額" in item:
                    if np.isfinite(value) and value > 0:
                        financing_twd = value * 1000

                elif (
                    item.startswith("融資")
                    and (
                        "交易單位" in item
                        or "張數" in item
                    )
                ):
                    if (
                        np.isfinite(value)
                        and value > 0
                        and value == int(value)
                    ):
                        summary_total_lots = int(value)

            if (
                financing_twd is not None
                and summary_total_lots is not None
            ):
                break

        if financing_twd is None:
            raise ValueError(
                f"找不到有效融資金額："
                f"{available_summary_fields}"
            )

        if summary_total_lots is None:
            raise ValueError(
                f"找不到市場融資總張數："
                f"{available_items}"
            )

        # 5. 核對明細與市場彙總
        if detail_total_lots != summary_total_lots:
            raise ValueError(
                f"{key} 融資總張數不一致："
                f"明細={detail_total_lots:,}，"
                f"彙總={summary_total_lots:,}，"
                f"差額="
                f"{detail_total_lots-summary_total_lots:,}"
            )

        # 6. 收盤價配對
        prices = self.close_prices(day)
        original = dict(zip(prices["code"], prices["close"]))
        balances = dict(zip(shares["code"], shares["lots"]))
        logging.info("PRICE_RAW_COVERAGE | primary | %s | %.6f%%", key,
                     100 * sum(lots for code, lots in balances.items() if code in original) / summary_total_lots)
        filled = apply_reviewed_prices("primary", day, balances, original)
        prices = pd.DataFrame(list(filled.items()), columns=["code", "close"])

        merged = shares.merge(
            prices,
            on="code",
            how="left",
            validate="one_to_one",
        )

        valid = (
            np.isfinite(merged["close"])
            & (merged["close"] > 0)
        )

        matched = merged.loc[valid].copy()
        missing = merged.loc[~valid].copy()

        matched_lots = int(matched["lots"].sum())
        missing_lots = int(missing["lots"].sum())

        coverage = matched_lots / summary_total_lots

        # 7. 缺價警告及配對率檢查
        if not missing.empty:
            missing_codes = missing["code"].tolist()

            logging.warning(
                "%s 融資收盤價缺失："
                "%d 檔、%s 張，配對率=%.6f%%，"
                "證券=%s",
                key,
                len(missing),
                f"{missing_lots:,}",
                coverage * 100,
                ", ".join(missing_codes[:20]),
            )

        if coverage < min_coverage:
            logging.warning(
                f"{key} 收盤價配對率不足："
                f"{coverage*100:.6f}% "
                f"低於 {min_coverage*100:.1f}%"
            )

        # 8. 維持原本 V3 核心公式
        collateral_twd = float(
            (
                matched["lots"]
                * 1000
                * matched["close"]
            ).sum()
        )

        if (
            not np.isfinite(collateral_twd)
            or collateral_twd <= 0
        ):
            raise ValueError("估算擔保品市值異常")

        maintenance_est = (
            collateral_twd / financing_twd * 100
        )

        if not np.isfinite(maintenance_est):
            raise ValueError("估算維持率異常")

        # 9. 紀錄資料品質
        logging.info(
            "%s V3 融資資料驗證通過："
            "明細=%s 張，彙總=%s 張，"
            "配對率=%.6f%%，"
            "維持率=%.4f%%",
            key,
            f"{detail_total_lots:,}",
            f"{summary_total_lots:,}",
            coverage * 100,
            maintenance_est,
        )

        return {
            "margin_balance_billion": financing_twd / 1e9,
            "maintenance_est": maintenance_est,
        }

    def margin(self, day: pd.Timestamp) -> dict:
        """市場融資擔保比估算。"""

        day = pd.Timestamp(day)

        primary = self._margin_primary(day)
        secondary = fetch_secondary_margin(day)

        return combine_market_margin(primary, secondary)
    
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
        """下載央行匯率及 Yahoo Finance 全球市場行情。"""

        fx = fetch_usdtwd(start, end)

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
            raise RuntimeError("Yahoo Finance 外部市場資料為空")

        if isinstance(raw.columns, pd.MultiIndex):
            try:
                close = raw["Close"]
            except KeyError:
                raise RuntimeError("Yahoo Finance 缺少 Close 欄位")
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

        out = (
            out[columns]
            .sort_values("date")
            .drop_duplicates("date")
        )

        # 全球市場日線收盤防護。
        # 保守採用美東時間當日 18:00 作為發布截止門檻。
        now_ny = datetime.now(
            ZoneInfo("America/New_York")
        )

        completed_date = now_ny.date()

        if now_ny.hour < 18:
            completed_date -= timedelta(days=1)

        completed_date = min(completed_date, end)

        before_count = len(out)

        out = out.loc[
            out["date"] <= pd.Timestamp(completed_date)
        ].copy()

        logging.info(
            "全球市場收盤防護：截至 %s；排除 %d 筆未完成日期",
            completed_date.isoformat(),
            before_count - len(out),
        )
        
        out = out.merge(
            fx,
            on="date",
            how="outer",
            validate="one_to_one",
        )

        logging.info(
            "央行匯率：%s = %.3f",
            fx.iloc[-1]["date"].strftime("%Y-%m-%d"),
            fx.iloc[-1]["usdtwd"],
        )

        return out.sort_values("date").reset_index(drop=True)


def fetch_twse_official_maintenance(date):
    """取得證交所官方整戶擔保維持率及追繳處分資料。"""
    import requests

    date_str = str(date).replace("-", "")[:8]

    url = (
        "https://www.twse.com.tw"
        "/rwd/zh/marginTrading/BFIJ3U"
    )

    params = {
        "response": "json",
        "date": date_str,
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("stat") != "OK":
        return None

    if data.get("date") != date_str:
        return None

    if data.get("keepRate") is None:
        return None

    return {
        "maintenance_official": float(data["keepRate"]),
        "maintenance_official_date": data["date"],
    }
