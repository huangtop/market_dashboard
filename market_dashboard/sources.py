from __future__ import annotations
from datetime import date, timedelta
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from .config import Settings
from .utils import get_json, find_table, month_starts, number, roc_date, tables

class Sources:
    def __init__(self, settings: Settings):
        self.cfg = settings
        self.session = requests.Session()

    def taiex(self, start: date, end: date) -> pd.DataFrame:
        chunks=[]
        for m in month_starts(start,end):
            p=get_json(self.session,"TAIEX/MI_5MINS_HIST",{"date":m.strftime("%Y%m01"),"response":"json"},self.cfg.request_timeout_seconds,self.cfg.max_retries)
            d=find_table(p,["日期","收盤指數"]); dc=next(c for c in d if "日期" in str(c)); cc=next(c for c in d if "收盤指數" in str(c))
            chunks.append(pd.DataFrame({"date":d[dc].map(roc_date),"taiex":d[cc].map(number)}))
        return pd.concat(chunks,ignore_index=True).dropna().drop_duplicates("date")

    def tsmc(self, start: date, end: date) -> pd.DataFrame:
        chunks=[]
        for m in month_starts(start,end):
            p=get_json(self.session,"afterTrading/STOCK_DAY",{"date":m.strftime("%Y%m01"),"stockNo":"2330","response":"json"},self.cfg.request_timeout_seconds,self.cfg.max_retries)
            d=find_table(p,["日期","收盤價"]); dc=next(c for c in d if "日期" in str(c)); cc=next(c for c in d if "收盤價" in str(c))
            chunks.append(pd.DataFrame({"date":d[dc].map(roc_date),"tsmc":d[cc].map(number)}))
        return pd.concat(chunks,ignore_index=True).dropna().drop_duplicates("date")

    def close_prices(self, day: pd.Timestamp) -> pd.DataFrame:
        p=get_json(self.session,"afterTrading/MI_INDEX",{"date":day.strftime("%Y%m%d"),"type":"ALLBUT0999","response":"json"},self.cfg.request_timeout_seconds,self.cfg.max_retries)
        d=find_table(p,["證券代號","收盤價"]); c=next(x for x in d if "證券代號" in str(x)); q=next(x for x in d if "收盤價" in str(x))
        return pd.DataFrame({"code":d[c].astype(str).str.strip(),"close":d[q].map(number)}).dropna()

    def margin(self, day: pd.Timestamp) -> dict:
        """Estimate market-wide maintenance ratio from TWSE margin data.

        TWSE may return duplicated column labels after merged headers are
        flattened (for example two columns both named ``今日餘額``).  This
        parser therefore selects columns by *position*, never by a duplicated
        pandas label.
        """
        common = {"date": day.strftime("%Y%m%d"), "response": "json"}
        detail_payload = get_json(
            self.session, "marginTrading/MI_MARGN",
            {**common, "selectType": "STOCK"},
            self.cfg.request_timeout_seconds, self.cfg.max_retries,
        )
        summary_payload = get_json(
            self.session, "marginTrading/MI_MARGN",
            {**common, "selectType": "MS"},
            self.cfg.request_timeout_seconds, self.cfg.max_retries,
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
            candidate_code = next((i for i, x in enumerate(names) if "代號" in x), None)
            candidate_balance = next(
                (i for i, x in enumerate(names) if "融資" in x and "今日餘額" in x),
                None,
            )
            # Merged TWSE headers can collapse to duplicated generic labels.
            # The documented stock-detail layout places margin today's balance
            # at zero-based position 6.
            if candidate_balance is None and candidate_code is not None and len(names) >= 7:
                candidate_balance = 6
            if candidate_code is not None and candidate_balance is not None and rows:
                detail_rows = rows
                code_idx = candidate_code
                balance_idx = candidate_balance
                break

        if detail_rows is None or code_idx is None or balance_idx is None:
            raise KeyError(f"找不到個股融資明細，TWSE 欄位={available_detail_fields}")

        parsed_shares = []
        for row in detail_rows:
            if not isinstance(row, (list, tuple)) or max(code_idx, balance_idx) >= len(row):
                continue
            code = str(row[code_idx]).strip()
            lots = number(row[balance_idx])
            if code.isdigit() and 4 <= len(code) <= 6 and np.isfinite(lots):
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
            item_idx = next((i for i, x in enumerate(names) if "項目" in x), None)
            today_candidates = [i for i, x in enumerate(names) if "今日餘額" in x]
            if item_idx is None or not today_candidates:
                continue
            # In the MS table, use the first today's-balance column belonging
            # to the financing section.  For a duplicated flattened header,
            # the row label "融資金額" identifies the correct record.
            today_idx = today_candidates[0]
            for row in rows:
                if not isinstance(row, (list, tuple)) or max(item_idx, today_idx) >= len(row):
                    continue
                if "融資金額" not in norm(row[item_idx]):
                    continue
                value = number(row[today_idx])
                if np.isfinite(value) and value > 0:
                    financing_twd = value * 1000  # TWSE unit: NT$ thousand
                    break
            if financing_twd is not None:
                break

        if financing_twd is None:
            raise KeyError(f"找不到融資金額，TWSE 欄位={available_summary_fields}")

        prices = self.close_prices(day)
        merged = shares.merge(prices, on="code", how="inner").dropna(subset=["lots", "close"])
        if merged.empty:
            raise ValueError("融資明細與當日收盤價無法配對")

        collateral_twd = float((merged["lots"] * 1000 * merged["close"]).sum())
        if collateral_twd <= 0:
            raise ValueError("估算擔保品市值不是正數")

        return {
            "margin_balance_billion": financing_twd / 1e9,
            "maintenance_est": collateral_twd / financing_twd * 100,
        }

    def foreign(self, day: pd.Timestamp) -> float:
        p=get_json(self.session,"fund/BFI82U",{"dayDate":day.strftime("%Y%m%d"),"type":"day","response":"json"},self.cfg.request_timeout_seconds,self.cfg.max_retries)
        d=find_table(p,["單位名稱","買賣差額"]); u=next(c for c in d if "單位名稱" in str(c)); n=next(c for c in d if "買賣差額" in str(c))
        r=d[d[u].astype(str).str.contains("外資及陸資",na=False)]
        return float(r[n].map(number).sum()/1e8) if not r.empty else np.nan

    def external(self, start: date, end: date) -> pd.DataFrame:
        raw=yf.download(["TWD=X","DX-Y.NYB","^VIX"],start=start.isoformat(),end=(end+timedelta(days=2)).isoformat(),auto_adjust=False,progress=False,threads=True)
        if raw.empty: return pd.DataFrame(columns=["date","usdtwd","dxy","vix"])
        close=raw["Close"] if isinstance(raw.columns,pd.MultiIndex) else raw
        out=pd.DataFrame(index=close.index)
        out["usdtwd"]=close.get("TWD=X"); out["dxy"]=close.get("DX-Y.NYB"); out["vix"]=close.get("^VIX")
        out.index=pd.to_datetime(out.index).tz_localize(None)
        return out.reset_index().rename(columns={"Date":"date","index":"date"})[["date","usdtwd","dxy","vix"]]
