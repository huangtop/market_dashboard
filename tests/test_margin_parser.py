import sys, types
sys.modules.setdefault("yfinance", types.SimpleNamespace())
from types import SimpleNamespace
import pandas as pd
import market_dashboard.sources as sources_module
from market_dashboard.sources import Sources


def test_margin_handles_duplicate_column_labels(monkeypatch):
    detail = {
        "stat": "OK",
        "tables": [{
            "title": "個股融資融券明細",
            "fields": ["代號", "名稱", "買進", "賣出", "現償", "前日餘額", "今日餘額", "限額", "買進", "賣出", "今日餘額"],
            "data": [
                ["2330", "台積電", "1", "0", "0", "9", "10", "100", "0", "0", "2"],
                ["2317", "鴻海", "1", "0", "0", "19", "20", "100", "0", "0", "3"],
            ],
        }],
    }
    summary = {
        "stat": "OK",
        "tables": [{
            "title": "信用交易統計",
            "fields": ["項目", "買進", "賣出", "現償", "前日餘額", "今日餘額"],
            "data": [["融資金額", "0", "0", "0", "0", "100000"]],
        }],
    }

    calls = iter([detail, summary])
    monkeypatch.setattr(sources_module, "get_json", lambda *args, **kwargs: next(calls))
    cfg = SimpleNamespace(request_timeout_seconds=10, max_retries=1)
    src = Sources(cfg)
    monkeypatch.setattr(src, "close_prices", lambda day: pd.DataFrame({
        "code": ["2330", "2317"], "close": [1000.0, 200.0]
    }))

    result = src.margin(pd.Timestamp("2025-06-02"))
    assert result["margin_balance_billion"] == 0.1
    # (10*1000*1000 + 20*1000*200) / 100,000,000 * 100 = 14%
    assert round(result["maintenance_est"], 6) == 14.0


def test_margin_prefers_positional_margin_balance(monkeypatch):
    detail = {
        "stat": "OK",
        "fields": ["證券代號", "證券名稱", "買進", "賣出", "現償", "前日餘額", "今日餘額", "限額", "買進", "賣出", "今日餘額"],
        "data": [["2330", "台積電", "0", "0", "0", "4", "5", "50", "0", "0", "999"]],
    }
    summary = {
        "stat": "OK",
        "fields": ["項目", "今日餘額"],
        "data": [["融資金額", "50000"]],
    }
    calls = iter([detail, summary])
    monkeypatch.setattr(sources_module, "get_json", lambda *args, **kwargs: next(calls))
    cfg = SimpleNamespace(request_timeout_seconds=10, max_retries=1)
    src = Sources(cfg)
    monkeypatch.setattr(src, "close_prices", lambda day: pd.DataFrame({"code": ["2330"], "close": [500.0]}))

    result = src.margin(pd.Timestamp("2025-06-02"))
    assert round(result["maintenance_est"], 6) == 5.0
