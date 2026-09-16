from datetime import date

import pandas as pd
import pytest

from market_dashboard.sources import Sources


class DummySettings:
    max_retries = 1
    request_timeout_seconds = 5
    request_delay_seconds = 0


def test_tsmc_change_pct_uses_consecutive_official_closes(monkeypatch):
    src = Sources(DummySettings())

    august = {
        "fields": ["日期", "收盤價"],
        "data": [
            ["115/08/31", "2,300.00"],
        ],
        "stat": "OK",
    }
    september = {
        "fields": ["日期", "收盤價"],
        "data": [
            ["115/09/01", "2,320.00"],
            ["115/09/16", "2,380.00"],
        ],
        "stat": "OK",
    }

    def fake_report(path, params):
        assert path == "exchangeReport/STOCK_DAY"
        if params["date"].startswith("202608"):
            return august
        return september

    monkeypatch.setattr(src, "_twse_report_json", fake_report)

    df = src.tsmc(date(2026, 9, 1), date(2026, 9, 16))

    assert list(df["tsmc"]) == [2320.0, 2380.0]
    assert pd.notna(df.iloc[0]["tsmc_change_pct"])
    assert pd.notna(df.iloc[1]["tsmc_change_pct"])

    # 2300 -> 2320
    assert df.iloc[0]["tsmc_change_pct"] == pytest.approx(
        (2320 / 2300 - 1) * 100
    )
    # 2320 -> 2380
    assert df.iloc[1]["tsmc_change_pct"] == pytest.approx(
        (2380 / 2320 - 1) * 100
    )


def test_tsmc_does_not_require_twse_change_column(monkeypatch):
    src = Sources(DummySettings())

    payloads = {
        "202608": {
            "fields": ["日期", "收盤價"],
            "data": [["115/08/31", "2,300.00"]],
            "stat": "OK",
        },
        "202609": {
            "fields": ["日期", "收盤價"],
            "data": [["115/09/01", "2,320.00"]],
            "stat": "OK",
        },
    }

    def fake_report(path, params):
        return payloads[params["date"][:6]]

    monkeypatch.setattr(src, "_twse_report_json", fake_report)

    df = src.tsmc(date(2026, 9, 1), date(2026, 9, 1))

    assert len(df) == 1
    assert df.iloc[0]["tsmc"] == 2320.0
    assert pd.notna(df.iloc[0]["tsmc_change_pct"])
