
import csv
import io
import logging
import math
import re
from pathlib import Path

import pandas as pd

from .price_fallback import apply_reviewed_prices


CODE_PATTERN = re.compile(r"[0-9][0-9A-Z]{3,5}")
MIN_PRICE_COVERAGE = 0.99


def read_csv_rows(path):
    raw = Path(path).read_bytes()

    for encoding in ("utf-8-sig", "cp950"):
        try:
            text = raw.decode(encoding)
            return list(csv.reader(io.StringIO(text)))
        except UnicodeDecodeError:
            continue

    raise ValueError(f"無法辨識 CSV 編碼：{path}")


def parse_number(value):
    number = float(str(value).replace(",", "").strip())

    if not math.isfinite(number):
        raise ValueError(f"無效數值：{value}")

    return number


def validate_date(rows, day):
    day = pd.Timestamp(day)

    expected = (
        f"{day.year - 1911}/"
        f"{day.month:02d}/{day.day:02d}"
    )

    for row in rows[:10]:
        text = " ".join(row).replace(" ", "")
        if "資料日期" in text:
            if expected not in text:
                raise ValueError(
                    f"資料日期不符：{text}，預期 {expected}"
                )
            return

    raise ValueError("找不到資料日期")


def parse_margin_csv(path, day):
    rows = read_csv_rows(path)
    validate_date(rows, day)

    balances = {}
    total_lots = None
    financing_twd = None

    header = next(
        (r for r in rows if r and r[0] == "代號"),
        None,
    )

    if header is None or "資餘額" not in header:
        raise ValueError("找不到上櫃融資欄位")

    idx = header.index("資餘額")

    for row in rows:
        if len(row) <= idx:
            continue

        code = row[0].strip()

        if code == "合計(張)":
            if total_lots is not None:
                raise ValueError("重複融資張數彙總")
            total_lots = parse_number(row[idx])
            continue

        if code == "融資金(仟元)":
            if financing_twd is not None:
                raise ValueError("重複融資金額彙總")
            financing_twd = parse_number(row[idx]) * 1000
            continue

        if not CODE_PATTERN.fullmatch(code):
            continue

        if code in balances:
            raise ValueError(f"重複融資代號：{code}")

        lots = parse_number(row[idx])

        if lots < 0 or not lots.is_integer():
            raise ValueError(f"不合理融資張數：{code}")

        balances[code] = int(lots)

    if total_lots is None or total_lots <= 0:
        raise ValueError("缺少有效融資總張數")

    if financing_twd is None or financing_twd <= 0:
        raise ValueError("缺少有效融資金額")

    if sum(balances.values()) != total_lots:
        raise ValueError("上櫃融資明細與彙總不一致")

    return balances, financing_twd


def parse_close_csv(path, day):
    rows = read_csv_rows(path)
    validate_date(rows, day)

    prices = {}
    seen = set()

    header = next(
        (r for r in rows if r and r[0] == "代號"),
        None,
    )

    if header is None:
        raise ValueError("找不到收盤行情欄位")

    idx = [v.strip() for v in header].index("收盤")

    for row in rows:
        if len(row) <= idx:
            continue

        code = row[0].strip()

        if not CODE_PATTERN.fullmatch(code):
            continue

        if code in seen:
            raise ValueError(f"重複收盤行情代號：{code}")

        seen.add(code)

        try:
            price = parse_number(row[idx])
        except ValueError:
            continue

        if price > 0:
            prices[code] = price

    if not prices:
        raise ValueError("沒有有效上櫃收盤價")

    return prices


def calculate_tpex_margin(margin_path, close_path, day):
    balances, financing_twd = parse_margin_csv(
        margin_path, day
    )

    prices = parse_close_csv(close_path, day)

    active = {
        code: lots
        for code, lots in balances.items()
        if lots > 0
    }

    total_lots = sum(active.values())
    logging.info("PRICE_RAW_COVERAGE | secondary | %s | %.6f%%", day,
                 100 * sum(lots for code, lots in active.items() if code in prices) / total_lots)
    prices = apply_reviewed_prices("secondary", day, active, prices)

    missing = {
        code: lots
        for code, lots in active.items()
        if code not in prices
    }

    missing_lots = sum(missing.values())
    coverage = 1 - missing_lots / total_lots

    if coverage < MIN_PRICE_COVERAGE:
        logging.warning(
            f"上櫃股價配對率不足：{coverage:.6%}；"
            f"缺少證券={missing}；"
            f"缺少張數={missing_lots:,}；"
            f"總融資張數={total_lots:,}"
        )

    if missing:
        logging.warning(
            "上櫃缺少收盤價：%s；配對率=%.6f%%",
            missing,
            coverage * 100,
        )

    collateral_twd = sum(
        lots * 1000 * prices[code]
        for code, lots in active.items()
        if code in prices
    )

    ratio = collateral_twd / financing_twd * 100

    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("上櫃融資擔保比不合理")

    return {
        "financing_twd": financing_twd,
        "collateral_twd": collateral_twd,
        "total_lots": total_lots,
        "total_securities": len(active),
        "price_coverage_pct": coverage * 100,
        "maintenance_est": ratio,
    }

def combine_market_margin(twse, tpex):
    """合併上市及上櫃融資擔保比。"""

    twse_financing = float(
        twse["margin_balance_billion"]
    ) * 1e9

    twse_ratio = float(twse["maintenance_est"])

    twse_collateral = (
        twse_financing * twse_ratio / 100
    )

    tpex_financing = float(tpex["financing_twd"])
    tpex_collateral = float(tpex["collateral_twd"])

    values = (
        twse_financing,
        twse_collateral,
        tpex_financing,
        tpex_collateral,
    )

    if not all(
        math.isfinite(v) and v > 0
        for v in values
    ):
        raise ValueError("上市櫃融資資料不完整或不合理")

    total_financing = twse_financing + tpex_financing
    total_collateral = twse_collateral + tpex_collateral

    ratio = total_collateral / total_financing * 100

    logging.info(
        "V4 上市櫃融資擔保比："
        "上市=%.4f%%；上櫃=%.4f%%；"
        "合併=%.4f%%；融資金額=%.4f 億元",
        twse_ratio,
        tpex_collateral / tpex_financing * 100,
        ratio,
        total_financing / 1e8,
    )

    return {
        "margin_balance_billion": total_financing / 1e9,
        "maintenance_est": ratio,
    }

def fetch_tpex_margin(day):
    import requests
    import tempfile

    day = pd.Timestamp(day)

    date_ad = day.strftime("%Y/%m/%d")
    date_roc = (
        f"{day.year - 1911}/"
        f"{day.month:02d}/{day.day:02d}"
    )

    base = "https://www.tpex.org.tw"

    endpoints = {
        "margin": (
            f"{base}/web/stock/margin_trading/"
            "margin_balance/margin_bal_result.php",
            {
                "l": "zh-tw",
                "d": date_roc,
                "o": "csv",
                "s": "0,asc",
            },
        ),
        "close": (
            f"{base}/www/zh-tw/afterTrading/otc",
            {
                "date": date_ad,
                "type": "AL",
                "response": "csv",
            },
        ),
    }

    with tempfile.TemporaryDirectory() as tmp:
        paths = {}

        with requests.Session() as session:
            for name, (url, params) in endpoints.items():
                response = session.get(
                    url,
                    params=params,
                    timeout=60,
                )
                response.raise_for_status()

                raw = response.content

                if not raw.strip():
                    raise ValueError(f"{name} 回傳空資料")

                prefix = raw[:500].lower()

                if b"<!doctype html" in prefix or b"<html" in prefix:
                    raise ValueError(
                        f"{name} 回傳 HTML：{response.url}"
                    )

                path = Path(tmp) / f"{name}.csv"
                path.write_bytes(raw)

                rows = read_csv_rows(path)
                validate_date(rows, day)

                paths[name] = path

        return calculate_tpex_margin(
            paths["margin"],
            paths["close"],
            day,
        )
