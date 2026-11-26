"""Reviewed historical valuation overrides; no automatic stale-price fill."""
import csv
import logging
import math
from pathlib import Path
from datetime import date

RULES_PATH = Path(__file__).with_name('price_fallback.csv')


def apply_reviewed_prices(market, day, balances, prices, rules_path=None):
    day = date.fromisoformat(str(day)[:10])
    result = dict(prices)
    path = Path(rules_path) if rules_path is not None else RULES_PATH
    if not path.exists():
        return result
    used = set()
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['market'] != market:
                continue
            start = date.fromisoformat(row['start'])
            end = date.fromisoformat(row['end'])
            if not start <= day <= end:
                continue
            code = row['code'].strip()
            if balances.get(code, 0) <= 0 or code in prices:
                continue
            previous = date.fromisoformat(row['price_date'])
            price = float(row['price'])
            if not (previous < start <= end and math.isfinite(price) and price > 0 and row['reason'].strip() and row['source'].strip()):
                raise ValueError(f'Invalid reviewed price rule: {market} {code}')
            if code in used:
                raise ValueError(f'Overlapping reviewed price rules: {market} {code}')
            used.add(code)
            result[code] = price
            logging.info('PRICE_FALLBACK | %s | %s | %s | lots=%s | price=%s | price_date=%s | reason=%s | source=%s', market, day, code, balances[code], price, previous, row['reason'], row['source'])
    return result
