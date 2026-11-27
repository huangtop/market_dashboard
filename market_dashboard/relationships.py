
import numpy as np
import pandas as pd


WINDOWS = (20, 60, 120, 252)

LABELS = {
    "taiex": "加權指數",
    "tsmc": "台積電",
    "maintenance_est": "V4 融資擔保比",
    "foreign_net_100m": "外資買賣超",
    "usdtwd": "美元兌台幣",
    "dxy": "美元指數",
    "sp500": "S&P 500",
    "nasdaq": "Nasdaq",
    "dow": "道瓊",
    "sox": "費城半導體",
    "vix": "VIX",
    "us10y": "美國10年期殖利率",
}

GLOBAL = (
    "usdtwd", "dxy", "sp500", "nasdaq",
    "dow", "sox", "vix", "us10y",
)

DIFFERENCE = (
    "maintenance_est", "vix", "us10y",
)


def number(value):
    if pd.isna(value) or not np.isfinite(value):
        return None
    return round(float(value), 6)


def prepare(frame):
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    return (
        df.sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")
    )


def values(frame, key):
    if key not in frame:
        return pd.Series(dtype=float)

    return (
        pd.to_numeric(frame[key], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def changes(frame, key):
    s = values(frame, key)

    if key == "foreign_net_100m":
        return s

    if key in DIFFERENCE:
        result = s.diff()
        return result * 100 if key == "us10y" else result

    return s.where(s > 0).pct_change(
        fill_method=None
    ) * 100


def align_to_taiwan(series, dates, exact=False):
    left = pd.DataFrame({"date": dates})

    right = (
        series.rename("value")
        .rename_axis("source_date")
        .reset_index()
    )

    result = pd.merge_asof(
        left.sort_values("date"),
        right.sort_values("source_date"),
        left_on="date",
        right_on="source_date",
        direction="backward",
        allow_exact_matches=exact,
        tolerance=pd.Timedelta(days=7),
    )

    # 同一來源交易日不重複計入相關性
    duplicate = result["source_date"].duplicated()
    result.loc[duplicate, "value"] = np.nan

    return result.set_index("date")["value"]


def pair_analysis(x, y, name_x, name_y):
    pair = pd.concat(
        [x.rename("x"), y.rename("y")],
        axis=1,
        join="inner",
    ).dropna()

    latest = {}
    rolling = {}

    for window in WINDOWS:
        key = str(window)
        corr = pair["x"].rolling(
            window, min_periods=window
        ).corr(pair["y"])

        latest[key] = {
            "correlation": (
                number(corr.iloc[-1])
                if len(corr) else None
            ),
            "samples": min(len(pair), window),
        }

        rolling[key] = [
            {
                "date": d.strftime("%Y-%m-%d"),
                "correlation": number(v),
            }
            for d, v in corr.dropna().items()
        ]

    return {
        "x": name_x,
        "y": name_y,
        "x_label": LABELS[name_x],
        "y_label": LABELS[name_y],
        "latest": latest,
        "rolling": rolling,
        "latest_date": (
            pair.index[-1].strftime("%Y-%m-%d")
            if len(pair) else None
        ),
    }


def lag_analysis(target, indicator, window=252):
    results = []

    for lag in range(-5, 6):
        pair = pd.concat(
            [
                target.rename("target"),
                indicator.shift(lag).rename("indicator"),
            ],
            axis=1,
        ).dropna().tail(window)

        corr = (
            pair["target"].corr(pair["indicator"])
            if len(pair) >= 30 else np.nan
        )

        results.append({
            "lag": lag,
            "correlation": number(corr),
            "samples": len(pair),
        })

    return results


def event_backtest(price, condition):
    condition = condition.reindex(
        price.index
    ).fillna(False).astype(bool)

    # 僅統計首次進入條件的交易日
    events = condition & ~condition.shift(
        1, fill_value=False
    )

    result = {}

    for horizon in (5, 10, 20, 60):
        returns = (
            price.shift(-horizon) / price - 1
        ) * 100

        sample = returns[events].dropna()

        result[str(horizon)] = {
            "samples": len(sample),
            "mean": number(sample.mean()),
            "median": number(sample.median()),
            "win_rate": (
                number((sample > 0).mean() * 100)
                if len(sample) else None
            ),
            "worst_return": number(sample.min()),
            "best_return": number(sample.max()),
        }

    return result


def quadrants(price, fx):
    data = pd.concat(
        [
            price.rename("stock"),
            fx.rename("fx"),
        ],
        axis=1,
    ).dropna()

    change = data.pct_change(
        periods=20, fill_method=None
    ) * 100

    result = []

    for day, row in change.dropna().iterrows():
        stock = row["stock"]
        currency = -row["fx"]

        if stock == 0 or currency == 0:
            label = "中性"
        elif stock > 0 and currency > 0:
            label = "股匯雙升"
        elif stock > 0:
            label = "股強匯弱"
        elif currency > 0:
            label = "股弱匯強"
        else:
            label = "股匯雙弱"

        result.append({
            "date": day.strftime("%Y-%m-%d"),
            "stock_change": number(stock),
            "twd_strength": number(currency),
            "quadrant": label,
        })

    return result


def build_relationships(taiwan_df, global_df):
    tw = prepare(taiwan_df)
    gl = prepare(global_df)

    tw = tw.loc[values(tw, "taiex").index]

    tw_changes = {}
    global_changes = {}

    for key in LABELS:
        if key in GLOBAL:
            global_changes[key] = changes(gl, key)

            tw_changes[key] = align_to_taiwan(
                global_changes[key],
                tw.index,
                exact=(key == "usdtwd"),
            )
        else:
            tw_changes[key] = changes(
                tw, key
            ).reindex(tw.index)

    pairs = {}

    for key in LABELS:
        if key == "taiex":
            continue

        pairs[f"taiex_{key}"] = pair_analysis(
            tw_changes["taiex"],
            tw_changes[key],
            "taiex",
            key,
        )

    for stock in ("sp500", "nasdaq", "dow", "sox"):
        for key in ("vix", "us10y", "dxy"):
            pairs[f"{stock}_{key}"] = pair_analysis(
                global_changes[stock],
                global_changes[key],
                stock,
                key,
            )

    rankings = {}

    for window in WINDOWS:
        key = str(window)
        items = []

        for pair_id, pair in pairs.items():
            item = pair["latest"][key]
            corr = item["correlation"]

            if corr is not None:
                items.append({
                    "pair": pair_id,
                    "label": (
                        pair["x_label"] + " / " +
                        pair["y_label"]
                    ),
                    "correlation": corr,
                    "samples": item["samples"],
                    "date": pair["latest_date"],
                })

        rankings[key] = {
            "positive": sorted(
                [r for r in items
                 if r["correlation"] > 0],
                key=lambda r: r["correlation"],
                reverse=True,
            ),
            "negative": sorted(
                [r for r in items
                 if r["correlation"] < 0],
                key=lambda r: r["correlation"],
            ),
        }

    lags = {
        key: lag_analysis(
            tw_changes["taiex"],
            tw_changes[key],
        )
        for key in GLOBAL
    }

    price = values(tw, "taiex")
    maintenance = values(tw, "maintenance_est")

    # 使用當時可取得的歷史分布，避免未來資料偏誤
    percentile = maintenance.expanding(
        min_periods=252
    ).rank(pct=True) * 100

    maintenance_tests = {
        str(level): event_backtest(
            price, percentile <= level
        )
        for level in (5, 10, 20)
    }

    vix = align_to_taiwan(
        values(gl, "vix"),
        tw.index,
    )

    vix_tests = {
        str(level): event_backtest(
            price, vix >= level
        )
        for level in (20, 25, 30, 35, 40)
    }

    divergence = []

    for pair_id, pair in pairs.items():
        short = pair["latest"]["20"]["correlation"]
        long = pair["latest"]["120"]["correlation"]

        if short is None or long is None:
            continue

        divergence.append({
            "pair": pair_id,
            "correlation_20": short,
            "correlation_120": long,
            "difference": number(short - long),
            "sign_reversal": short * long < 0,
        })

    divergence.sort(
        key=lambda r: abs(r["difference"]),
        reverse=True,
    )

    global_series = {}

    for key in GLOBAL:
        global_series[key] = [
            {
                "date": day.strftime("%Y-%m-%d"),
                "value": number(value),
            }
            for day, value in values(gl, key).items()
        ]

    return {
        "version": "6.0.0",
        "windows": list(WINDOWS),
        "pairs": pairs,
        "rankings": rankings,
        "lead_lag": lags,
        "backtests": {
            "maintenance_percentile": maintenance_tests,
            "vix_threshold": vix_tests,
        },
        "quadrants": quadrants(
            price, values(gl, "usdtwd")
        ),
        "divergence": divergence,
        "global_series": global_series,
        "methodology": {
            "correlation": "Pearson on daily changes",
            "window_unit": "valid paired observations",
            "maintenance": "percentage point change",
            "vix": "index point change",
            "us10y": "basis point change",
            "foreign": "daily net flow in 100m TWD",
            "lag_positive": "indicator leads target",
            "lag_negative": "indicator follows target",
            "backtest": "first threshold entry",
            "maintenance_min_history": 252,
            "quadrant_window": 20,
        },
    }

