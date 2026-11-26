from datetime import date

import pandas as pd
import requests


FX_URL = "https://cpx.cbc.gov.tw/api/OpenData/FTDOpenData_Day"


def fetch_usdtwd(start: date, end: date) -> pd.DataFrame:
    response = requests.get(FX_URL, timeout=30)
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or not data:
        raise ValueError("央行匯率資料格式異常")

    df = pd.DataFrame(data)

    df = df.rename(columns={
        "日期": "date",
        "NTD_USD": "usdtwd",
    })

    df = df[["date", "usdtwd"]].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y%m%d",
        errors="raise",
    )

    df["usdtwd"] = pd.to_numeric(
        df["usdtwd"],
        errors="raise",
    )

    if df["date"].duplicated().any():
        raise ValueError("央行匯率日期重複")

    if not df["usdtwd"].between(0, 100).all():
        raise ValueError("央行匯率數值異常")

    df = df.loc[
        (df["date"] >= pd.Timestamp(start))
        & (df["date"] <= pd.Timestamp(end))
    ]

    if df.empty:
        raise ValueError("指定期間沒有央行匯率資料")

    return df.sort_values("date").reset_index(drop=True)