#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, logging, time
from datetime import date, datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from market_dashboard.config import Settings
from market_dashboard.sources import Sources
from market_dashboard.storage import Store
from market_dashboard.analysis import enrich, backtests, commentary
from market_dashboard.utils import safe_float

def setup_log():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",handlers=[logging.StreamHandler(),logging.FileHandler("logs/collector.log",encoding="utf-8")])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default=None); ap.add_argument("--years",type=int); ap.add_argument("--refresh-days",type=int,default=10); ap.add_argument("--full",action="store_true"); args=ap.parse_args()
    setup_log(); cfg=Settings.load(args.config)
    if args.years: cfg.years=args.years
    out=Path(cfg.output_dir); out.mkdir(parents=True,exist_ok=True)
    store=Store(cfg.database); src=Sources(cfg)
    end=date.today(); start=end-timedelta(days=365*cfg.years+45)
    logging.info("下載 TAIEX 與 2330 月資料")
    core=src.taiex(start,end).merge(src.tsmc(start,end),on="date",how="outer").sort_values("date")
    external=src.external(start,end)
    cutoff=end-timedelta(days=max(args.refresh_days, 0))
    all_days=list(core.date.dropna().drop_duplicates().sort_values())
    # --full 才補抓整個 years 範圍；一般執行只更新 refresh-days 視窗。
    days=all_days if args.full else [d for d in all_days if d.date()>=cutoff]
    logging.info("本次待抓 %d 個交易日（模式：%s）", len(days), "完整回補" if args.full else f"最近 {args.refresh_days} 天")
    interrupted=False
    try:
        for i,day in enumerate(days,1):
            key=day.strftime("%Y-%m-%d"); row={"date":key,"updated_at":datetime.now().astimezone().isoformat(timespec="seconds")}
            c=core[core.date.eq(day)].iloc[-1]
            row["taiex"]=safe_float(c.get("taiex")); row["tsmc"]=safe_float(c.get("tsmc"))
            try: row.update({k:safe_float(v) for k,v in src.margin(day).items()})
            except Exception as e: logging.warning("%s margin: %s",key,e)
            try: row["foreign_net_100m"]=safe_float(src.foreign(day))
            except Exception as e: logging.warning("%s foreign: %s",key,e)
            ext=external[external.date.eq(day)]
            if not ext.empty:
                for c in ["usdtwd","dxy","vix"]: row[c]=safe_float(ext.iloc[-1].get(c))
            store.upsert(row); logging.info("%d/%d %s saved",i,len(days),key); time.sleep(cfg.request_delay_seconds)
    except KeyboardInterrupt:
        interrupted=True
        logging.warning("收到 Ctrl+C，停止下載並輸出目前已保存的資料")
    df=store.frame()
    # 月資料及外部資料可補齊資料庫中未更新的欄位
    df=df.drop(columns=[c for c in ["taiex","tsmc","usdtwd","dxy","vix"] if c in df],errors="ignore").merge(core,on="date",how="outer").merge(external,on="date",how="left")
    df=enrich(df.sort_values("date")); recent=df[df.date.dt.date>=end-timedelta(days=365*cfg.years+5)].copy()
    if recent.empty: raise RuntimeError("沒有可輸出的資料")
    cols=["taiex","tsmc","maintenance_est","margin_balance_billion","foreign_net_100m","foreign_20d_sum_100m","usdtwd","dxy","vix","outflow_pressure_score","market_temperature","maintenance_percentile"]
    records=[{"date":r.date.strftime("%Y-%m-%d"),**{c:safe_float(r.get(c)) for c in cols}} for _,r in recent.iterrows()]
    last=recent.dropna(subset=["taiex"]).iloc[-1]
    payload={"meta":{"generated_at":datetime.now().astimezone().isoformat(timespec="seconds"),"timezone":"Asia/Taipei","maintenance_ratio_note":"Estimated market-wide proxy derived from listed margin balances and closing prices; not an official TWSE account maintenance ratio.","disclaimer":"資料僅供研究與資訊用途，不構成投資建議。"},"latest_date":last.date.strftime("%Y-%m-%d"),"latest":{c:safe_float(last.get(c)) for c in cols},"summary":commentary(last),"backtests":backtests(recent,cfg.maintenance_thresholds),"series":records}
    (out/"market-dashboard.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    recent.to_csv(out/"market-dashboard.csv",index=False,encoding="utf-8-sig")
    logging.info("%s：%s", "中斷後已輸出" if interrupted else "完成", out/"market-dashboard.json")
if __name__=="__main__": main()
