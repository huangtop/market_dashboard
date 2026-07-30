from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(slots=True)
class Settings:
    years: int = 5
    output_dir: str = "public/data"
    database: str = "public/data/market_dashboard.sqlite3"
    request_delay_seconds: float = 0.35
    request_timeout_seconds: int = 40
    max_retries: int = 4
    maintenance_thresholds: list[float] = field(default_factory=lambda: [140,145,150,155])
    wordpress_json_url: str = "/wp-content/uploads/market-dashboard/market-dashboard.json"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"找不到設定檔：{p}")
        raw = json.loads(p.read_text(encoding="utf-8"))
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in raw.items() if k in allowed})
