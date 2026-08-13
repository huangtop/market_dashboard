from types import SimpleNamespace

import pytest

from market_dashboard import utils


class FakeResponse:
    def __init__(self, status_code, text, content_type="application/json"):
        self.status_code = status_code
        self.text = text
        self.headers = {"content-type": content_type}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        import json
        return json.loads(self.text)


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return next(self.responses)


def test_get_json_falls_back_to_second_twse_host(monkeypatch):
    monkeypatch.setattr(utils.time, "sleep", lambda *_: None)

    first = FakeResponse(
        200,
        "<html>因為安全性考量，您所執行的頁面無法呈現。</html>",
        "text/html; charset=utf-8",
    )
    second = FakeResponse(200, '{"stat":"OK","data":[1]}', "application/json")

    session = FakeSession([first, second])
    data = utils.get_json(session, "marginTrading/MI_MARGN", {}, 5, 1)

    assert data["stat"] == "OK"
    assert session.urls[0].startswith("https://www.twse.com.tw/")
    assert session.urls[1].startswith("https://wwwc.twse.com.tw/")
