from datetime import timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import requests
from xlrd import XLRDError

from market_data import (DataError, MarketClient, clean_history, normalize_a_share_code,
                         normalize_constituent_frame, now_cn, parse_codes)


def bars(n=120):
    end = pd.offsets.BDay().rollback(pd.Timestamp(now_cn().date() - timedelta(days=1)))
    dates = pd.bdate_range(end=end, periods=n)
    close = 100 + np.sin(np.arange(n) / 4) * 8 + np.arange(n) * 0.03
    return pd.DataFrame({"Open": close, "High": close + 2, "Low": close - 2,
                         "Close": close, "Volume": 1000.0}, index=dates)


@pytest.mark.parametrize("raw,expected", [
    ("600519", "600519.SS"), ("600519.SH", "600519.SS"), ("sh600519", "600519.SS"),
    ("000001.SZ", "000001.SZ"), (1.0, "000001.SZ"), ("510300", "510300.SS"),
    ("INVALID.SZ", None), ("600519.SZ", None), ("sz600519.SH", None),
    ("000000", None), ("900901", None), ("AAPL", None), (None, None), ("6005199", None),
])
def test_codes(raw, expected):
    assert normalize_a_share_code(raw) == expected


def test_input_separators_dedup_and_bad_symbols():
    valid, invalid = parse_codes("600519，sh600519；000001.SZ AAPL")
    assert valid == ["600519.SS", "000001.SZ"]
    assert invalid == ["AAPL"]


def test_empty_constituents_no_crash():
    assert normalize_constituent_frame(None).empty
    assert normalize_constituent_frame(pd.DataFrame()).empty
    assert normalize_constituent_frame(pd.DataFrame({"unknown": [1]})).empty


def test_bilingual_constituents_keep_codes_and_names():
    raw = pd.DataFrame({"成份券代码Constituent Code": [1.0, 600519, None, "bogus", 1],
                        "成份券名称Constituent Name": ["平安银行", "贵州茅台", "", "", "重复"]})
    result = normalize_constituent_frame(raw)
    assert result["代码"].tolist() == ["000001.SZ", "600519.SS"]
    assert result["名称"].tolist() == ["平安银行", "贵州茅台"]


def test_clean_history_does_not_invent_prices_or_volume():
    data = bars(100)
    data.iloc[-3, data.columns.get_loc("Close")] = np.nan
    data.iloc[-2, data.columns.get_loc("Volume")] = 0
    data.iloc[-1, data.columns.get_loc("High")] = 1
    cleaned = clean_history(pd.concat([data.iloc[:1], data.iloc[::-1]]))
    assert len(cleaned) == 98
    assert cleaned.index.is_monotonic_increasing
    assert cleaned.index.is_unique
    assert cleaned["Volume"].iloc[-1] == 0


def test_fallback_on_timeout_and_preserve_reason(monkeypatch):
    client = MarketClient()
    monkeypatch.setattr(client, "tencent", Mock(side_effect=requests.Timeout("slow")))
    monkeypatch.setattr(client, "eastmoney", Mock(return_value=(bars(), "测试", "前复权")))
    result = client.history("600519.SS")
    assert result.source == "东方财富"
    assert "超时" in result.notes[0]
    client.close()


def test_both_sources_fail_with_details(monkeypatch):
    client = MarketClient()
    monkeypatch.setattr(client, "tencent", Mock(return_value=(pd.DataFrame(), "", "前复权")))
    monkeypatch.setattr(client, "eastmoney", Mock(side_effect=requests.ConnectionError("offline")))
    with pytest.raises(DataError) as caught:
        client.history("600519.SS")
    assert "腾讯" in str(caught.value) and "东方财富" in str(caught.value)
    client.close()


def test_stale_source_triggers_alternate(monkeypatch):
    client = MarketClient()
    old = bars()
    old.index -= pd.Timedelta(days=50)
    monkeypatch.setattr(client, "tencent", Mock(return_value=(old, "测试", "前复权")))
    monkeypatch.setattr(client, "eastmoney", Mock(return_value=(bars(), "测试", "前复权")))
    assert client.history("600519.SS").source == "东方财富"
    client.close()


def test_index_does_not_become_shenzhen_stock(monkeypatch):
    client = MarketClient()
    fake = Mock(return_value=(bars(), "上证指数", "不复权"))
    monkeypatch.setattr(client, "tencent", fake)
    assert client.history("000001.SS", "index").name == "上证指数"
    fake.assert_called_once_with("000001.SS", "index")
    with pytest.raises(DataError):
        client.history("000001.SS", "stock")
    client.close()


def test_http_timeout_and_retry_are_bounded(monkeypatch):
    client = MarketClient()
    response = Mock(content=b"{}")
    fake = Mock(return_value=response)
    monkeypatch.setattr(client.session, "get", fake)
    client.get("https://example.com")
    assert fake.call_args.kwargs["timeout"] == (3.05, 8)
    retry = client.session.get_adapter("https://").max_retries
    assert retry.total == 1 and retry.read == 0 and not retry.respect_retry_after_header
    client.close()


def test_partial_constituent_list_is_rejected(monkeypatch):
    client = MarketClient()
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: pd.DataFrame({"code": ["600519"]}))
    response = Mock(content=b"xls")
    response.json.return_value = [{"code": "600519", "name": "贵州茅台"}]
    monkeypatch.setattr(client, "get", lambda *a, **k: response)
    with pytest.raises(DataError, match="完整性检查"):
        client.constituents()
    client.close()


@pytest.mark.parametrize("payload", [[], {"code": 0, "data": ["changed"]},
                                    {"code": 0, "data": {"sh600519": {"qfqday": [["short"]]}}}])
def test_unexpected_json_schema_falls_back(monkeypatch, payload):
    client = MarketClient()
    response = Mock()
    response.json.return_value = payload
    monkeypatch.setattr(client, "get", Mock(return_value=response))
    monkeypatch.setattr(client, "eastmoney", Mock(return_value=(bars(), "测试", "前复权")))
    result = client.history("600519.SS")
    assert result.source == "东方财富"
    assert "格式异常" in result.notes[0]
    client.close()


def test_corrupt_constituent_workbook_uses_sina(monkeypatch):
    client = MarketClient()
    monkeypatch.setattr(pd, "read_excel", Mock(side_effect=XLRDError("corrupt workbook")))

    def get(url, **kwargs):
        response = Mock(content=b"invalid workbook")
        if "params" in kwargs:
            start = (kwargs["params"]["page"] - 1) * 100
            response.json.return_value = [{"code": str(600000 + i), "name": f"stock{i}"}
                                          for i in range(start, start + 100)]
        return response

    monkeypatch.setattr(client, "get", get)
    result = client.constituents()
    assert result["source"] == "新浪"
    assert len(result["frame"]) == 300
    assert "XLRDError" in result["notes"][0]
    client.close()
