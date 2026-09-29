from pathlib import Path
from unittest.mock import Mock

import pandas as pd
from streamlit.testing.v1 import AppTest
import pytest

import app_services
import analysis_engine
from market_data import DataError
from test_analysis import history

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def index_fetch(monkeypatch):
    fetch = Mock(return_value=history())
    monkeypatch.setattr(app_services, "get_index_history", fetch)
    return fetch


def test_page_starts_with_four_index_cards_only(monkeypatch, index_fetch):
    fetch = Mock(side_effect=AssertionError("Startup must not fetch stocks or the universe"))
    monkeypatch.setattr(app_services, "get_history", fetch)
    monkeypatch.setattr(app_services, "get_constituents", fetch)
    app = AppTest.from_file(APP).run()
    assert not app.exception
    fetch.assert_not_called()
    assert index_fetch.call_count == 4
    content = "".join(item.value for item in app.markdown)
    for name in ("沪深300", "上证指数", "创业板指", "中证500", "模型架构简介", "操作手册"):
        assert name in content
    assert "market-card" in content


def test_successful_report_survives_widget_rerun(monkeypatch):
    monkeypatch.setattr(app_services, "get_history", lambda *a, **k: history())
    app = AppTest.from_file(APP).run()
    app.text_input[0].set_value("600519")
    next(button for button in app.button if button.label == "开始诊断").click().run(timeout=30)
    assert not app.exception
    assert len(app.session_state["single_report"]["rows"]) == 1
    next(select for select in app.selectbox if select.label == "行情源").select("东方财富").run()
    assert not app.exception
    assert len(app.session_state["single_report"]["rows"]) == 1
    assert app.session_state["single_report"]["provider"] == "自动切换"


def test_offline_page_displays_reasons(monkeypatch):
    monkeypatch.setattr(app_services, "get_history", Mock(side_effect=DataError("腾讯：请求超时；东方财富：HTTP 429")))
    app = AppTest.from_file(APP).run()
    app.text_input[0].set_value("600519")
    next(button for button in app.button if button.label == "开始诊断").click().run(timeout=30)
    assert not app.exception
    report = app.session_state["single_report"]
    assert not report["rows"]
    assert "429" in report["errors"][0]["原因"]
    assert app.error


def test_constituent_failure_is_visible(monkeypatch):
    monkeypatch.setattr(app_services, "get_constituents", Mock(side_effect=DataError("成分名单不完整")))
    app = AppTest.from_file(APP).run()
    next(button for button in app.button if button.label == "开始沪深300扫描").click().run()
    assert not app.exception
    assert any("不完整" in error.value for error in app.error)


def test_successful_small_scan(monkeypatch):
    monkeypatch.setattr(app_services, "get_history", lambda *a, **k: history())
    monkeypatch.setattr(app_services, "get_constituents", lambda: {
        "frame": pd.DataFrame({"代码": ["600519.SS", "000001.SZ"], "名称": ["", ""]}),
        "source": "测试", "as_of": "20260921", "fetched_at": "测试时间", "notes": []})
    app = AppTest.from_file(APP).run()
    next(button for button in app.button if button.label == "开始沪深300扫描").click().run(timeout=30)
    assert not app.exception
    assert len(app.session_state["scan_report"]["rows"]) == 2


def test_compatibility_entry_reruns(monkeypatch):
    monkeypatch.setattr(app_services, "get_history", lambda *a, **k: history())
    entry = str(Path(APP).with_name("streamlit_app.py"))
    app = AppTest.from_file(entry).run(timeout=30)
    app.text_input[0].set_value("600519")
    next(button for button in app.button if button.label == "开始诊断").click().run(timeout=30)
    assert not app.exception
    assert len(app.session_state["single_report"]["rows"]) == 1
    app.run(timeout=30)
    assert not app.exception
    assert app.dataframe


def test_failed_requests_are_retryable_and_success_is_cached(monkeypatch):
    fetch = Mock(side_effect=[DataError("temporary"), history()])
    app_services.get_history.clear()
    monkeypatch.setattr(app_services, "load_history", fetch)
    with pytest.raises(DataError):
        app_services.get_history("600519.SS")
    app_services.get_history("600519.SS")
    app_services.get_history("600519.SS")
    assert fetch.call_count == 2
    app_services.get_history.clear()


def test_scan_stops_on_failure_and_can_resume(monkeypatch):
    codes = [f"60000{i}.SS" for i in range(9)]
    monkeypatch.setattr(app_services, "get_history", Mock(side_effect=DataError("offline")))
    monkeypatch.setattr(app_services, "get_constituents", lambda: {
        "frame": pd.DataFrame({"代码": codes, "名称": [""] * 9}),
        "source": "测试", "as_of": "20260921", "fetched_at": "测试时间", "notes": []})
    app = AppTest.from_file(APP).run()
    next(button for button in app.button if button.label == "开始沪深300扫描").click().run(timeout=30)
    report = app.session_state["scan_report"]
    assert len(report["errors"]) == 6 and len(report["pending"]) == 3
    monkeypatch.setattr(app_services, "get_history", lambda *a, **k: history())
    next(button for button in app.button if button.label == "继续扫描").click().run(timeout=30)
    assert not app.exception
    report = app.session_state["scan_report"]
    assert len(report["rows"]) == 3 and not report["pending"]


def test_all_300_are_scanned_and_ranked_before_display_filter(monkeypatch):
    codes = [f"600{i:03d}.SS" for i in range(300)]
    data = history()
    fetch = Mock(return_value=data)
    monkeypatch.setattr(app_services, "get_history", fetch)
    monkeypatch.setattr(app_services, "get_constituents", lambda: {
        "frame": pd.DataFrame({"代码": codes, "名称": codes}), "source": "测试",
        "as_of": "20260928", "fetched_at": "测试时间", "notes": []})
    # Exercise the actual scan and UI for all 300; modeling math has separate tests.
    monkeypatch.setattr(analysis_engine, "analyze", lambda history, ticker, *args: {
        "名称": ticker, "代码": ticker, "综合评分": float(ticker[:6]), "评分日期": "2026-09-25"})
    app = AppTest.from_file(APP).run()
    assert not any(select.label == "本次扫描数量" for select in app.selectbox)
    next(button for button in app.button if button.label == "开始沪深300扫描").click().run(timeout=60)
    assert not app.exception
    report = app.session_state["scan_report"]
    assert len(report["checked"]) == 300 and len(report["rows"]) == 300 and not report["pending"]
    assert {call.args[0] for call in fetch.call_args_list} == set(codes)
    ranking = next(item.value for item in app.dataframe if "名次" in item.value.columns)
    assert len(ranking) == 300
    assert ranking["名次"].tolist() == list(range(1, 301))
    assert ranking["代码"].iloc[0] == codes[-1]
    calls_before = fetch.call_count
    next(radio for radio in app.radio if radio.label == "排名显示范围").set_value("前20名").run()
    assert not app.exception
    visible = next(item.value for item in app.dataframe if "名次" in item.value.columns)
    assert len(visible) == 20
    assert len(app.session_state["scan_report"]["rows"]) == 300
    assert fetch.call_count == calls_before


def test_failed_stock_can_be_retried_without_losing_success(monkeypatch):
    attempts = {}
    data = history()
    def fetch(ticker, *args):
        attempts[ticker] = attempts.get(ticker, 0) + 1
        if ticker == "600519.SS" and attempts[ticker] == 1:
            raise DataError("temporary outage")
        return data
    monkeypatch.setattr(app_services, "get_history", fetch)
    app = AppTest.from_file(APP).run()
    app.text_input[0].set_value("000001 600519")
    next(button for button in app.button if button.label == "开始诊断").click().run(timeout=30)
    assert len(app.session_state["single_report"]["rows"]) == 1
    next(button for button in app.button if button.label == "重试失败标的").click().run(timeout=30)
    report = app.session_state["single_report"]
    assert not app.exception
    assert len(report["rows"]) == 2 and not report["errors"]
    assert attempts == {"000001.SZ": 1, "600519.SS": 2}
