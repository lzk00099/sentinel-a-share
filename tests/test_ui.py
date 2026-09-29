from ui_components import market_cards_html, ranked_frame


def test_ranking_is_global_and_ties_are_deterministic():
    rows = [{"代码": "600002.SS", "综合评分": 10}, {"代码": "600001.SS", "综合评分": 10},
            {"代码": "600003.SS", "综合评分": -1}, {"代码": "600003.SS", "综合评分": 20}]
    frame = ranked_frame(rows)
    assert frame["代码"].tolist() == ["600003.SS", "600001.SS", "600002.SS"]
    assert frame["名次"].tolist() == [1, 2, 3]


def test_unavailable_index_cards_remain_visible():
    html = market_cards_html({"details": []})
    assert html.count('class="market-card"') == 4
    assert "暂未取得有效行情" in html


def test_provider_text_is_escaped():
    html = market_cards_html({"details": [{"代码": "000300.SS", "涨跌幅(%)": 1.2,
        "点位": 4000, "走势": [1, 2, 3], "趋势": "高于 MA20", "MA20": 3900,
        "日期": "2026-09-25", "行情日期": "2026-09-28", "来源": "<script>alert(1)</script>",
        "抓取时间": "2026-09-28"}]})
    assert "<script>" not in html and "&lt;script&gt;" in html
