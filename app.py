from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging

import pandas as pd
import streamlit as st

from analysis_engine import AnalysisError, analyze, completed_bars
from app_services import clear_data_cache, get_constituents, get_history, get_index_history
from market_data import DataError, INDICES, now_cn, parse_codes
from ui_components import (ranked_frame, render_header, render_leaders, render_market,
                           render_sidebar_guide, styled_ranking)

LOG = logging.getLogger(__name__)
st.set_page_config(page_title="SENTINEL A-SHARE ADVANCED V27", page_icon="🛡️", layout="wide")
render_header()


def safe_fetch(args):
    ticker, kind, provider = args
    try:
        data = get_index_history(ticker, provider) if kind == "index" else get_history(ticker, kind, provider)
        return ticker, data, None
    except DataError as error:
        return ticker, None, str(error)
    except Exception as error:
        LOG.exception("Unexpected data failure for %s", ticker)
        return ticker, None, f"程序异常：{type(error).__name__}: {str(error)[:160]}"


def market_environment(provider):
    history, errors = {}, []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for ticker, data, error in pool.map(safe_fetch, [(t, "index", provider) for t in INDICES]):
            if error:
                errors.append({"代码": ticker, "阶段": "指数读取", "原因": error})
            else:
                history[ticker] = data
    details, positive, series = [], 0, {}
    benchmark = None
    for ticker, item in history.items():
        bars = completed_bars(item.frame)
        if len(bars) < 20 or (now_cn().date() - bars.index[-1].date()).days > 7:
            errors.append({"代码": ticker, "阶段": "指数校验", "原因": "指数过旧或日线不足"})
            continue
        if ticker == "000300.SS":
            benchmark = bars
        series[INDICES[ticker]] = bars["Close"].tail(120)
        close = float(bars["Close"].iloc[-1])
        ma20 = float(bars["Close"].tail(20).mean())
        above = close > ma20
        positive += (2 if ticker in ("000300.SS", "000001.SS") else 1) * above
        latest = float(item.frame["Close"].iloc[-1])
        daily_change = (latest / float(item.frame["Close"].iloc[-2]) - 1) * 100
        details.append({"代码": ticker, "指数": INDICES[ticker], "收盘": close, "MA20": ma20,
                        "趋势": "高于 MA20" if above else "低于 MA20",
                        "日期": str(bars.index[-1].date()), "来源": item.source,
                        "点位": latest, "涨跌幅(%)": daily_change,
                        "行情日期": str(item.frame.index[-1].date()),
                        "抓取时间": item.fetched_at, "走势": bars["Close"].tail(30).tolist()})
    weight, status = 1.0, "四指数未齐全，环境调整停用（乘数 1.0）"
    if len(details) == 4 and len({row["日期"] for row in details}) == 1:
        ratio = positive / 6
        weight = 1.3 if ratio >= 0.8 else 1.0 if ratio >= 0.5 else 0.7
        status = "多数指数高于均线" if ratio >= 0.8 else "指数趋势分化" if ratio >= 0.5 else "多数指数低于均线"
    elif len(details) == 4:
        status = "指数日期不一致，环境调整停用（乘数 1.0）"
    return {"details": details, "errors": errors, "weight": weight, "status": status,
            "benchmark": benchmark, "chart": pd.concat(series, axis=1) if series else pd.DataFrame(),
            "checked_at": now_cn().strftime("%Y-%m-%d %H:%M:%S %Z"), "provider": provider}


def run_report(tickers, provider, previous=None, report_key=None):
    tickers = list(dict.fromkeys(tickers))
    report = previous or {"rows": [], "errors": [], "histories": {}, "total": len(tickers),
                          "created": now_cn().strftime("%Y-%m-%d %H:%M:%S %Z"), "provider": provider,
                          "checked": []}
    report["pending"] = list(tickers)
    report["updated"] = now_cn().strftime("%Y-%m-%d %H:%M:%S %Z")
    if report_key:
        st.session_state[report_key] = report
    with st.spinner("读取并核对四个基准指数…"):
        env = report.get("environment") or market_environment(provider)
    report["environment"] = env
    progress = st.progress(0, text="准备读取个股行情")
    failures, processed = 0, 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for offset in range(0, len(tickers), 3):
            batch = tickers[offset:offset + 3]
            for ticker, history, error in pool.map(safe_fetch, [(t, "stock", provider) for t in batch]):
                processed += 1
                if error:
                    failures += 1
                    report["errors"].append({"代码": ticker, "阶段": "行情读取", "原因": error})
                else:
                    failures = 0
                    report["histories"][ticker] = history
                    try:
                        report["rows"].append(analyze(history, ticker, env["benchmark"], env["weight"]))
                    except AnalysisError as error:
                        report["errors"].append({"代码": ticker, "阶段": "模型计算", "原因": str(error)})
                    except Exception as error:
                        LOG.exception("Unexpected model failure for %s", ticker)
                        report["errors"].append({"代码": ticker, "阶段": "模型计算",
                                                 "原因": f"{type(error).__name__}: {str(error)[:160]}"})
                if ticker not in report.setdefault("checked", []):
                    report["checked"].append(ticker)
                report["pending"] = tickers[processed:]
                report["updated"] = now_cn().strftime("%Y-%m-%d %H:%M:%S %Z")
                if report_key:
                    st.session_state[report_key] = report
                done = len(report["checked"])
                progress.progress(done / report["total"], text=f"已检查 {done}/{report['total']} · 已评分 {len(report['rows'])} · {ticker}")
            # Isolated failures do not shrink the universe. Only a sustained outage pauses it.
            if failures >= 6:
                if report["pending"]:
                    st.warning(f"连续6只读取失败，已保存进度。尚有 {len(report['pending'])} 只待检查，可点击“继续扫描”。")
                break
    report["updated"] = now_cn().strftime("%Y-%m-%d %H:%M:%S %Z")
    progress.empty()
    return report


def render_report(report, key):
    env = report["environment"]
    st.caption(f"本次结果生成：{report['updated']} · 请求设置：{report['provider']} · 缓存最长 10 分钟")
    cols = st.columns(4)
    cols[0].metric("计划标的", report["total"])
    cols[1].metric("已检查", len(report.get("checked", report["histories"])))
    cols[2].metric("完成评分", len(report["rows"]))
    cols[3].metric("待继续", len(report["pending"]))
    with st.expander("本次评分使用的市场环境快照", expanded=False):
        st.write(f"{env['status']} · 评分乘数 {env['weight']:.2f}")
        if env["details"]:
            snapshot = pd.DataFrame(env["details"]).drop(columns=["走势"], errors="ignore")
            st.dataframe(snapshot, hide_index=True, width="stretch")
        if env["errors"]:
            st.dataframe(pd.DataFrame(env["errors"]), hide_index=True, width="stretch")
    if report["rows"]:
        frame = ranked_frame(report["rows"])
        is_scan = key == "scan_report"
        if is_scan:
            if report["pending"]:
                st.warning(f"暂定排名：尚有 {len(report['pending'])} 只未检查，当前榜单只包含已完成评分的标的。")
            else:
                st.success(f"全名单检查完成：{report['total']}/{report['total']}；{len(frame)} 只成功评分并排名，{len(report['errors'])} 只存在异常。")
            render_leaders(frame)
            selection = st.radio("排名显示范围", ["全部排名", "前20名", "前50名"], horizontal=True, key=f"ranking_view_{key}")
            display = frame if selection == "全部排名" else frame.head(20 if selection == "前20名" else 50)
            st.caption(f"共 {len(frame)} 只可排名 · 当前显示 {len(display)} 只。名次基于完整已评分集合，默认按综合评分降序；点击表头可临时换序。")
        else:
            display = frame
        configs = {column: st.column_config.NumberColumn(format="%.2f%%")
                   for column in frame.columns if column.endswith("(%)")}
        st.dataframe(styled_ranking(display), hide_index=True, width="stretch", column_config=configs,
                     height=min(760, max(180, (len(display) + 1) * 35)))
        st.download_button("下载全部排名 CSV" if is_scan else "下载本次诊断 CSV",
                           frame.to_csv(index=False).encode("utf-8-sig"),
                           file_name="hs300_full_ranking.csv" if is_scan else "sentinel_results.csv",
                           mime="text/csv", key=f"export_{key}")
    else:
        st.error("本次没有可评分结果。下方列出了具体原因；非交易时段仍应可以读取历史日线。")
    if report["histories"]:
        with st.expander("查看日线与数据来源"):
            selected = st.selectbox("选择标的", list(report["histories"]), key=f"chart_{key}")
            history = report["histories"][selected]
            st.caption(f"{history.name} · {history.source} · {history.adjustment} · 抓取时间 {history.fetched_at}")
            for note in history.notes:
                st.info(note)
            st.line_chart(history.frame[["Close"]].tail(120))
            st.dataframe(history.frame.tail(10), width="stretch")
    if report["errors"]:
        with st.expander(f"查看 {len(report['errors'])} 条异常记录", expanded=not report["rows"]):
            errors = pd.DataFrame(report["errors"])
            st.dataframe(errors, hide_index=True, width="stretch")
            st.download_button("下载异常记录", errors.to_csv(index=False).encode("utf-8-sig"),
                               "sentinel_errors.csv", "text/csv", key=f"errors_{key}")
    if report["pending"] and st.button("继续扫描", key=f"continue_{key}"):
        st.session_state[key] = run_report(report["pending"], report["provider"], report, report_key=key)
        st.rerun()
    if report["errors"] and not report["pending"] and st.button("重试失败标的", key=f"retry_{key}"):
        retry_codes = list(dict.fromkeys(row["代码"] for row in report["errors"]))
        report["errors"] = []
        st.session_state[key] = run_report(retry_codes, report["provider"], report, report_key=key)
        st.rerun()


with st.sidebar:
    st.subheader("🧬 SENTINEL 决策面板")
    provider = st.selectbox("行情源", ["自动切换", "腾讯", "东方财富"])
    st.caption("自动模式先读取腾讯，失败后尝试东方财富。只缓存成功响应。")
    if st.button("清除行情缓存", width="stretch"):
        clear_data_cache()
        st.success("已清除缓存；下次诊断将重新读取。已有报告保留原始时间。")
    render_sidebar_guide()


@st.fragment(run_every="60s")
def market_tracker(provider):
    title, action = st.columns([5, 1])
    title.subheader("🇨🇳 境内四指数 · 大盘追踪")
    if action.button("刷新大盘", width="stretch"):
        get_index_history.clear()
    with st.spinner("正在读取大盘快照…"):
        environment = market_environment(provider)
    st.session_state["tracking_environment"] = environment
    render_market(environment)
    st.caption(f"页面打开时约每60秒自动刷新 · 本次检查 {environment['checked_at']} · 点位为日线接口最新快照，数据可能延迟。")


market_tracker(provider)

single, scan, help_tab = st.tabs(["单股诊断", "沪深300扫描", "数据说明"])
with single:
    st.subheader("🔍 A股单股精准诊断")
    st.markdown('<div class="guide-banner"><b>单股深度观察</b> · 输入最多5个股票或场内基金代码，查看突破概率、波动参考位与风险提示。<br>请同时核对<b>行情日期</b>与<b>评分日期</b>；盘中价格快照可能比用于评分的完整日线更新。</div>', unsafe_allow_html=True)
    with st.form("single_form"):
        text = st.text_input("股票／基金代码（最多 5 只）", "000807 002463 600183 002384 000630",
                             help="支持 600519、600519.SH、sh600519、000001.SZ；可用中英文逗号或空格分隔。")
        submitted = st.form_submit_button("开始诊断", type="primary")
    if submitted:
        codes, invalid = parse_codes(text)
        if invalid:
            st.warning("未识别或市场后缀不匹配：" + "、".join(invalid))
        if len(codes) > 5:
            st.warning("本次处理前 5 个唯一代码。")
        if codes:
            st.session_state["single_report"] = run_report(codes[:5], provider, report_key="single_report")
        else:
            st.error("请输入有效的沪深 A 股／场内基金代码。")
    if "single_report" in st.session_state:
        render_report(st.session_state["single_report"], "single_report")

with scan:
    st.subheader("🏆 沪深300 · 全量扫描与综合排名")
    st.markdown('<div class="guide-banner"><b>扫描范围：完整300只成分股</b> · 一次启动，逐只读取与建模，按综合评分统一排名。<br>榜单默认展示<b>全部排名</b>，也可切换前20／50名；显示范围不会改变实际扫描数量。</div>', unsafe_allow_html=True)
    st.caption("开始前校验300只唯一成分股。一般需要数分钟，页面会持续更新进度；个别失败会记录原因并继续其他股票。")
    if st.button("开始沪深300扫描", type="primary"):
        try:
            with st.spinner("获取并校验成分股名单…"):
                constituents = get_constituents()
            codes = constituents["frame"].sort_values("代码")["代码"].tolist()
            st.session_state["scan_metadata"] = constituents
            st.session_state["scan_report"] = run_report(codes, provider, report_key="scan_report")
        except DataError as error:
            st.error(str(error))
        except Exception as error:
            LOG.exception("Constituent retrieval failed")
            st.error(f"成分股读取异常：{type(error).__name__}: {str(error)[:160]}")
    if "scan_report" in st.session_state:
        meta = st.session_state["scan_metadata"]
        st.caption(f"名单来源：{meta['source']} · 名单标注日期：{meta['as_of']} · 抓取：{meta['fetched_at']}")
        for note in meta["notes"]:
            st.info(note)
        render_report(st.session_state["scan_report"], "scan_report")

with help_tab:
    st.markdown("""
**数据口径**

- 最近价是日线接口的最新快照；行情日期与评分日期分别展示。
- 价格序列优先采用前复权；指数采用不复权。每个标的使用一个来源的完整序列，成交量不跨源拼接。
- 未来 5 个交易日最高价超过评分基准价加 1.5 倍 ATR 为训练目标；最后 5 行未知标签不参与训练。
- ATR 包含隔夜跳空。停牌零成交量不会被伪造为正常成交量。
- 未获得齐全且日期一致的四个指数时，环境调整停用；基准特征缺失时降级为个股特征并提示。

**读取失败时**

先查看异常记录中的来源、阶段及原因；可清除缓存后重试，或选择另一行情源。
HTTP 429 通常表示限流，连接超时可能与云服务器网络有关。免费公开接口的可用性需要在实际部署环境核验。
全量扫描会持续处理全部300只，不再按30只或3分钟自动截断。仅连续6只读取失败时暂停，保留进度后可继续；结束后可重试失败标的。

**排名与显示范围**

默认对全部成功评分的股票按综合评分降序排名，同分按代码排序。前20／50名只影响榜单展示，不影响扫描范围或CSV导出。
若存在失败标的，完整报告会同时显示已检查数量、可排名数量和异常明细；尚未检查完的榜单明确标为暂定排名。

**四指数大盘追踪**

首页始终展示沪深300、上证指数、创业板指和中证500。点位与日涨跌使用最新日线快照；MA20状态和模型环境权重使用完整日线。
大盘刷新不会重跑个股模型。每份诊断报告保留当时用于评分的市场快照，便于复查。
""")
