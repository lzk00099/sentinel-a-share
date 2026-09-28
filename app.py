from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import time

import pandas as pd
import streamlit as st

from analysis_engine import AnalysisError, analyze, completed_bars
from app_services import clear_data_cache, get_constituents, get_history
from market_data import DataError, INDICES, now_cn, parse_codes

LOG = logging.getLogger(__name__)
st.set_page_config(page_title="SENTINEL · A股观察台", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container {max-width:1500px;padding-top:2rem;}
.sentinel-hero {padding:24px 28px;border:1px solid #7c632c;border-radius:16px;
 background:linear-gradient(120deg,#171b28,#36221c);margin-bottom:24px;color:#faf8f0;}
.sentinel-hero h1 {font-size:2rem;margin:0 0 8px;color:#e9c87d;}
.sentinel-hero p {margin:0;color:#c1c7d5;}
</style>""", unsafe_allow_html=True)
st.markdown("""<div class="sentinel-hero"><h1>SENTINEL · A股观察台</h1>
<p>行情有来源，结果有日期，异常可追溯。</p></div>""", unsafe_allow_html=True)


def safe_fetch(args):
    ticker, kind, provider = args
    try:
        return ticker, get_history(ticker, kind, provider), None
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
    details, positive = [], 0
    benchmark = None
    for ticker, item in history.items():
        bars = completed_bars(item.frame)
        if len(bars) < 20 or (now_cn().date() - bars.index[-1].date()).days > 7:
            errors.append({"代码": ticker, "阶段": "指数校验", "原因": "指数过旧或日线不足"})
            continue
        if ticker == "000300.SS":
            benchmark = bars
        close = float(bars["Close"].iloc[-1])
        ma20 = float(bars["Close"].tail(20).mean())
        above = close > ma20
        positive += (2 if ticker in ("000300.SS", "000001.SS") else 1) * above
        details.append({"指数": INDICES[ticker], "收盘": close, "MA20": ma20,
                        "趋势": "高于 MA20" if above else "低于 MA20",
                        "日期": str(bars.index[-1].date()), "来源": item.source})
    weight, status = 1.0, "四指数未齐全，环境调整停用（乘数 1.0）"
    if len(details) == 4 and len({row["日期"] for row in details}) == 1:
        ratio = positive / 6
        weight = 1.3 if ratio >= 0.8 else 1.0 if ratio >= 0.5 else 0.7
        status = "多数指数高于均线" if ratio >= 0.8 else "指数趋势分化" if ratio >= 0.5 else "多数指数低于均线"
    elif len(details) == 4:
        status = "指数日期不一致，环境调整停用（乘数 1.0）"
    return {"details": details, "errors": errors, "weight": weight, "status": status,
            "benchmark": benchmark}


def run_report(tickers, provider, previous=None):
    started = time.monotonic()
    report = previous or {"rows": [], "errors": [], "histories": {}, "total": len(tickers),
                          "created": now_cn().strftime("%Y-%m-%d %H:%M:%S %Z"), "provider": provider}
    with st.spinner("读取并核对四个基准指数…"):
        env = report.get("environment") or market_environment(provider)
    report["environment"] = env
    progress = st.progress(0, text="准备读取个股行情")
    pending, failures, processed = [], 0, 0
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
                progress.progress(processed / len(tickers), text=f"本轮已检查 {processed}/{len(tickers)} · {ticker}")
            if failures >= 6 or time.monotonic() - started >= 180:
                pending = tickers[offset + len(batch):]
                if pending:
                    st.warning(f"本轮已保存结果，尚有 {len(pending)} 只待检查。可点击下方“继续扫描”。")
                break
    report["pending"] = pending
    report["updated"] = now_cn().strftime("%Y-%m-%d %H:%M:%S %Z")
    progress.empty()
    return report


def render_report(report, key):
    env = report["environment"]
    st.caption(f"本次结果生成：{report['updated']} · 请求设置：{report['provider']} · 缓存最长 10 分钟")
    cols = st.columns(4)
    cols[0].metric("计划标的", report["total"])
    cols[1].metric("取得行情", len(report["histories"]))
    cols[2].metric("完成评分", len(report["rows"]))
    cols[3].metric("待继续", len(report["pending"]))
    with st.expander("查看市场环境", expanded=False):
        st.write(f"{env['status']} · 评分乘数 {env['weight']:.2f}")
        if env["details"]:
            st.dataframe(pd.DataFrame(env["details"]), hide_index=True, width="stretch")
        if env["errors"]:
            st.dataframe(pd.DataFrame(env["errors"]), hide_index=True, width="stretch")
    if report["rows"]:
        frame = pd.DataFrame(report["rows"]).sort_values("综合评分", ascending=False)
        configs = {column: st.column_config.NumberColumn(format="%.2f%%")
                   for column in frame.columns if column.endswith("(%)")}
        st.dataframe(frame, hide_index=True, width="stretch", column_config=configs)
        st.download_button("下载本次诊断 CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                           file_name="sentinel_results.csv", mime="text/csv", key=f"export_{key}")
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
        st.session_state[key] = run_report(report["pending"], report["provider"], report)
        st.rerun()


with st.sidebar:
    st.subheader("数据与运行")
    provider = st.selectbox("行情源", ["自动切换", "腾讯", "东方财富"])
    st.caption("自动模式先读取腾讯，失败后尝试东方财富。只缓存成功响应。")
    if st.button("清除行情缓存", width="stretch"):
        clear_data_cache()
        st.success("已清除缓存；下次诊断将重新读取。已有报告保留原始时间。")
    st.divider()
    st.markdown("**日线观察规则**")
    st.write("支持沪深 A 股及常见场内基金。评分使用完整日线；北京时间 15:15 前排除当日未收盘日线。")
    st.write("突破概率是未校准的模型输出。参考位和盈亏分值用于比较，不代表已验证胜率或可实现收益。")
    st.caption("启动页面不会请求全市场股票字典。名称随行情返回，降低首次加载耗时。")

single, scan, help_tab = st.tabs(["单股诊断", "沪深300扫描", "数据说明"])
with single:
    st.subheader("输入代码，检查行情与信号")
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
            st.session_state["single_report"] = run_report(codes[:5], provider)
        else:
            st.error("请输入有效的沪深 A 股／场内基金代码。")
    if "single_report" in st.session_state:
        render_report(st.session_state["single_report"], "single_report")

with scan:
    st.subheader("核验成分名单后再扫描")
    st.caption("成分名单必须包含 300 只唯一股票。接口失败时会明确停止，不会用少数示例股票冒充完整名单。")
    limit = st.selectbox("本次扫描数量", [30, 100, 300], help="按代码排序选择；30 只用于先检查接口，并非 Top 30 推荐。")
    if st.button("开始沪深300扫描", type="primary"):
        try:
            with st.spinner("获取并校验成分股名单…"):
                constituents = get_constituents()
            codes = constituents["frame"].sort_values("代码")["代码"].head(limit).tolist()
            st.session_state["scan_metadata"] = constituents
            st.session_state["scan_report"] = run_report(codes, provider)
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
扫描每轮最多运行约 3 分钟（会等待当前批次完成），结果保留后可继续；连续 6 只读取失败则提前停止本轮。
""")
