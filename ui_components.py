"""Presentation components for the red-and-gold SENTINEL dashboard."""
from html import escape

import pandas as pd
import streamlit as st

from market_data import INDICES

CSS = """<style>
.stApp {background:radial-gradient(ellipse at 90% 0%,#2d111a 0%,#0b0e15 45%);}
.block-container {max-width:1540px;padding-top:1.7rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#11141d,#1c1117);border-right:1px solid #493124;}
.main-header {background:linear-gradient(120deg,#11121b 0%,#530b17 74%,#740c19 100%);
 border:1px solid #a78843;border-radius:18px;padding:32px 34px;margin:0 0 22px;
 box-shadow:0 12px 36px #0005;position:relative;overflow:hidden;}
.main-header:after {content:'';position:absolute;right:-60px;top:-110px;width:310px;height:310px;
 border:1px solid #d8b65835;border-radius:50%;pointer-events:none;}
.eyebrow {font-size:.72rem;letter-spacing:.23em;color:#ccaf75;font-weight:700;}
.main-header h1 {color:#f1d080;font-size:2.2rem;line-height:1.25;margin:12px 0 8px;letter-spacing:.01em;}
.main-header p {color:#d4c5c7;line-height:1.7;margin:0;font-size:.95rem;}
.hero-chips {display:flex;gap:10px;flex-wrap:wrap;margin-top:20px;}
.hero-chips span {font-size:.75rem;border:1px solid #a7805266;background:#0b0d184d;border-radius:30px;padding:5px 13px;color:#e3c995;}
.section-label {font-size:.74rem;letter-spacing:.15em;color:#bb9564;margin:8px 0;}
.market-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:14px 0;}
.market-card {padding:20px 20px 15px;border:1px solid #3d3540;border-radius:14px;
 background:linear-gradient(145deg,#1b1d29,#11141e);box-shadow:0 4px 18px #0003;min-width:0;}
.market-name {color:#eee7dc;font-weight:700;font-size:1.03rem;}
.market-code {color:#9295a8;font-size:.7rem;margin-top:2px;letter-spacing:.08em;}
.market-price {font-size:1.9rem;font-weight:700;margin:14px 0 3px;line-height:1.1;font-variant-numeric:tabular-nums;}
.market-change {font-size:.86rem;}.up {color:#ff727e;}.down {color:#4bd6b0;}.muted {color:#999daf;}
.market-meta {font-size:.73rem;color:#a5a6b7;line-height:1.8;margin-top:7px;}
.trend-pill {display:inline-block;padding:3px 8px;border-radius:5px;font-size:.7rem;margin-top:9px;background:#e1b76b16;color:#e7c98a;}
.market-card svg {display:block;width:100%;height:36px;margin-top:12px;}
.environment-strip {display:flex;gap:18px;justify-content:space-between;align-items:center;flex-wrap:wrap;
 border:1px solid #80673555;border-radius:10px;background:#c2983810;padding:14px 18px;margin-bottom:12px;color:#d5c2a1;}
.environment-strip strong {color:#f2d180;}.environment-strip small {color:#a7a4ad;}
.side-card {border:1px solid #36313b;border-left:3px solid var(--accent,#c1a36b);border-radius:9px;
 background:#121722;margin:15px 0;padding:15px;color:#bac2d1;font-size:.82rem;line-height:1.85;}
.side-card h4 {color:var(--accent,#e6c680);font-size:.9rem;margin:0 0 9px;}
.side-card b {color:#e2decb;}.side-card ol,.side-card ul {padding-left:18px;margin:6px 0 0;}
.guide-banner {border:1px solid #513841;border-radius:12px;padding:17px 20px;background:linear-gradient(100deg,#33141d,#171a27);color:#cfbdc3;line-height:1.8;margin-bottom:18px;}
.guide-banner b {color:#ecc777;}
.leader-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:14px 0 20px;}
.leader-card {border:1px solid #756036;border-radius:12px;background:linear-gradient(135deg,#302719,#171a26);padding:17px 19px;}
.leader-rank {color:#e5c277;font-size:.73rem;letter-spacing:.1em;}.leader-card h4 {color:#f2e5c9;margin:8px 0 4px;}
.leader-card small {color:#aaa7b6;}.leader-score {color:#f1ce79;font-size:1.55rem;font-weight:700;float:right;}
div.stButton>button[kind="primary"],div.stFormSubmitButton>button {background:linear-gradient(110deg,#831327,#ac1c2e);color:#fff3d7;border:1px solid #c09a51;border-radius:9px;font-weight:700;}
div.stButton>button[kind="primary"]:hover,div.stFormSubmitButton>button:hover {background:#d8b765;color:#161318;border-color:#f1d88e;}
button[data-baseweb="tab"] {font-weight:600;padding:12px 18px;}
button[aria-selected="true"][data-baseweb="tab"] {color:#f0cc7e;}
[data-testid="stMetric"] {background:#171b27;border:1px solid #373040;border-radius:11px;padding:13px 17px;}
[data-testid="stMetricValue"] {color:#ebca82;}
@media(max-width:1000px) {.market-grid {grid-template-columns:repeat(2,minmax(0,1fr));}.main-header h1 {font-size:1.7rem;}}
@media(max-width:600px) {.leader-grid {grid-template-columns:1fr;}.market-price {font-size:1.5rem;}.market-card {padding:14px;}.main-header {padding:22px;}.hero-chips {gap:6px;}}
</style>"""


def render_header():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class="main-header"><div class="eyebrow">SENTINEL / MAINLAND EQUITY INTELLIGENCE</div>
<h1>🛡 SENTINEL A-SHARE ADVANCED <span style="font-size:.55em;color:#c8a973">V27</span></h1>
<p>A 股多周期量化观察台 · 大盘共振追踪 · 沪深300全量排名</p>
<div class="hero-chips"><span>四指数环境追踪</span><span>300只完整扫描</span><span>随机森林 × 量价特征</span><span>双源行情自动切换</span></div></div>""", unsafe_allow_html=True)


def render_sidebar_guide():
    st.markdown("""
<div class="side-card" style="--accent:#e2be71"><h4>📋 模型架构简介</h4>
以<b>随机森林分类器</b>分析个股量价特征，叠加<b>沪深300基准特征</b>与四指数环境权重。
流程为：读取日线 → 核验完整性 → 提取特征 → 估计突破概率 → 生成参考区间与排序分值。
<br>支持沪深 A 股与常见场内基金；单股诊断最多5只，沪深300扫描覆盖完整名单。</div>
<div class="side-card" style="--accent:#53caaa"><h4>🔬 核心指标解读</h4><ul>
<li><b>量比：</b>当日成交量相对近5日均量。</li>
<li><b>乖离率：</b>价格偏离20日均线的幅度。</li>
<li><b>RSI：</b>近14日涨跌强弱。</li>
<li><b>ATR：</b>包含隔夜跳空的14日平均真实波幅。</li>
<li><b>相对强弱／相关性：</b>与沪深300的10日表现差、20日相关性及基准乖离。</li></ul></div>
<div class="side-card" style="--accent:#d09aec"><h4>🏆 如何理解排名</h4>
<b>综合评分</b>结合模型突破概率、K线形态修正、波动区间与市场权重。默认从高到低排名；同分按代码排序。
<br>先扫描全部300只，再查看完整榜单或前20／50名。失败标的单独列出，不会被静默丢弃。
<br><b>概率与分值尚未经样本外校准</b>，用于筛选比较，不等同于已验证胜率或收益。</div>
<div class="side-card" style="--accent:#e5a467"><h4>⏱ 时间与观察周期</h4>
大盘区域在页面保持打开时约每60秒刷新，展示<b>日线接口最新快照</b>，并非逐笔实时行情。
<br>个股评分使用完整日线；北京时间<b>15:15前</b>排除当日未收盘K线。
<br>训练观察窗口为未来<b>5个交易日</b>。盘后或周末也可查看最近交易日结果。</div>
<div class="side-card" style="--accent:#71a7e8"><h4>🛠 操作手册</h4><ol>
<li>先看四个指数的涨跌、MA20位置与环境权重。</li>
<li>点击全量扫描，等待300只检查结束；结果按综合评分排名。</li>
<li>用单股诊断复查目标；展开日线图核对趋势、日期与来源。</li>
<li>下载完整排名或异常记录。网络连续失败时可继续剩余标的或重试失败项。</li></ol></div>
""", unsafe_allow_html=True)


def sparkline(values, color):
    if len(values) < 2:
        return ""
    lower, upper = min(values), max(values)
    spread = upper - lower or 1
    points = " ".join(f"{i * 240 / (len(values) - 1):.1f},{33 - (value - lower) / spread * 29:.1f}"
                      for i, value in enumerate(values))
    return f'<svg viewBox="0 0 240 36" aria-hidden="true"><polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>'


def market_cards_html(environment):
    by_code = {row["代码"]: row for row in environment["details"]}
    cards = []
    for ticker, name in INDICES.items():
        row = by_code.get(ticker)
        title = f'<div class="market-name">{escape(name)}</div><div class="market-code">{ticker}</div>'
        if row is None:
            cards.append(f'<div class="market-card">{title}<div class="market-price muted">—</div><div class="market-meta">暂未取得有效行情<br>可用“刷新大盘”重试</div></div>')
            continue
        change = row["涨跌幅(%)"]
        tone, color = ("up", "#ff727e") if change >= 0 else ("down", "#4bd6b0")
        cards.append(f'''<div class="market-card">{title}<div class="market-price {tone}">{row['点位']:,.2f}</div>
<div class="market-change {tone}">{change:+.2f}% <span class="muted">较上一日线收盘</span></div>
{sparkline(row['走势'], color)}<span class="trend-pill">{escape(row['趋势'])}</span>
<div class="market-meta">MA20 {row['MA20']:,.2f} · 完整日线 {escape(row['日期'])}<br>
行情 {escape(row['行情日期'])} · {escape(row['来源'])}<br>抓取 {escape(row['抓取时间'])}</div></div>''')
    return '<div class="market-grid">' + "".join(cards) + '</div>'


def render_market(environment):
    st.markdown(market_cards_html(environment), unsafe_allow_html=True)
    st.markdown(f'''<div class="environment-strip"><span>环境权重 <strong>× {environment['weight']:.2f}</strong></span>
<span>{escape(environment['status'])}</span><small>基于完整日线与MA20判断</small></div>''', unsafe_allow_html=True)
    chart = environment.get("chart")
    if chart is not None and not chart.empty:
        with st.expander("📈 展开四指数走势对比", expanded=False):
            days = st.radio("观察窗口", [20, 60, 120], index=1, horizontal=True, format_func=lambda v: f"近{v}个交易日")
            comparison = chart.tail(days).dropna(how="any")
            if not comparison.empty:
                st.line_chart(comparison.div(comparison.iloc[0]).mul(100),
                              color=["#edc56f", "#fa7b88", "#54c7ae", "#a799ef"][:len(comparison.columns)])
                st.caption("共同有效日期内，各指数以窗口首日=100归一化；用于比较走势，不是原始点位。")
    if environment["errors"]:
        with st.expander(f"大盘数据提示（{len(environment['errors'])}项）"):
            st.dataframe(pd.DataFrame(environment["errors"]), hide_index=True, width="stretch")


def ranked_frame(rows):
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).drop_duplicates("代码", keep="last")
    frame = frame.sort_values(["综合评分", "代码"], ascending=[False, True], kind="stable").reset_index(drop=True)
    frame.insert(0, "名次", range(1, len(frame) + 1))
    primary = ["名次", "名称", "代码", "综合评分", "最近价", "最近完整日涨跌(%)",
               "模型突破概率(%)", "形态修正值(%)", "启发式盈亏分值(%)", "评分基准价", "上方参考位", "下方参考位"]
    return frame[[column for column in primary if column in frame] + [column for column in frame if column not in primary]]


def render_leaders(frame):
    cards = []
    for row in frame.head(3).to_dict("records"):
        cards.append(f'''<div class="leader-card"><div class="leader-rank">RANK {row['名次']:02d}<span class="leader-score">{row['综合评分']:.2f}</span></div>
<h4>{escape(str(row['名称']))}</h4><small>{escape(row['代码'])} · 评分日期 {escape(row['评分日期'])}</small></div>''')
    st.markdown('<div class="leader-grid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


def styled_ranking(frame):
    formats = {column: "{:+.2f}%" for column in frame if column.endswith("(%)")}
    formats.update({"综合评分": "{:.3f}", "最近价": "{:.3f}", "评分基准价": "{:.3f}",
                    "上方参考位": "{:.3f}", "下方参考位": "{:.3f}"})
    scores = frame["综合评分"]
    lower, spread = scores.min(), scores.max() - scores.min() or 1
    def shade(value):
        opacity = 0.10 + 0.22 * (value - lower) / spread
        return f"background-color: rgba(211,169,78,{opacity:.2f}); color: #f6dfa2; font-weight: 700;"
    return frame.style.format({k: v for k, v in formats.items() if k in frame}).map(shade, subset=["综合评分"])
