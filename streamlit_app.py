import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
from sklearn.ensemble import RandomForestClassifier
import warnings

# --- 基础配置 ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="SENTINEL A-Share V24", layout="wide")

# --- 1. A股专用 CSS 渲染（强化颜色表现）---
def get_v24_css():
    return """
    <style>
        .main-header { background: linear-gradient(135deg, #800000 0%, #333 100%); color: #ffd700; padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
        .env-card { background: #1a1a1a; color: #ffd700; border: 1px solid #444; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .stat-box { background: #262626; padding: 12px; border-radius: 8px; text-align: center; border-bottom: 3px solid #ffd700; }
        .sidebar-box { background: #ffffff; padding: 15px; border-radius: 8px; border-left: 5px solid #cc0000; margin-bottom: 15px; color: #333; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .u-tips { font-size: 0.85rem; color: #666; line-height: 1.5; }
        /* 表格颜色增强 */
        .dataframe th { background-color: #2c2c2c; color: #ffd700; }
        .dataframe td { font-size: 0.9rem; }
    </style>
    """

# --- 2. 辅助函数：获取A股股票名称（带缓存）---
@st.cache_data(ttl=3600)
def get_all_stock_names():
    """获取全市场A股代码->名称映射，用于手动诊断模块"""
    try:
        df = ak.stock_zh_a_spot()
        return dict(zip(df['代码'], df['名称']))
    except Exception:
        return {}

def get_stock_name(ticker):
    """根据ticker（如600519.SS）返回中文名称"""
    code = ticker.split('.')[0]
    name_map = get_all_stock_names()
    return name_map.get(code, ticker)  # 若获取失败则返回代码本身

# --- 3. 核心诊断引擎（保持不变）---
@st.cache_data(ttl=3600)
def get_a_shares_pool():
    try:
        df_300 = ak.index_stock_cons_csindex(symbol="000300")
        df_top = df_300.head(50)
        return { (f"{row['成分券代码']}.SS" if row['成分券代码'].startswith('60') else f"{row['成分券代码']}.SZ"): row['成分券名称']
                for _, row in df_top.iterrows() }
    except:
        return {"600519.SS": "贵州茅台", "300750.SZ": "宁德时代", "601318.SS": "中国平安"}

def diagnostic_engine_a(ticker, name, risk_weight):
    try:
        df = yf.download(ticker, period="250d", progress=False, auto_adjust=True)
        if len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df['Ret'] = df['Close'].pct_change()
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Bias'] = (df['Close'] - df['MA20']) / df['MA20']
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        atr_now = df['ATR'].iloc[-1]
        
        change = df['Close'].diff()
        gain = (change.where(change > 0, 0)).rolling(14).mean()
        loss = (-change.where(change < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        
        df['Target'] = (df['High'].shift(-5).rolling(5).max() > df['Close'] * 1.06).astype(int)
        
        feats = ['Vol_Ratio', 'Bias', 'RSI']
        train = df[feats + ['Target']].dropna()
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(train[feats].iloc[:-5].values, train['Target'].iloc[:-5].values)
        
        last_feat = df[feats].iloc[[-1]].values
        win_p = float(rf.predict_proba(last_feat)[0][1])
        
        ev = (win_p * 0.08) - ((1 - win_p) * 0.04)
        score = win_p * ev * risk_weight * 1000
        
        curr_price = df['Close'].iloc[-1]
        tp_price = curr_price + (atr_now * 2.5)
        sl_price = curr_price - (atr_now * 1.5)
        
        return {
            '代码': ticker, '名称': name, '现价': round(curr_price, 2),
            '预测胜率': f"{win_p:.1%}", 
            '期望值': f"{ev*100:+.2f}%", 
            '止盈参考': round(tp_price, 2), 
            '止损建议': round(sl_price, 2),
            '综合评分': round(score, 2),
            'Score_Raw': score 
        }
    except: return None

# --- 4. 统一表格样式函数（使两个引擎外观一致）---
def style_dataframe(df):
    """对综合评分列应用渐变，对期望值列根据正负设置颜色"""
    styled = df.style.background_gradient(subset=['综合评分'], cmap='RdYlGn', low=0, high=20)
    # 额外美化期望值列：正收益绿色，负收益红色
    def color_expect(val):
        if isinstance(val, str) and val.startswith('+'):
            return 'color: #00cc66'
        elif isinstance(val, str) and val.startswith('-'):
            return 'color: #ff4d4d'
        return ''
    styled = styled.applymap(color_expect, subset=['期望值'])
    return styled

# --- 5. Streamlit 界面布局 ---
st.markdown(get_v24_css(), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🛡️ SENTINEL A-Share V24</h1><p>沪深300核心资产量化扫描系统 • 机器学习增强版</p></div>', unsafe_allow_html=True)

# 侧边栏（保持不变）
with st.sidebar:
    st.markdown("### 🧬 系统逻辑简介")
    st.markdown("""
    <div class="sidebar-box">
    本系统核心为 <b>Hybrid-RF (混合随机森林)</b> 模型，专为 A 股 T+1 环境定制。
    <br><br>
    <b>核心因子：</b>
    <li><b>Bias (量价乖离):</b> 监控价格回归动能。</li>
    <li><b>RSI (强弱对比):</b> 评估超买超卖极端情绪。</li>
    <li><b>Vol_Ratio (量能修正):</b> 过滤无量假拉升。</li>
    <li><b>ATR (动态波幅):</b> 自动适配不同股性的震荡空间。</li>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🕒 最佳运行时间")
    st.markdown("""
    <div class="sidebar-box">
    <b>1. 盘前观察 (09:15-09:25):</b>
    查看大盘风控乘数，制定当日仓位基调。
    <br><br>
    <b>2. 盘中确认 (10:30 & 14:00):</b>
    此时量能比值最具参考性，适合捕捉趋势。
    <br><br>
    <b>3. 尾盘博弈 (14:45):</b>
    锁定次日 T+1 期望值，过滤单日波动噪声。
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ 操作手册")
    st.markdown("""
    <div class="u-tips">
    <li><b>评分 > 10:</b> 强信号，建议关注。</li>
    <li><b>评分 5-10:</b> 观察信号，需结合板块效应。</li>
    <li><b>评分 < 0:</b> 避险区域，模型判定期望值为负。</li>
    <br>
    <i>*注：止盈止损已根据个股 ATR 自动动态调整，无需人工计算。</i>
    </div>
    """, unsafe_allow_html=True)

# 顶部大盘环境评估
m_ticker = "000300.SS"
m_df = yf.download(m_ticker, period="100d", progress=False)

if not m_df.empty:
    if isinstance(m_df.columns, pd.MultiIndex): m_df.columns = m_df.columns.get_level_values(0)
    m_close = m_df['Close'].iloc[-1]
    m_ma20 = m_df['Close'].rolling(20).mean().iloc[-1]
    m_ma60 = m_df['Close'].rolling(60).mean().iloc[-1]
    
    risk_weight = 1.2 if m_close > m_ma20 else 0.8 if m_close > m_ma60 else 0.5
    m_status = "强势反弹 (进攻)" if m_close > m_ma20 else "缩量回调 (观察)" if m_close > m_ma60 else "空头趋势 (避险)"
    
    st.markdown(f"""
    <div class="env-card">
        <div style="margin-bottom:12px; font-weight:bold; font-size:1.1rem; border-left:4px solid #ffd700; padding-left:10px;">🚨 A股宏观情绪监测</div>
        <div class="grid-2">
            <div class="stat-box"><small>沪深300指数</small><br><b style="font-size:1.4rem;">{m_close:.2f}</b></div>
            <div class="stat-box"><small>战术环境建议</small><br><b style="font-size:1.4rem;">{m_status}</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    risk_weight = 0.8

# 功能标签页
tab1, tab2 = st.tabs(["🚀 核心资产 Top 50 扫描", "🔍 跨市场标的单兵诊断"])

DISPLAY_COLS = ['代码', '名称', '现价', '预测胜率', '期望值', '止盈参考', '止损建议', '综合评分']

with tab1:
    st.write("点击下方按钮，系统将实时抓取沪深 300 权重前 50 标的并运行机器学习引擎。")
    if st.button("开始全量量化扫描"):
        pool = get_a_shares_pool()
        results = []
        progress_bar = st.progress(0)
        for i, (t, n) in enumerate(pool.items()):
            res = diagnostic_engine_a(t, n, risk_weight)
            if res: results.append(res)
            progress_bar.progress((i + 1) / len(pool))
        
        if results:
            df_final = pd.DataFrame(results).sort_values('Score_Raw', ascending=False).head(12)
            st.subheader("🔥 SENTINEL 选股池 (高期望值标的)")
            # 统一使用带样式的 DataFrame
            st.dataframe(style_dataframe(df_final[DISPLAY_COLS]), use_container_width=True)
        else:
            st.warning("⚠️ 当前市场环境下，模型未发现具有正向期望值的标的，建议持币观望。")

with tab2:
    st.write("输入 A 股代码（含后缀），支持多代码批量诊断。")
    user_input = st.text_input("示例：600519.SS 300750.SZ 000001.SZ", "600519.SS 300750.SZ 601318.SS")
    if st.button("执行精准诊断"):
        tickers = user_input.replace(',', ' ').split()
        results = []
        for t in tickers:
            # 获取真实股票名称
            real_name = get_stock_name(t)
            res = diagnostic_engine_a(t, real_name, risk_weight)
            if res: results.append(res)
        
        if results:
            df_user = pd.DataFrame(results).sort_values('Score_Raw', ascending=False)
            st.subheader("📊 诊断报告")
            # 应用与全量扫描完全相同的表格样式
            st.dataframe(style_dataframe(df_user[DISPLAY_COLS]), use_container_width=True)
        else:
            st.error("数据抓取失败。请确保输入格式正确（如 600519.SS）。")
