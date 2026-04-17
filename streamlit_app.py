import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
from sklearn.ensemble import RandomForestClassifier
import warnings
import time

# --- 基础配置 ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="SENTINEL A-Share V24.2", layout="wide")

# --- 1. CSS 样式 ---
def get_v24_css():
    return """
    <style>
        .main-header { background: linear-gradient(135deg, #800000 0%, #333 100%); color: #ffd700; padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 25px; }
        .env-card { background: #1a1a1a; color: #ffd700; border: 1px solid #444; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .stat-box { background: #262626; padding: 12px; border-radius: 8px; text-align: center; border-bottom: 3px solid #ffd700; }
        .sidebar-box { background: #ffffff; padding: 15px; border-radius: 8px; border-left: 5px solid #cc0000; margin-bottom: 15px; color: #333; }
    </style>
    """

# --- 2. 增强容错的数据获取 ---
@st.cache_data(ttl=600, show_spinner=False)
def get_a_market_snapshot():
    """带限时策略的市场快照获取"""
    try:
        # 增加极速请求，避免大规模计算
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty: return pd.DataFrame()
        df['ticker'] = df['代码'].apply(lambda x: f"{x}.SS" if x.startswith('60') or x.startswith('68') else f"{x}.SZ")
        return df.set_index('ticker')[['名称', '换手率', '主力净流入-净占比']]
    except Exception as e:
        # 如果 akshare 挂了，返回空表而不报错
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_hs300_pool():
    try:
        df_300 = ak.index_stock_cons_csindex(symbol="000300")
        return df_300['成分券代码'].tolist()
    except:
        return ["600519", "300750", "601318", "000858", "600036", "600276"]

# --- 3. 核心诊断引擎 ---
def diagnostic_engine_a(ticker, snapshot, risk_weight):
    try:
        # 提取指标，如果快照为空则使用默认值
        name = snapshot.loc[ticker, '名称'] if not snapshot.empty and ticker in snapshot.index else ticker
        turnover = snapshot.loc[ticker, '换手率'] if not snapshot.empty and ticker in snapshot.index else 0
        money_flow = snapshot.loc[ticker, '主力净流入-净占比'] if not snapshot.empty and ticker in snapshot.index else 0
        
        # 限制历史数据长度以提速
        df = yf.download(ticker, period="180d", progress=False, auto_adjust=True, timeout=10)
        if df is None or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 因子计算
        df['Ret'] = df['Close'].pct_change()
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Bias'] = (df['Close'] - df['MA20']) / df['MA20']
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        
        change = df['Close'].diff()
        gain = (change.where(change > 0, 0)).rolling(14).mean()
        loss = (-change.where(change < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        
        # 机器学习预测
        df['Target'] = (df['High'].shift(-5).rolling(5).max() > df['Close'] * 1.05).astype(int)
        feats = ['Vol_Ratio', 'Bias', 'RSI']
        train = df[feats + ['Target']].dropna()
        
        if len(train) < 20: return None
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42) # 减小树的数量提速
        rf.fit(train[feats].iloc[:-5].values, train['Target'].iloc[:-5].values)
        
        win_p = float(rf.predict_proba(df[feats].iloc[[-1]].values)[0][1])
        
        # 评分
        flow_bonus = 1.1 if money_flow > 0 else 0.9
        ev = (win_p * 0.07) - ((1 - win_p) * 0.04) # 稍微保守的EV
        score = win_p * ev * risk_weight * flow_bonus * 1000
        
        curr_price = df['Close'].iloc[-1]
        return {
            '代码': ticker, '名称': name, '现价': round(curr_price, 2),
            '换手率%': round(turnover, 2), '主力资金%': round(money_flow, 2),
            '预测胜率': win_p, 
            '期望值%': round(ev * 100, 2), 
            '止盈参考': round(curr_price + df['ATR'].iloc[-1] * 2.2, 2), 
            '止损建议': round(curr_price - df['ATR'].iloc[-1] * 1.5, 2),
            '综合评分': round(score, 2)
        }
    except: return None

# --- 4. 界面渲染 ---
def display_styled_df(results_list):
    if not results_list:
        st.warning("🔎 未发现符合条件的标的，或数据接口响应超时。")
        return
    
    df = pd.DataFrame(results_list).sort_values('综合评分', ascending=False)
    df['预测胜率'] = df['预测胜率'].map('{:.1%}'.format)

    def color_val(val):
        color = '#d32f2f' if val < 0 else '#2e7d32'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        df.style.applymap(color_val, subset=['期望值%', '综合评分'])
        .background_gradient(subset=['综合评分'], cmap='RdYlGn', vmin=-5, vmax=15),
        use_container_width=True
    )

# --- 5. 主程序 ---
st.markdown(get_v24_css(), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🛡️ SENTINEL A-Share V24.2</h1><p>多因子机器学习诊断系统 (极速容错版)</p></div>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown("### 🧬 核心说明")
    st.markdown("""<div class="sidebar-box">
    <b>数据降级逻辑:</b> 如果AkShare实时接口繁忙，系统将优先确保技术面诊断正常运行。<br>
    <b>响应提速:</b> 缩减了回测样本量和模型深度，确保云端秒开。</div>""", unsafe_allow_html=True)
    if st.button("清除缓存重启"):
        st.cache_data.clear()
        st.rerun()

# 顶部评估 (逻辑简化避免挂起)
st.write("🔄 正在初始化环境...")
snapshot = get_a_market_snapshot()

try:
    m_df = yf.download("000300.SS", period="60d", progress=False, timeout=5)
    if not m_df.empty:
        if isinstance(m_df.columns, pd.MultiIndex): m_df.columns = m_df.columns.get_level_values(0)
        m_close = m_df['Close'].iloc[-1]
        risk_weight = 1.2 if m_close > m_df['Close'].rolling(20).mean().iloc[-1] else 0.8
        st.markdown(f"""<div class="env-card"><div class="grid-2">
        <div class="stat-box"><small>沪深300</small><br><b>{m_close:.2f}</b></div>
        <div class="stat-box"><small>风控系数</small><br><b>{risk_weight}</b></div>
        </div></div>""", unsafe_allow_html=True)
    else: risk_weight = 1.0
except: risk_weight = 1.0

# 界面展示
tab1, tab2 = st.tabs(["🚀 权重扫描", "🔍 个股诊断"])

with tab1:
    if st.button("全量执行 (Top 30)"):
        pool_codes = get_hs300_pool()[:30] # 减少初次扫描数量确保速度
        results = []
        bar = st.progress(0)
        for i, c in enumerate(pool_codes):
            ticker = f"{c}.SS" if c.startswith('60') or c.startswith('68') else f"{c}.SZ"
            res = diagnostic_engine_a(ticker, snapshot, risk_weight)
            if res: results.append(res)
            bar.progress((i+1)/len(pool_codes))
        display_styled_df(results)

with tab2:
    codes_input = st.text_input("代码 (后缀必须: .SS 或 .SZ)", "600519.SS 300750.SZ 601318.SS")
    if st.button("开始诊断"):
        results = []
        tickers = codes_input.replace(',', ' ').split()
        for t in tickers:
            res = diagnostic_engine_a(t, snapshot, risk_weight)
            if res: results.append(res)
        display_styled_df(results)
