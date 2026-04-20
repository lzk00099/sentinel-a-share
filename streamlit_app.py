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
st.set_page_config(page_title="SENTINEL A-Share V24 PRO", layout="wide")

# --- 1. CSS 渲染 (Safari 兼容补丁) ---
def get_v24_css():
    return """
    <style>
        .stApp { -webkit-overflow-scrolling: touch; }
        .main-header { 
            background: linear-gradient(135deg, #1a1a1a 0%, #800000 100%); 
            color: #ffd700; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 1px solid #ffd700; 
        }
        .env-card { background: #0e1117; color: #ffd700; border: 1px solid #333; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .grid-3 { display: flex; justify-content: space-between; gap: 15px; }
        .stat-box { flex: 1; background: #1c1c1c; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #444; }
        .sidebar-box { background: #ffffff; padding: 15px; border-radius: 8px; border-left: 5px solid #cc0000; margin-bottom: 15px; color: #333; }
        .stButton>button { width: 100%; border-radius: 8px; background: #800000; color: white; border: none; transition: 0.3s; }
        .stButton>button:hover { background: #ffd700; color: #000; }
    </style>
    """

# --- 2. 增强数据函数 ---
@st.cache_data(ttl=600) # 缩短缓存时间，提升实时性
def fetch_market_snapshot():
    try:
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            # 统一清理代码格式，确保匹配
            df['代码'] = df['代码'].astype(str).str.zfill(6)
            return df.set_index('代码')[['名称', '换手率']].to_dict('index')
    except: pass
    return {}

def get_north_flow_safe(symbol_6digit):
    """安全获取北向资金"""
    try:
        hsgt_df = ak.stock_hsgt_individual_em(symbol=symbol_6digit)
        if not hsgt_df.empty:
            # 优先获取最新一行数据
            val = hsgt_df['当日净买入额'].iloc[0]
            return round(val / 10000, 2) if not pd.isna(val) else 0.0
    except: pass
    return 0.0

# --- 3. 核心诊断逻辑 (名称与换手率锁定补丁) ---
def diagnostic_core(ticker, risk_weight, snapshot, include_pro=False):
    try:
        symbol_6digit = ticker.split('.')[0]
        
        # 1. 基础行情下载
        time.sleep(0.1) 
        df = yf.download(ticker, period="250d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 2. 指标计算
        df['Vol_Ratio'] = (df['Volume'] / df['Volume'].rolling(5).mean()).fillna(1.0)
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Bias'] = ((df['Close'] - df['MA20']) / df['MA20']).fillna(0.0)
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean().fillna(df['Close']*0.02)
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = (100 - (100 / (1 + gain/loss))).fillna(50.0)
        
        # 3. 机器学习模型
        df['Target'] = (df['High'].shift(-5).rolling(5).max() > df['Close'] * 1.06).astype(int)
        feats = ['Vol_Ratio', 'Bias', 'RSI']
        train = df[feats + ['Target']].dropna()
        rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        rf.fit(train[feats].iloc[:-5].values, train['Target'].iloc[:-5].values)
        
        win_p = float(rf.predict_proba(df[feats].iloc[[-1]].fillna(0).values)[0][1])
        ev = (win_p * 0.08) - ((1 - win_p) * 0.04)
        
        # 4. 组装显示数据 (关键修复点)
        snap_data = snapshot.get(symbol_6digit, {})
        
        # 名称补丁
        name = snap_data.get('名称')
        if not name:
            try:
                info = ak.stock_individual_info_em(symbol=symbol_6digit)
                name = info[info['item'] == '股票简称']['value'].values[0]
            except: name = ticker

        # 换手率补丁：如果快照没有（比如还没开盘），则通过历史接口补位
        turnover = snap_data.get('换手率')
        if turnover is None:
            try:
                hist = ak.stock_zh_a_hist(symbol=symbol_6digit, period="daily", start_date=time.strftime("%Y%m%d"), adjust="qfq")
                turnover = hist['换手率'].iloc[-1]
            except: turnover = 0.0

        # 北向资金补丁
        north_val = "跳过"
        north_multiplier = 1.0
        if include_pro:
            north_val = get_north_flow_safe(symbol_6digit)
            if north_val > 500: north_multiplier = 1.15
            elif north_val < -500: north_multiplier = 0.85

        score = win_p * ev * risk_weight * north_multiplier * 1000
        curr_price = df['Close'].iloc[-1]
        atr_now = df['ATR'].iloc[-1]

        return {
            '名称': name, '代码': ticker, '现价': round(curr_price, 2),
            '预测胜率': f"{win_p:.1%}", '期望值(EV)': f"{ev*100:+.2f}%", 
            '周期': "5-10交易日", '建议买入': round(curr_price * 0.99, 2),
            '止盈参考': round(curr_price + (atr_now * 2.5), 2), 
            '止损建议': round(curr_price - (atr_now * 1.5), 2),
            '换手率': f"{turnover:.2f}%", '北向资金(万)': north_val,
            '综合评分': round(score, 2), 'Score_Raw': score 
        }
    except: return None

# --- 4. 界面渲染 ---
st.markdown(get_v24_css(), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🛡️ SENTINEL A-Share V24 PRO</h1><p>全市场量化扫描系统 • Safari & 换手率修复版</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🧬 系统逻辑简介")
    st.markdown('<div class="sidebar-box">核心为 <b>Hybrid-RF</b> 模型。包含：<li><b>Bias/RSI:</b> 价格动能。</li><li><b>Vol/Turnover:</b> 资金活跃度。</li><li><b>North_Flow:</b> 机构权重增益。</li></div>', unsafe_allow_html=True)

def get_market_env():
    try:
        m_df = yf.download("000300.SS", period="60d", progress=False)
        if not m_df.empty:
            if isinstance(m_df.columns, pd.MultiIndex): m_df.columns = m_df.columns.get_level_values(0)
            m_close = m_df['Close'].iloc[-1]
            m_ma20 = m_df['Close'].rolling(20).mean().iloc[-1]
            risk_weight = 1.2 if m_close > m_ma20 else 0.8
            st.markdown(f"""<div class="env-card"><div class="grid-3">
                <div class="stat-box"><small>沪深300</small><br><b>{m_close:.2f}</b></div>
                <div class="stat-box"><small>大盘环境</small><br><b>{"良性" if risk_weight > 1 else "谨慎"}</b></div>
                <div class="stat-box"><small>风控乘数</small><br><b style="color:#ffd700;">x{risk_weight}</b></div>
            </div></div>""", unsafe_allow_html=True)
            return risk_weight
    except: pass
    return 0.8

risk_weight = get_market_env()
tab1, tab2 = st.tabs(["🚀 核心资产 Top 50 扫描", "🔍 跨市场标的单兵诊断"])
DISPLAY_COLS = ['名称', '代码', '现价', '预测胜率', '期望值(EV)', '周期', '建议买入', '止盈参考', '止损建议', '换手率', '北向资金(万)', '综合评分']

with tab1:
    mode = st.radio("选择扫描引擎", ["⚡ 极速轻量版", "🧠 机器学习 Pro 版"], horizontal=True)
    if st.button("开始量化扫描"):
        snapshot = fetch_market_snapshot()
        try:
            df_300 = ak.index_stock_cons_csindex(symbol="000300")
            pool = [f"{c}.SS" if c.startswith('60') else f"{c}.SZ" for c in df_300['成分券代码'].head(50).tolist()]
        except: pool = ["600519.SS", "300750.SZ"]
        
        results = []
        p_bar = st.progress(0)
        for i, t in enumerate(pool):
            res = diagnostic_core(t, risk_weight, snapshot, include_pro=("Pro" in mode))
            if res: results.append(res)
            p_bar.progress((i + 1) / len(pool))
        
        if results:
            df_res = pd.DataFrame(results).sort_values('Score_Raw', ascending=False)
            st.dataframe(df_res[DISPLAY_COLS].style.background_gradient(subset=['综合评分'], cmap='RdYlGn'), width='stretch')

with tab2:
    user_input = st.text_input("代码 (示例: 600519.SS 300750.SZ)", "600519.SS 300750.SZ")
    if st.button("执行精准诊断"):
        snapshot = fetch_market_snapshot()
        tickers = user_input.replace(',', ' ').split()[:5]
        results = []
        for t in tickers:
            with st.spinner(f"分析中 {t}..."):
                res = diagnostic_core(t, risk_weight, snapshot, include_pro=True)
                if res: results.append(res)
        if results:
            st.dataframe(pd.DataFrame(results)[DISPLAY_COLS].style.background_gradient(subset=['综合评分'], cmap='RdYlGn'), width='stretch')
