import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
from sklearn.ensemble import RandomForestClassifier
import warnings

# --- 基础配置 ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="SENTINEL A-Share V24.1", layout="wide")

# --- 1. A股专用 CSS (增强色彩感) ---
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

# --- 2. 增强型数据中心 (换手率与资金流) ---
@st.cache_data(ttl=600)
def get_a_market_snapshot():
    """获取全市场快照，包含名称、换手率和资金净占比"""
    try:
        df = ak.stock_zh_a_spot_em()
        # 映射代码格式 000001 -> 000001.SZ
        df['ticker'] = df['代码'].apply(lambda x: f"{x}.SS" if x.startswith('60') or x.startswith('68') else f"{x}.SZ")
        return df.set_index('ticker')[['名称', '换手率', '主力净流入-净占比']]
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_hs300_pool():
    try:
        df_300 = ak.index_stock_cons_csindex(symbol="000300")
        return df_300['成分券代码'].tolist()
    except:
        return ["600519", "300750", "601318"]

# --- 3. 核心诊断引擎 ---
def diagnostic_engine_a(ticker, snapshot, risk_weight):
    try:
        # 获取名称及附加指标
        raw_code = ticker.split('.')[0]
        name = snapshot.loc[ticker, '名称'] if ticker in snapshot.index else "未知"
        turnover = snapshot.loc[ticker, '换手率'] if ticker in snapshot.index else 0
        money_flow = snapshot.loc[ticker, '主力净流入-净占比'] if ticker in snapshot.index else 0
        
        # 抓取历史数据
        df = yf.download(ticker, period="200d", progress=False, auto_adjust=True)
        if len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 技术因子计算
        df['Ret'] = df['Close'].pct_change()
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Bias'] = (df['Close'] - df['MA20']) / df['MA20']
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        atr_now = df['ATR'].iloc[-1]
        
        # RSI
        change = df['Close'].diff()
        gain = (change.where(change > 0, 0)).rolling(14).mean()
        loss = (-change.where(change < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        
        # 机器学习预测
        df['Target'] = (df['High'].shift(-5).rolling(5).max() > df['Close'] * 1.06).astype(int)
        feats = ['Vol_Ratio', 'Bias', 'RSI']
        train = df[feats + ['Target']].dropna()
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(train[feats].iloc[:-5].values, train['Target'].iloc[:-5].values)
        
        win_p = float(rf.predict_proba(df[feats].iloc[[-1]].values)[0][1])
        
        # 评分模型 (加入资金流修正)
        flow_bonus = 1.1 if money_flow > 0 else 0.9
        ev = (win_p * 0.08) - ((1 - win_p) * 0.04)
        score = win_p * ev * risk_weight * flow_bonus * 1000
        
        curr_price = df['Close'].iloc[-1]
        return {
            '代码': ticker, '名称': name, '现价': round(curr_price, 2),
            '换手率%': round(turnover, 2), '主力资金%': round(money_flow, 2),
            '预测胜率': win_p, 
            '期望值%': round(ev * 100, 2), 
            '止盈参考': round(curr_price + atr_now * 2.5, 2), 
            '止损建议': round(curr_price - atr_now * 1.5, 2),
            '综合评分': round(score, 2)
        }
    except: return None

# --- 4. 界面渲染函数 ---
def display_styled_df(results_list):
    if not results_list:
        st.warning("⚠️ 未发现符合正向期望值的标的。")
        return
    
    df = pd.DataFrame(results_list).sort_values('综合评分', ascending=False)
    
    # 格式化百分比显示（仅用于显示）
    df_styled = df.copy()
    df_styled['预测胜率'] = df_styled['预测胜率'].map('{:.1%}'.format)

    # 颜色渲染逻辑
    def color_negative_red(val):
        color = '#d32f2f' if val < 0 else '#2e7d32'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        df_styled.style.applymap(color_negative_red, subset=['期望值%', '综合评分'])
        .background_gradient(subset=['综合评分'], cmap='RdYlGn', vmin=-5, vmax=15),
        use_container_width=True,
        height=450
    )

# --- 5. Streamlit 主程序 ---
st.markdown(get_v24_css(), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🛡️ SENTINEL A-Share V24.1</h1><p>多因子机器学习诊断系统 (资金流/换手率增强版)</p></div>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown("### 🧬 核心因子说明")
    st.markdown("""<div class="sidebar-box">
    <b>换手率:</b> 监控个股活跃度及筹码交换频率。<br>
    <b>主力资金:</b> 追踪大单净流入占比(EM数据)。<br>
    <b>动态ATR:</b> 基于波动率自适应止盈止损。</div>""", unsafe_allow_html=True)
    st.info("🕒 最佳建议：收盘前15分钟观察模型结论，锁定T+1胜率。")

# 环境评估
snapshot = get_a_market_snapshot()
m_df = yf.download("000300.SS", period="100d", progress=False)
risk_weight = 0.8
if not m_df.empty:
    if isinstance(m_df.columns, pd.MultiIndex): m_df.columns = m_df.columns.get_level_values(0)
    m_close = m_df['Close'].iloc[-1]
    m_ma20 = m_df['Close'].rolling(20).mean().iloc[-1]
    risk_weight = 1.2 if m_close > m_ma20 else 0.8
    st.markdown(f"""<div class="env-card"><div class="grid-2">
    <div class="stat-box"><small>沪深300</small><br><b>{m_close:.2f}</b></div>
    <div class="stat-box"><small>风控权重</small><br><b>{risk_weight}</b></div>
    </div></div>""", unsafe_allow_html=True)

# 标签页同步
tab1, tab2 = st.tabs(["🚀 核心资产权重扫描", "🔍 个股精准诊断"])

with tab1:
    if st.button("全量诊断 沪深300 前50"):
        pool_codes = get_hs300_pool()[:50]
        results = []
        bar = st.progress(0)
        for i, c in enumerate(pool_codes):
            ticker = f"{c}.SS" if c.startswith('60') or c.startswith('68') else f"{c}.SZ"
            res = diagnostic_engine_a(ticker, snapshot, risk_weight)
            if res: results.append(res)
            bar.progress((i+1)/len(pool_codes))
        display_styled_df(results)

with tab2:
    codes_input = st.text_input("输入代码 (示例: 600519.SS 000001.SZ)", "600519.SS 300750.SZ 601318.SS")
    if st.button("开始单兵诊断"):
        results = []
        tickers = codes_input.replace(',', ' ').split()
        for t in tickers:
            res = diagnostic_engine_a(t, snapshot, risk_weight)
            if res: results.append(res)
        display_styled_df(results)
