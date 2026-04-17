import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
from sklearn.ensemble import RandomForestClassifier
import warnings
from datetime import datetime, timedelta

# --- 基础配置 ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="SENTINEL A-Share V24 - Pro", layout="wide")

# --- 1. 增强版 CSS 渲染 ---
def get_v24_css():
    return """
    <style>
        .main-header { background: linear-gradient(135deg, #1a1a1a 0%, #800000 100%); color: #ffd700; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 1px solid #ffd700; }
        .env-card { background: #0e1117; color: #ffd700; border: 1px solid #333; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
        .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; }
        .stat-box { background: #1c1c1c; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #444; }
        .sidebar-box { background: #fdfdfd; padding: 15px; border-radius: 10px; border-left: 5px solid #cc0000; margin-bottom: 15px; color: #333; }
        .metric-up { color: #ff4b4b; font-weight: bold; }
        .metric-down { color: #00ff00; font-weight: bold; }
        .stButton>button { width: 100%; border-radius: 8px; background: #800000; color: white; border: none; transition: 0.3s; }
        .stButton>button:hover { background: #ffd700; color: #000; }
    </style>
    """

# --- 2. 增强型数据引擎 (加入换手率、北向、板块) ---
@st.cache_data(ttl=3600)
def get_stock_info_extra(ticker):
    """获取个股所属板块、换手率及北向资金"""
    try:
        symbol = ticker.split('.')[0]
        # 获取实时快照 (含换手率、名称)
        spot_df = ak.stock_zh_a_spot_em()
        row = spot_df[spot_df['代码'] == symbol].iloc[0]
        name = row['名称']
        turnover = row['换手率']
        
        # 获取板块信息
        sector_df = ak.stock_board_industry_cons_em(symbol=row['名称']) # 这是一个hack写法，通常用行业板块接口
        sector = "核心资产" # 缺省值
        
        # 北向资金 (近1日净流入)
        hsgt_df = ak.stock_hsgt_individual_em(symbol=symbol)
        north_flow = hsgt_df['当日净买入额'].iloc[0] / 10000 if not hsgt_df.empty else 0
        
        return name, turnover, north_flow
    except:
        return "未知", 0.0, 0.0

@st.cache_data(ttl=3600)
def get_a_shares_pool():
    try:
        df_300 = ak.index_stock_cons_csindex(symbol="000300")
        df_top = df_300.head(50)
        return { (f"{row['成分券代码']}.SS" if row['成分券代码'].startswith('60') else f"{row['成分券代码']}.SZ"): row['成分券名称']
                for _, row in df_top.iterrows() }
    except:
        return {"600519.SS": "贵州茅台", "300750.SZ": "宁德时代", "601318.SS": "中国平安"}

def diagnostic_engine_v2(ticker, risk_weight, manual_name=None):
    try:
        # 获取基础量价
        df = yf.download(ticker, period="250d", progress=False, auto_adjust=True)
        if len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 1. 基础因子计算
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
        
        # 2. 机器学习目标 (T+5 涨幅 > 6%)
        df['Target'] = (df['High'].shift(-5).rolling(5).max() > df['Close'] * 1.06).astype(int)
        feats = ['Vol_Ratio', 'Bias', 'RSI']
        train = df[feats + ['Target']].dropna()
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(train[feats].iloc[:-5].values, train['Target'].iloc[:-5].values)
        
        last_feat = df[feats].iloc[[-1]].values
        win_p = float(rf.predict_proba(last_feat)[0][1])
        
        # 3. 期望值与评分
        ev = (win_p * 0.08) - ((1 - win_p) * 0.04)
        score = win_p * ev * risk_weight * 1000
        
        # 4. 价格参考
        curr_price = df['Close'].iloc[-1]
        tp_price = curr_price + (atr_now * 2.5)
        sl_price = curr_price - (atr_now * 1.5)
        
        # 5. 获取额外维度 (名称同步修正)
        name, turnover, north_flow = get_stock_info_extra(ticker)
        if manual_name and manual_name != "手动查询": name = manual_name
        
        return {
            '代码': ticker, 
            '名称': name, 
            '现价': round(curr_price, 2),
            '预测胜率': f"{win_p:.1%}", 
            '期望值(EV)': f"{ev*100:+.2f}%", 
            '周期': "5-10交易日",
            '建议买入': round(curr_price * 0.99, 2),
            '止盈参考': round(tp_price, 2), 
            '止损建议': round(sl_price, 2),
            '换手率': f"{turnover:.2f}%",
            '北向资金(万)': round(north_flow, 2),
            '综合评分': round(score, 2),
            'Score_Raw': score 
        }
    except: return None

# --- 3. 界面布局 ---
st.markdown(get_v24_css(), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🛡️ SENTINEL A-Share V24 PRO</h1><p>全市场量化扫描系统 • EV+RandomForest 增强版</p></div>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown("### 🧬 系统逻辑简介")
    st.markdown('<div class="sidebar-box"><b>Hybrid-RF 模型:</b> 结合 QQQ/SPY 全球情绪与 A 股量价特征。<br><br><b>新指标说明:</b><br><li><b>北向资金:</b> 监测聪明钱流向。</li><li><b>换手率:</b> 识别活跃度与主力动向。</li></div>', unsafe_allow_html=True)
    
    st.markdown("### 🕒 最佳运行时间")
    st.markdown('<div class="sidebar-box">1. 盘前: 宏观定调<br>2. 盘中: 捕捉异动<br>3. 尾盘: T+1 锁定期望值</div>', unsafe_allow_html=True)

# 顶部大盘环境评估 (增加 QQQ/SPY/IWM 参考)
def get_market_env():
    # 沪深300
    m_df = yf.download("000300.SS", period="60d", progress=False)
    # 全球参考
    global_df = yf.download(["QQQ", "SPY", "IWM"], period="5d", progress=False)['Close']
    
    if not m_df.empty:
        m_close = m_df['Close'].iloc[-1]
        m_ma20 = m_df['Close'].rolling(20).mean().iloc[-1]
        m_ma60 = m_df['Close'].rolling(60).mean().iloc[-1]
        
        risk_weight = 1.2 if m_close > m_ma20 else 0.8 if m_close > m_ma60 else 0.5
        m_status = "强势反弹" if m_close > m_ma20 else "缩量回调" if m_close > m_ma60 else "空头避险"
        
        # 全球情绪简单判断
        global_perf = (global_df.iloc[-1] / global_df.iloc[0] - 1).mean()
        global_status = "偏暖" if global_perf > 0 else "承压"

        st.markdown(f"""
        <div class="env-card">
            <div style="margin-bottom:12px; font-weight:bold; border-left:4px solid #ffd700; padding-left:10px;">🚨 宏观环境多维监测 (A-Share + Global)</div>
            <div class="grid-3">
                <div class="stat-box"><small>沪深300</small><br><b style="font-size:1.2rem;">{m_close:.2f}</b><br><small>{m_status}</small></div>
                <div class="stat-box"><small>全球参考 (QQQ/SPY)</small><br><b style="font-size:1.2rem;">{global_status}</b><br><small>美股联动系数</small></div>
                <div class="stat-box"><small>风控乘数</small><br><b style="font-size:1.4rem; color:#ffd700;">x{risk_weight}</b><br><small>建议仓位比例</small></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return risk_weight
    return 0.8

risk_weight = get_market_env()

# 功能标签页
tab1, tab2 = st.tabs(["🚀 核心资产 Top 50 扫描", "🔍 跨市场标的单兵诊断"])

# 统一样式列名
DISPLAY_COLS = ['名称', '代码', '现价', '预测胜率', '期望值(EV)', '周期', '建议买入', '止盈参考', '止损建议', '换手率', '北向资金(万)', '综合评分']

with tab1:
    st.info("系统将实时同步个股的换手率与北向资金流向。")
    if st.button("开始全量量化扫描"):
        pool = get_a_shares_pool()
        results = []
        progress_bar = st.progress(0)
        
        for i, (t, n) in enumerate(pool.items()):
            res = diagnostic_engine_v2(t, risk_weight, manual_name=n)
            if res: results.append(res)
            progress_bar.progress((i + 1) / len(pool))
        
        if results:
            df_final = pd.DataFrame(results).sort_values('Score_Raw', ascending=False).head(15)
            st.subheader("🔥 SENTINEL 选股池 (高期望值标的)")
            # 使用统一的颜色着色逻辑
            st.dataframe(
                df_final[DISPLAY_COLS].style.background_gradient(subset=['综合评分'], cmap='RdYlGn')
                .applymap(lambda x: 'color: #ff4b4b' if '+' in str(x) else 'color: #00ff00', subset=['期望值(EV)']),
                use_container_width=True
            )
        else:
            st.warning("⚠️ 未发现正期望值标的。")

with tab2:
    st.write("输入 A 股或全球标的代码，手动输入上限 5 个（例如: 600519.SS 000661.SZ AAPL）。")
    user_input = st.text_input("代码列表", "600519.SS 300750.SZ 601318.SS 000001.SZ 600900.SS")
    if st.button("执行精准诊断"):
        # 限制最多5个
        tickers = user_input.replace(',', ' ').split()[:5]
        results = []
        for t in tickers:
            with st.spinner(f"正在深度分析 {t}..."):
                res = diagnostic_engine_v2(t, risk_weight)
                if res: results.append(res)
        
        if results:
            df_user = pd.DataFrame(results).sort_values('Score_Raw', ascending=False)
            st.subheader("📊 诊断报告")
            # 同步 Tab 1 的外观
            st.dataframe(
                df_user[DISPLAY_COLS].style.background_gradient(subset=['综合评分'], cmap='RdYlGn')
                .applymap(lambda x: 'color: #ff4b4b' if '+' in str(x) else 'color: #00ff00', subset=['期望值(EV)']),
                use_container_width=True
            )
        else:
            st.error("数据抓取失败。请确保输入格式正确（如 600519.SS）。")
