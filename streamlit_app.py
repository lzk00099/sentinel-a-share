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

# --- 1. A股专用 CSS 渲染 ---
def get_v24_css():
    return """
    <style>
        .main-header { background: linear-gradient(135deg, #800000 0%, #333 100%); color: #ffd700; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
        .env-card { background: #1a1a1a; color: #ffd700; border: 1px solid #444; border-radius: 10px; padding: 15px; margin-bottom: 20px; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .stat-box { background: #262626; padding: 10px; border-radius: 6px; text-align: center; border-bottom: 2px solid #ffd700; }
        .info-sidebar { font-size: 0.9rem; line-height: 1.6; }
    </style>
    """

# --- 2. 核心诊断引擎 (已集成 ATR 动态逻辑) ---
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
        
        # 特征工程
        df['Ret'] = df['Close'].pct_change()
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Bias'] = (df['Close'] - df['MA20']) / df['MA20']
        
        # ATR 波动率计算 (新增强点)
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        atr_now = df['ATR'].iloc[-1]
        
        # RSI 因子
        change = df['Close'].diff()
        gain = (change.where(change > 0, 0)).rolling(14).mean()
        loss = (-change.where(change < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        
        # 标签：5日内最大涨幅 > 6%
        df['Target'] = (df['High'].shift(-5).rolling(5).max() > df['Close'] * 1.06).astype(int)
        
        feats = ['Vol_Ratio', 'Bias', 'RSI']
        train = df[feats + ['Target']].dropna()
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(train[feats].iloc[:-5].values, train['Target'].iloc[:-5].values)
        
        last_feat = df[feats].iloc[[-1]].values
        win_p = float(rf.predict_proba(last_feat)[0][1])
        
        # EV 与 评分
        ev = (win_p * 0.08) - ((1 - win_p) * 0.04)
        score = win_p * ev * risk_weight * 1000
        
        curr_price = df['Close'].iloc[-1]
        
        # 动态止盈止损 (基于 ATR)
        # 止盈设为 2.5 倍波动，止损设为 1.5 倍波动，更加适配个股股性
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

# --- 3. Streamlit 界面布局 ---
st.markdown(get_v24_css(), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🛡️ SENTINEL A-Share V24</h1><p>A股核心资产量化诊断系统</p></div>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown("### 📖 系统简介")
    st.markdown("""
    本系统采用 **机器学习 (RF)** 融合 **ATR 波动率算法**。
    针对 A 股 T+1 制度，重点捕捉量价乖离后的回归机会。
    """)
    st.markdown("---")
    st.markdown("### 🛠️ 使用手册")
    st.write("1. **权重扫描**：自动分析沪深300中权数最高的蓝筹标的。")
    st.write("2. **精确打击**：支持手动输入，如 `600519.SS`。")
    st.write("3. **动态止盈**：系统根据个股近 14 天的平均波动幅度自动锁定目标位。")
    st.markdown("---")
    st.info("数据来源：Yahoo Finance / AkShare")

# 顶部大盘环境评估
m_ticker = "000300.SS"
m_df = yf.download(m_ticker, period="100d", progress=False)

if not m_df.empty:
    if isinstance(m_df.columns, pd.MultiIndex): m_df.columns = m_df.columns.get_level_values(0)
    m_close = m_df['Close'].iloc[-1]
    m_ma20 = m_df['Close'].rolling(20).mean().iloc[-1]
    m_ma60 = m_df['Close'].rolling(60).mean().iloc[-1]
    
    risk_weight = 1.2 if m_close > m_ma20 else 0.8 if m_close > m_ma60 else 0.5
    m_status = "强势反弹" if m_close > m_ma20 else "缩量回调" if m_close > m_ma60 else "多头禁区"
    
    st.markdown(f"""
    <div class="env-card">
        <div style="margin-bottom:10px; font-weight:bold;">🚨 A股宏观情绪监测 (沪深300)</div>
        <div class="grid-2">
            <div class="stat-box"><small>当前点位</small><br><b>{m_close:.2f}</b></div>
            <div class="stat-box"><small>战术环境</small><br><b>{m_status} (系数: {risk_weight})</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    risk_weight = 0.8

# 功能标签页
tab1, tab2 = st.tabs(["🚀 沪深300权重扫描", "🔍 任意标的诊断"])

# 定义统一的显示列，确保“止盈”和“止损”不会被漏掉
DISPLAY_COLS = ['代码', '名称', '现价', '预测胜率', '期望值', '止盈参考', '止损建议', '综合评分']

with tab1:
    if st.button("开始执行全量扫描"):
        pool = get_a_shares_pool()
        results = []
        progress_bar = st.progress(0)
        for i, (t, n) in enumerate(pool.items()):
            res = diagnostic_engine_a(t, n, risk_weight)
            if res: results.append(res)
            progress_bar.progress((i + 1) / len(pool))
        
        if results:
            df_final = pd.DataFrame(results).sort_values('Score_Raw', ascending=False).head(10)
            st.subheader("🔥 核心资产最佳机会 Top 10")
            # 使用 st.table 确保在各种屏幕下都能强制显示完整列
            st.table(df_final[DISPLAY_COLS])
        else:
            st.warning("当前环境下暂无高分标的。")

with tab2:
    user_input = st.text_input("请输入 A 股代码，格式示例：`600519.SS` (沪市) 或 `000001.SZ` (深市)")
    if st.button("开始精准诊断"):
        tickers = user_input.replace(',', ' ').split()
        results = []
        for t in tickers:
            res = diagnostic_engine_a(t, "手动查询", risk_weight)
            if res: results.append(res)
        
        if results:
            df_user = pd.DataFrame(results).sort_values('Score_Raw', ascending=False)
            st.subheader("📊 诊断报告")
            # 交互式表格显示
            st.dataframe(
                df_user[DISPLAY_COLS].style.background_gradient(subset=['综合评分'], cmap='RdYlGn'),
                use_container_width=True
            )
        else:
            st.error("未获取到数据，请检查代码后缀是否为 .SS(沪) 或 .SZ(深)")
