import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
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
        .info-sidebar { background: #fdf2f2; border-left: 4px solid #cc0000; padding: 15px; border-radius: 4px; font-size: 0.9rem; }
    </style>
    """

# --- 2. 核心诊断引擎 ---
@st.cache_data(ttl=3600) # 缓存1小时，避免重复请求akshare
def get_a_shares_pool():
    try:
        df_300 = ak.index_stock_cons_csindex(symbol="000300")
        df_top = df_300.head(50)
        return { (f"{row['成分券代码']}.SS" if row['成分券代码'].startswith('60') else f"{row['成分券代码']}.SZ"): row['成分券名称']
                for _, row in df_top.iterrows() }
    except:
        return {"600519.SS": "贵州茅台", "300750.SZ": "宁德时代", "601318.SS": "中国平安", "000858.SZ": "五粮液"}

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
        
        # A股增强因子：波动率与RSI
        change = df['Close'].diff()
        gain = (change.where(change > 0, 0)).rolling(14).mean()
        loss = (-change.where(change < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        
        # 标签：5日内涨幅 > 6%
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
        return {
            '代码': ticker, '名称': name, '现价': round(curr_price, 2),
            '预测胜率': f"{win_p:.1%}", '期望值': f"{ev*100:+.2f}%", 
            '综合评分': round(score, 2),
            '止盈参考': round(curr_price * 1.08, 2), 
            '止损建议': round(curr_price * 0.96, 2),
            'Score_Raw': score # 用于排序
        }
    except: return None

# --- 3. Streamlit 界面 ---
st.markdown(get_v24_css(), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🛡️ SENTINEL A-Share V24</h1><p>沪深300核心资产量化扫描系统</p></div>', unsafe_allow_html=True)

# 侧边栏：简介与手册
with st.sidebar:
    st.markdown("### 📖 系统简介")
    st.markdown("""
    本系统采用 **Random Forest (随机森林)** 机器学习模型，针对 A 股 T+1 交易制度进行优化。通过分析量价乖离度 (Bias) 和相对强弱指数 (RSI) 预测未来 5 个交易日的获利概率。
    """)
    st.markdown("---")
    st.markdown("### 🛠️ 操作手册")
    st.write("1. **全局扫描**：一键分析沪深 300 前 50 大权重蓝筹股。")
    st.write("2. **精确打击**：输入 A 股代码（如 600519.SS 或 000001.SZ）。")
    st.write("3. **评分规则**：综合评分 > 5 为关注，> 10 为强信号。")
    st.markdown("---")
    st.info("💡 提示：A 股数据建议在收盘后（15:30）或开盘半小时后观察。")

# 主界面：环境评估
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
        <div style="margin-bottom:10px; font-weight:bold;">🚨 A股宏观情绪监测 (沪深300基准)</div>
        <div class="grid-2">
            <div class="stat-box"><small>基准点位</small><br><b>{m_close:.2f}</b></div>
            <div class="stat-box"><small>当前策略</small><br><b>{m_status} (风控系数: {risk_weight})</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 功能标签页
tab1, tab2 = st.tabs(["🚀 沪深300权重扫描", "🔍 任意标的诊断"])

with tab1:
    if st.button("开始扫描核心资产"):
        pool = get_a_shares_pool()
        results = []
        progress_bar = st.progress(0)
        for i, (t, n) in enumerate(pool.items()):
            res = diagnostic_engine_a(t, n, risk_weight)
            if res: results.append(res)
            progress_bar.progress((i + 1) / len(pool))
        
        if results:
            df_final = pd.DataFrame(results).sort_values('Score_Raw', ascending=False).head(10)
            st.table(df_final.drop(columns=['Score_Raw']))
        else:
            st.warning("当前市场环境下未发现符合期望的标的。")

with tab2:
    st.write("请输入 A 股代码，格式示例：`600519.SS` (沪市) 或 `000001.SZ` (深市)")
    user_input = st.text_input("代码列表（用空格分隔）", "600519.SS 300750.SZ 000001.SZ")
    if st.button("执行诊断"):
        tickers = user_input.replace(',', ' ').split()
        results = []
        for t in tickers:
            res = diagnostic_engine_a(t, "手动查询", risk_weight)
            if res: results.append(res)
        
        if results:
            df_user = pd.DataFrame(results).sort_values('Score_Raw', ascending=False)
            st.dataframe(df_user.drop(columns=['Score_Raw']).style.background_gradient(subset=['综合评分'], cmap='YlOrRd'))
        else:
            st.error("无法获取数据，请检查代码格式是否正确。")
