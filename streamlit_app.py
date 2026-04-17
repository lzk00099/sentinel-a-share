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
st.set_page_config(page_title="SENTINEL A-Share V24 Lite", layout="wide")

# --- 1. CSS 样式（无网络请求）---
def get_v24_css():
    return """
    <style>
        .main-header { background: linear-gradient(135deg, #800000 0%, #333 100%); color: #ffd700; padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 25px; }
        .env-card { background: #1a1a1a; color: #ffd700; border: 1px solid #444; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .stat-box { background: #262626; padding: 12px; border-radius: 8px; text-align: center; border-bottom: 3px solid #ffd700; }
        .sidebar-box { background: #ffffff; padding: 15px; border-radius: 8px; border-left: 5px solid #cc0000; margin-bottom: 15px; color: #333; }
        .u-tips { font-size: 0.85rem; color: #666; line-height: 1.5; }
    </style>
    """

st.markdown(get_v24_css(), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🛡️ SENTINEL A-Share V24 Lite</h1><p>轻量启动版 • 点击按钮后加载数据</p></div>', unsafe_allow_html=True)

# 侧边栏（纯静态）
with st.sidebar:
    st.markdown("### 🧬 模型说明")
    st.markdown("""
    <div class="sidebar-box">
    随机森林预测未来5日涨幅>6%的概率。<br>
    因子：量比、乖离率、RSI。<br>
    止盈止损基于ATR，仓位建议基于凯利公式。
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### ⚠️ 首次运行")
    st.markdown("""
    <div class="u-tips">
    点击扫描按钮后，系统将实时获取数据，<br>
    请耐心等待20-30秒。
    </div>
    """, unsafe_allow_html=True)

# 初始化 session_state 存储风险权重（避免重复请求）
if 'risk_weight' not in st.session_state:
    st.session_state.risk_weight = 0.8
    st.session_state.risk_status = "未获取（点击扫描时自动获取）"

# 显示当前风险权重（占位，实际点击按钮后才更新）
st.info(f"当前风控系数：{st.session_state.risk_weight} | 大盘状态：{st.session_state.risk_status}")

# --- 2. 核心诊断函数（内部含网络请求，仅在调用时执行）---
@st.cache_data(ttl=3600)
def get_a_shares_pool():
    try:
        df_300 = ak.index_stock_cons_csindex(symbol="000300")
        df_top = df_300.head(50)
        return { (f"{row['成分券代码']}.SS" if row['成分券代码'].startswith('60') else f"{row['成分券代码']}.SZ"): row['成分券名称']
                for _, row in df_top.iterrows() }
    except:
        return {"600519.SS": "贵州茅台", "300750.SZ": "宁德时代", "601318.SS": "中国平安"}

def get_market_risk():
    """获取大盘风险系数，内部有网络请求"""
    try:
        m_df = yf.download("000300.SS", period="100d", progress=False)
        if m_df.empty:
            return 0.8, "数据获取失败"
        if isinstance(m_df.columns, pd.MultiIndex):
            m_df.columns = m_df.columns.get_level_values(0)
        m_close = m_df['Close'].iloc[-1]
        m_ma20 = m_df['Close'].rolling(20).mean().iloc[-1]
        m_ma60 = m_df['Close'].rolling(60).mean().iloc[-1]
        if m_close > m_ma20:
            return 1.2, "强势反弹 (进攻)"
        elif m_close > m_ma60:
            return 0.8, "缩量回调 (观察)"
        else:
            return 0.5, "空头趋势 (避险)"
    except:
        return 0.8, "默认系数"

def diagnostic_engine_a(ticker, name, risk_weight):
    try:
        df = yf.download(ticker, period="250d", progress=False, auto_adjust=True)
        if len(df) < 60:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
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
        if len(train) < 50:
            return None
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(train[feats].iloc[:-5].values, train['Target'].iloc[:-5].values)
        last_feat = df[feats].iloc[[-1]].values
        win_p = float(rf.predict_proba(last_feat)[0][1])
        
        ev = (win_p * 0.08) - ((1 - win_p) * 0.04)
        score = win_p * ev * risk_weight * 1000
        curr_price = df['Close'].iloc[-1]
        tp_price = curr_price + (atr_now * 2.5)
        sl_price = curr_price - (atr_now * 1.5)
        
        b = 0.08 / 0.04
        f = (win_p * (b + 1) - 1) / b if b > 0 else 0
        suggest_position = max(0, min(0.1, f * risk_weight))
        
        return {
            '代码': ticker, '名称': name, '现价': round(curr_price, 2),
            '预测胜率': f"{win_p:.1%}", '期望值': f"{ev*100:+.2f}%",
            '止盈参考': round(tp_price, 2), '止损建议': round(sl_price, 2),
            '综合评分': round(score, 2), '建议仓位': f"{suggest_position*100:.1f}%",
            'Score_Raw': score
        }
    except Exception:
        return None

def style_dataframe(df):
    styled = df.style.background_gradient(subset=['综合评分'], cmap='RdYlGn', low=0, high=20)
    def color_expect(val):
        if isinstance(val, str) and val.startswith('+'):
            return 'color: #00cc66'
        elif isinstance(val, str) and val.startswith('-'):
            return 'color: #ff4d4d'
        return ''
    styled = styled.applymap(color_expect, subset=['期望值'])
    return styled

# --- 3. 交互界面（所有网络请求均在按钮回调中）---
tab1, tab2 = st.tabs(["🚀 核心资产 Top 50 扫描", "🔍 单股诊断"])

with tab1:
    if st.button("开始全量扫描"):
        with st.spinner("正在获取大盘环境..."):
            risk_w, status = get_market_risk()
            st.session_state.risk_weight = risk_w
            st.session_state.risk_status = status
            st.success(f"大盘状态：{status}，风控系数：{risk_w}")
        
        with st.spinner("正在获取股票池..."):
            pool = get_a_shares_pool()
        
        results = []
        progress_bar = st.progress(0)
        for i, (t, n) in enumerate(pool.items()):
            res = diagnostic_engine_a(t, n, st.session_state.risk_weight)
            if res:
                results.append(res)
            progress_bar.progress((i + 1) / len(pool))
            time.sleep(0.05)  # 礼貌性延迟，避免请求过频
        
        if results:
            df_final = pd.DataFrame(results).sort_values('Score_Raw', ascending=False).head(12)
            st.subheader("🔥 推荐标的")
            st.dataframe(style_dataframe(df_final[['代码', '名称', '现价', '预测胜率', '期望值', '止盈参考', '止损建议', '综合评分', '建议仓位']]), use_container_width=True)
        else:
            st.warning("未发现有效信号，建议持币观望")

with tab2:
    user_input = st.text_input("输入 A 股代码（含后缀）", "600519.SS")
    if st.button("诊断单股"):
        with st.spinner("正在获取大盘环境..."):
            risk_w, status = get_market_risk()
            st.session_state.risk_weight = risk_w
        with st.spinner("正在分析..."):
            # 获取股票名称（简单处理）
            code = user_input.split('.')[0]
            try:
                name_df = ak.stock_info_a_code_name()
                name_map = dict(zip(name_df['code'], name_df['name']))
                name = name_map.get(code, user_input)
            except:
                name = user_input
            res = diagnostic_engine_a(user_input, name, st.session_state.risk_weight)
        if res:
            df_user = pd.DataFrame([res])
            st.dataframe(style_dataframe(df_user[['代码', '名称', '现价', '预测胜率', '期望值', '止盈参考', '止损建议', '综合评分', '建议仓位']]), use_container_width=True)
        else:
            st.error("数据获取失败，请检查代码格式（如 600519.SS）")