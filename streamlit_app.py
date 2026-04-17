import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
from sklearn.ensemble import RandomForestClassifier
import warnings

# --- 基础配置 ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="SENTINEL A-Share V24 Lite", layout="wide")

# --- 1. A股专用 CSS 渲染 ---
def get_v24_css():
    return """
    <style>
        .main-header { background: linear-gradient(135deg, #800000 0%, #333 100%); color: #ffd700; padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
        .env-card { background: #1a1a1a; color: #ffd700; border: 1px solid #444; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .stat-box { background: #262626; padding: 12px; border-radius: 8px; text-align: center; border-bottom: 3px solid #ffd700; }
        .sidebar-box { background: #ffffff; padding: 15px; border-radius: 8px; border-left: 5px solid #cc0000; margin-bottom: 15px; color: #333; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .u-tips { font-size: 0.85rem; color: #666; line-height: 1.5; }
        .dataframe th { background-color: #2c2c2c; color: #ffd700; }
        .dataframe td { font-size: 0.9rem; }
    </style>
    """

# --- 2. 辅助函数（仅获取股票池和名称，无额外请求）---
@st.cache_data(ttl=3600)
def get_a_shares_pool():
    """获取沪深300成分股前50只（使用akshare，但有缓存）"""
    try:
        df_300 = ak.index_stock_cons_csindex(symbol="000300")
        df_top = df_300.head(50)
        return { (f"{row['成分券代码']}.SS" if row['成分券代码'].startswith('60') else f"{row['成分券代码']}.SZ"): row['成分券名称']
                for _, row in df_top.iterrows() }
    except:
        return {"600519.SS": "贵州茅台", "300750.SZ": "宁德时代", "601318.SS": "中国平安"}

@st.cache_data(ttl=3600)
def get_stock_name(ticker):
    """手动诊断时获取股票名称（简单映射，失败则返回代码）"""
    code = ticker.split('.')[0]
    try:
        # 只获取一次全市场名称映射（使用缓存）
        df = ak.stock_info_a_code_name()
        name_map = dict(zip(df['code'], df['name']))
        return name_map.get(code, ticker)
    except:
        return ticker

# --- 3. 核心诊断引擎（只依赖yfinance，快速稳定）---
def diagnostic_engine_a(ticker, name, risk_weight):
    try:
        df = yf.download(ticker, period="250d", progress=False, auto_adjust=True)
        if len(df) < 60:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
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
        
        # 凯利公式仓位建议（可选输出）
        b = 0.08 / 0.04
        f = (win_p * (b + 1) - 1) / b if b > 0 else 0
        suggest_position = max(0, min(0.1, f * risk_weight))
        
        return {
            '代码': ticker, '名称': name, '现价': round(curr_price, 2),
            '预测胜率': f"{win_p:.1%}", 
            '期望值': f"{ev*100:+.2f}%", 
            '止盈参考': round(tp_price, 2), 
            '止损建议': round(sl_price, 2),
            '综合评分': round(score, 2),
            '建议仓位': f"{suggest_position*100:.1f}%",
            'Score_Raw': score
        }
    except Exception as e:
        # 静默失败
        return None

# --- 4. 统一表格样式 ---
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

# --- 5. Streamlit 界面 ---
st.markdown(get_v24_css(), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🛡️ SENTINEL A-Share V24 Lite</h1><p>沪深300核心资产量化扫描 • 轻量快速版</p></div>', unsafe_allow_html=True)

# 侧边栏（精简）
with st.sidebar:
    st.markdown("### 🧬 模型逻辑")
    st.markdown("""
    <div class="sidebar-box">
    <b>随机森林分类器</b> 预测未来5日涨幅超6%的概率。<br>
    因子：量比(Vol_Ratio)、乖离率(Bias)、RSI。<br>
    动态止盈止损基于ATR，仓位建议基于凯利公式。
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 🛠️ 使用说明")
    st.markdown("""
    <div class="u-tips">
    <li>综合评分 > 10 → 强信号</li>
    <li>综合评分 5~10 → 观察</li>
    <li>综合评分 < 0 → 规避</li>
    </div>
    """, unsafe_allow_html=True)

# 顶部大盘环境（仅依赖沪深300指数，无额外请求）
try:
    # 使用yfinance获取沪深300指数（代码000300.SS）
    m_df = yf.download("000300.SS", period="100d", progress=False)
    if not m_df.empty:
        if isinstance(m_df.columns, pd.MultiIndex):
            m_df.columns = m_df.columns.get_level_values(0)
        m_close = m_df['Close'].iloc[-1]
        m_ma20 = m_df['Close'].rolling(20).mean().iloc[-1]
        m_ma60 = m_df['Close'].rolling(60).mean().iloc[-1]
        
        risk_weight = 1.2 if m_close > m_ma20 else 0.8 if m_close > m_ma60 else 0.5
        m_status = "强势反弹 (进攻)" if m_close > m_ma20 else "缩量回调 (观察)" if m_close > m_ma60 else "空头趋势 (避险)"
        
        st.markdown(f"""
        <div class="env-card">
            <div style="margin-bottom:12px; font-weight:bold; font-size:1.1rem;">📊 大盘环境</div>
            <div class="grid-2">
                <div class="stat-box"><small>沪深300</small><br><b>{m_close:.2f}</b></div>
                <div class="stat-box"><small>战术建议</small><br><b>{m_status}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        risk_weight = 0.8
        st.warning("无法获取大盘数据，使用默认风险权重")
except:
    risk_weight = 0.8

# 功能标签页
tab1, tab2 = st.tabs(["🚀 核心资产 Top 50 扫描", "🔍 单股诊断"])

DISPLAY_COLS = ['代码', '名称', '现价', '预测胜率', '期望值', '止盈参考', '止损建议', '综合评分', '建议仓位']

with tab1:
    st.write("点击下方按钮，扫描沪深300权重前50只标的（每次约30秒，请耐心等待）")
    if st.button("开始扫描"):
        pool = get_a_shares_pool()
        results = []
        progress_bar = st.progress(0)
        for i, (t, n) in enumerate(pool.items()):
            res = diagnostic_engine_a(t, n, risk_weight)
            if res:
                results.append(res)
            progress_bar.progress((i + 1) / len(pool))
        if results:
            df_final = pd.DataFrame(results).sort_values('Score_Raw', ascending=False).head(12)
            st.subheader("🔥 推荐标的")
            st.dataframe(style_dataframe(df_final[DISPLAY_COLS]), use_container_width=True)
        else:
            st.warning("未发现有效信号，建议持币观望")

with tab2:
    st.write("输入A股代码（含后缀），例如：600519.SS")
    user_input = st.text_input("代码", "600519.SS")
    if st.button("诊断"):
        name = get_stock_name(user_input)
        res = diagnostic_engine_a(user_input, name, risk_weight)
        if res:
            df_user = pd.DataFrame([res])
            st.dataframe(style_dataframe(df_user[DISPLAY_COLS]), use_container_width=True)
        else:
            st.error("获取数据失败，请检查代码格式")