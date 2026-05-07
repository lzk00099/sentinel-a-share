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
st.set_page_config(page_title="SENTINEL V26 - EV & ML DIAGNOSIS", layout="wide")

# --- 1. 样式表 ---
def get_v24_css():
    return """
    <style>
        .main-header { background: linear-gradient(135deg, #000000 0%, #1a3a5a 100%); color: #00d4ff; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 1px solid #00d4ff; }
        .env-card { background: #0e1117; color: #ffffff; border: 1px solid #333; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }
        .stat-box { background: #1c1c1c; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #444; }
        .sidebar-box { background: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 5px solid #1a3a5a; margin-bottom: 15px; color: #333; }
    </style>
    """

# --- 2. 数据获取增强 ---
@st.cache_data(ttl=600)
def fetch_market_snapshot():
    """修复点：匹配最新的 akshare 字段名"""
    try:
        df = ak.stock_zh_a_spot_em()
        # 建立 6 位代码到名称和换手率的映射
        df['code_clean'] = df['代码'].astype(str).str.zfill(6)
        return df.set_index('code_clean')[['名称', '换手率']].to_dict('index')
    except Exception as e:
        st.warning(f"行情快照获取失败: {e}")
        return {}

def get_north_flow(symbol):
    """个股北向资金流向"""
    try:
        # 仅针对 A 股进行查询
        df = ak.stock_hsgt_individual_em(symbol=symbol)
        if not df.empty:
            return round(df['当日净买入额'].iloc[0] / 10000, 2) # 万
    except:
        pass
    return 0.0

# --- 3. 核心诊断模型 (Random Forest + EV) ---
def diagnostic_engine(ticker, market_weight, snapshot, include_pro=True):
    try:
        # 1. 代码解析
        clean_symbol = "".join(filter(str.isdigit, ticker))
        
        # 2. 下载数据 (包含计算 EV 所需的历史)
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 3. 特征工程
        df['Returns'] = df['Close'].pct_change()
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()
        df['RSI'] = 100 - (100 / (1 + (df['Returns'].where(df['Returns'] > 0, 0).rolling(14).mean() / 
                                     df['Returns'].where(df['Returns'] < 0, 0).abs().rolling(14).mean())))
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Bias'] = (df['Close'] - df['MA20']) / df['MA20']
        
        # 4. 随机森林预测 (Target: 5日内是否有 6% 的涨幅)
        df['Target'] = (df['High'].shift(-5).rolling(5).max() > df['Close'] * 1.06).astype(int)
        features = ['Vol_Ratio', 'RSI', 'Bias']
        data = df[features + ['Target']].dropna()
        
        X = data[features].iloc[:-5]
        y = data['Target'].iloc[:-5]
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
        
        # 预测当前胜率
        curr_feat = df[features].iloc[[-1]].values
        win_prob = float(model.predict_proba(curr_feat)[0][1])
        
        # 5. EV (期望值) 计算
        # EV = (胜率 * 预期收益) + (败率 * 预期亏损)
        avg_gain = 0.07  # 设定目标盈利 7%
        avg_loss = -0.04 # 设定止损 4%
        ev = (win_prob * avg_gain) + ((1 - win_prob) * avg_loss)
        
        # 6. 价格逻辑 (ATR 辅助)
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        curr_price = df['Close'].iloc[-1]
        
        # 7. 匹配快照数据 (解决“未知”名称问题)
        stock_meta = snapshot.get(clean_symbol, {})
        name = stock_meta.get('名称', '海外标的/指数')
        turnover = stock_meta.get('换手率', 0.0)
        
        north_val = get_north_flow(clean_symbol) if ('.SS' in ticker or '.SZ' in ticker) else "N/A"

        return {
            '名称': name,
            '代码': ticker,
            '现价': round(curr_price, 2),
            '预测胜率': f"{win_prob:.1%}",
            '期望值(EV)': f"{ev*100:+.2f}%",
            '周期': "5-8 交易日",
            '建议买入': round(curr_price * 0.995, 2),
            '止盈参考': round(curr_price + (atr * 2.2), 2),
            '止损建议': round(curr_price - (atr * 1.5), 2),
            '换手率': f"{turnover:.2f}%",
            '北向流入(万)': north_val,
            '综合评分': round(win_prob * ev * market_weight * 1000, 2)
        }
    except:
        return None

# --- 4. 界面与市场评分 ---
st.markdown(get_v24_css(), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🛡️ SENTINEL V26 PRO</h1><p>Expected Value + Random Forest Multi-Market Diagnosis</p></div>', unsafe_allow_html=True)

@st.cache_data(ttl=1800)
def calculate_market_score():
    """综合评估 QQQ, IWM, SPY 和 沪深300"""
    indices = {"沪深300": "000300.SS", "NASDAQ (QQQ)": "QQQ", "S&P500 (SPY)": "SPY", "Russell 2000 (IWM)": "IWM"}
    scores = {}
    total_weight = 1.0
    
    try:
        data = yf.download(list(indices.values()), period="20d", progress=False)['Close']
        for label, ticker in indices.items():
            current = data[ticker].iloc[-1]
            ma20 = data[ticker].mean()
            change = (current - data[ticker].iloc[-2]) / data[ticker].iloc[-2]
            scores[label] = {"val": current, "trend": "UP" if current > ma20 else "DOWN", "chg": change}
        
        # 如果大盘都在 MA20 上方，风控系数增加
        up_count = sum(1 for v in scores.values() if v['trend'] == "UP")
        total_weight = 0.7 + (up_count * 0.15)
    except:
        pass
    return total_weight, scores

m_weight, m_details = calculate_market_score()

# 显示大盘看板
cols = st.columns(4)
for i, (k, v) in enumerate(m_details.items()):
    color = "#00ff88" if v['trend'] == "UP" else "#ff4b4b"
    cols[i].markdown(f"""
    <div class="stat-box">
        <small>{k}</small><br>
        <b style="color:{color}; font-size:1.2rem;">{v['trend']}</b><br>
        <small>{v['chg']:.2%}</small>
    </div>
    """, unsafe_allow_html=True)

# --- 5. 交互诊断区域 ---
tab1, tab2 = st.tabs(["🔍 手动单兵诊断 (Max 5)", "📡 核心资产扫描"])

with tab1:
    user_input = st.text_input("输入股票代码 (空格分隔):", "600519.SS 300750.SZ AAPL NVDA 000001.SZ")
    if st.button("开始深度诊断"):
        snapshot = fetch_market_snapshot()
        tickers = user_input.replace(',', ' ').split()[:5]
        
        results = []
        for t in tickers:
            with st.spinner(f"分析中: {t}..."):
                res = diagnostic_engine(t, m_weight, snapshot)
                if res: results.append(res)
        
        if results:
            st.table(pd.DataFrame(results))
        else:
            st.error("未获取到有效数据，请检查代码格式（A股需后缀 .SS 或 .SZ）。")

with tab2:
    if st.button("启动核心资产 Top 20 扫描"):
        snapshot = fetch_market_snapshot()
        # 默认核心池
        core_pool = ["600519.SS", "300750.SZ", "601318.SS", "000858.SZ", "600036.SS", "600900.SS", "002594.SZ", "300059.SZ"]
        
        scan_results = []
        bar = st.progress(0)
        for i, t in enumerate(core_pool):
            res = diagnostic_engine(t, m_weight, snapshot)
            if res: scan_results.append(res)
            bar.progress((i + 1) / len(core_pool))
        
        if scan_results:
            df_scan = pd.DataFrame(scan_results).sort_values('综合评分', ascending=False)
            st.dataframe(df_scan.style.background_gradient(subset=['综合评分'], cmap='RdYlGn'))

with st.sidebar:
    st.markdown(f"### 🛡️ 风控系数: **{m_weight:.2f}**")
    st.info("该系数基于 QQQ, IWM, SPY 与 HS300 的均线位置计算。")
    st.write("---")
    st.markdown("""
    **模型说明：**
    1. **EV (Expected Value)**: 基于历史波动计算的每笔交易预期盈亏。
    2. **Random Forest**: 通过成交量比率、RSI和乖离率预测未来5日的潜在爆发力。
    3. **ATR Stop**: 自动根据波动率锁定止损。
    """)
