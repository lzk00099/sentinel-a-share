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

# --- 1. CSS 渲染 (严格还原你的风格) ---
def get_v24_css():
    return """
    <style>
        .main-header { background: linear-gradient(135deg, #1a1a1a 0%, #800000 100%); color: #ffd700; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 1px solid #ffd700; }
        .env-card { background: #0e1117; color: #ffd700; border: 1px solid #333; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .stat-box { background: #1c1c1c; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #444; }
        .sidebar-box { background: #ffffff; padding: 15px; border-radius: 8px; border-left: 5px solid #cc0000; margin-bottom: 15px; color: #333; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .u-tips { font-size: 0.85rem; color: #666; line-height: 1.5; }
        .stButton>button { width: 100%; border-radius: 8px; background: #800000; color: white; border: none; transition: 0.3s; }
        .stButton>button:hover { background: #ffd700; color: #000; }
    </style>
    """

# --- 2. 增强型名称映射器 ---
@st.cache_data(ttl=86400)
def get_stock_name_map():
    """获取沪深300成分股代码与名称的映射，确保名称显示正确"""
    try:
        df_300 = ak.index_stock_cons_csindex(symbol="000300")
        # 建立 {'600519': '贵州茅台'} 这种映射
        return dict(zip(df_300['成分券代码'], df_300['成分券名称']))
    except:
        return {}

# --- 3. 核心诊断逻辑 (Hybrid-RF) ---
def diagnostic_core(ticker, risk_weight, name_map):
    try:
        # 提取纯数字代码用于名称匹配
        raw_code = "".join(filter(str.isdigit, ticker))
        stock_name = name_map.get(raw_code, ticker) # 找不到就显示代码

        # 1. 行情下载 (yf 为准)
        df = yf.download(ticker, period="250d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 2. 指标计算 (严格保留你的逻辑)
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Bias'] = (df['Close'] - df['MA20']) / df['MA20']
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        atr_now = df['ATR'].iloc[-1]
        
        change = df['Close'].diff()
        gain = (change.where(change > 0, 0)).rolling(14).mean()
        loss = (-change.where(change < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        
        # 3. 机器学习模型 (Random Forest)
        # Target: 未来5日内最高价触及 6% 涨幅
        df['Target'] = (df['High'].shift(-5).rolling(5).max() > df['Close'] * 1.06).astype(int)
        feats = ['Vol_Ratio', 'Bias', 'RSI']
        train = df[feats + ['Target']].dropna()
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        rf.fit(train[feats].iloc[:-5].values, train['Target'].iloc[:-5].values)
        
        # 预测当前胜率
        win_p = float(rf.predict_proba(df[feats].iloc[[-1]].values)[0][1])
        # EV逻辑：(胜率 * 8% 预期收益) - (败率 * 4% 预期风险)
        ev = (win_p * 0.08) - ((1 - win_p) * 0.04)
        score = win_p * ev * risk_weight * 1000
        
        curr_price = df['Close'].iloc[-1]

        return {
            '名称': stock_name,
            '代码': ticker,
            '现价': round(curr_price, 2),
            '预测胜率': f"{win_p:.1%}",
            '期望值(EV)': f"{ev*100:+.2f}%",
            '周期': "5-10交易日",
            '建议买入': round(curr_price * 0.99, 2),
            '止盈参考': round(curr_price + (atr_now * 2.5), 2),
            '止损建议': round(curr_price - (atr_now * 1.5), 2),
            '综合评分': round(score, 2),
            'Score_Raw': score 
        }
    except:
        return None

# --- 4. 界面渲染 ---
st.markdown(get_v24_css(), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🛡️ SENTINEL A-Share V24 PRO</h1><p>全市场量化扫描系统 • 纯净数据版</p></div>', unsafe_allow_html=True)

# 侧边栏：使用手册与逻辑
with st.sidebar:
    st.markdown("### 🧬 系统逻辑简介")
    st.markdown('<div class="sidebar-box">本系统核心为 <b>Hybrid-RF (混合随机森林)</b> 模型。<br><br><b>核心因子：</b><li><b>Bias:</b> 监控价格回归动能。</li><li><b>RSI:</b> 评估强弱对比。</li><li><b>Vol_Ratio:</b> 过滤无量假拉升。</li><li><b>ATR:</b> 自动适配震荡空间。</div>', unsafe_allow_html=True)
    st.markdown("### 🕒 最佳运行时间")
    st.markdown('<div class="sidebar-box"><b>1. 盘前:</b> 趋势定调。<br><b>2. 盘中:</b> 动能监控。<br><b>3. 尾盘:</b> 锁定期望值。</div>', unsafe_allow_html=True)
    st.markdown("### 🛠️ 操作手册")
    st.markdown('<div class="u-tips"><li>评分 > 10 为强信号。</li><li>止盈止损采用动态 ATR 逻辑。</li><li>全量扫描针对沪深300指数全样本。</li></div>', unsafe_allow_html=True)

# 大盘环境诊断 (仅限大陆股指)
def get_market_env():
    # 使用沪深300作为环境锚点
    m_df = yf.download("000300.SS", period="60d", progress=False)
    if not m_df.empty:
        if isinstance(m_df.columns, pd.MultiIndex): m_df.columns = m_df.columns.get_level_values(0)
        m_close = m_df['Close'].iloc[-1]
        m_ma20 = m_df['Close'].rolling(20).mean().iloc[-1]
        
        # 趋势强度计算
        trend_status = "上升趋势" if m_close > m_ma20 else "调整趋势"
        risk_weight = 1.2 if m_close > m_ma20 else 0.8
        
        st.markdown(f"""
        <div class="env-card">
            <div class="grid-2">
                <div class="stat-box"><small>沪深 300 基准</small><br><b style="font-size:1.5rem;">{m_close:.2f}</b></div>
                <div class="stat-box"><small>大盘风控乘数</small><br><b style="color:#ffd700; font-size:1.5rem;">x{risk_weight} ({trend_status})</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return risk_weight
    return 1.0

risk_weight = get_market_env()
name_map = get_stock_name_map()

# 页面主要功能
tab1, tab2 = st.tabs(["🚀 沪深300 全量扫描", "🔍 跨市场单兵诊断"])
DISPLAY_COLS = ['名称', '代码', '现价', '预测胜率', '期望值(EV)', '周期', '建议买入', '止盈参考', '止损建议', '综合评分']

with tab1:
    st.write("点击下方按钮，对沪深300指数所有成分股进行 Hybrid-RF 建模扫描。")
    if st.button("开始 300 蓝筹全量扫描"):
        try:
            df_300 = ak.index_stock_cons_csindex(symbol="000300")
            # 格式化 yf 代码
            pool = []
            for code in df_300['成分券代码']:
                yf_code = f"{code}.SS" if code.startswith('60') else f"{code}.SZ"
                pool.append(yf_code)
        except:
            pool = ["600519.SS", "300750.SZ"] # 兜底

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(pool):
            status_text.text(f"正在分析 ({i+1}/300): {t}...")
            res = diagnostic_core(t, risk_weight, name_map)
            if res: results.append(res)
            progress_bar.progress((i + 1) / len(pool))
        
        status_text.success("全量扫描完成！")
        if results:
            df_final = pd.DataFrame(results).sort_values('Score_Raw', ascending=False)
            st.subheader("🔥 SENTINEL 选股池 (高评分 Top 20)")
            st.dataframe(
                df_final[DISPLAY_COLS].head(20).style.background_gradient(subset=['综合评分'], cmap='RdYlGn'),
                width='stretch'
            )

with tab2:
    st.write("手动输入代码（支持 A股/美股/港股，上限5个）。")
    user_input = st.text_input("示例：601318.SS 000001.SZ NVDA 0700.HK", "600519.SS 300750.SZ 601318.SS")
    if st.button("执行单兵精准诊断"):
        tickers = user_input.replace(',', ' ').split()[:5]
        results = []
        for t in tickers:
            with st.spinner(f"正在诊断 {t}..."):
                res = diagnostic_core(t, risk_weight, name_map)
                if res: results.append(res)
        
        if results:
            df_user = pd.DataFrame(results).sort_values('Score_Raw', ascending=False)
            st.dataframe(df_user[DISPLAY_COLS].style.background_gradient(subset=['综合评分'], cmap='RdYlGn'), width='stretch')
        else:
            st.error("诊断失败，请检查输入代码是否符合 yfinance 格式。")
