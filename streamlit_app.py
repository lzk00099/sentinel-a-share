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

# --- 1. CSS 渲染 ---
def get_v24_css():
    return """
    <style>
        .main-header { background: linear-gradient(135deg, #1a1a1a 0%, #800000 100%); color: #ffd700; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 1px solid #ffd700; }
        .env-card { background: #0e1117; color: #ffd700; border: 1px solid #333; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
        .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; }
        .stat-box { background: #1c1c1c; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #444; }
        .sidebar-box { background: #ffffff; padding: 15px; border-radius: 8px; border-left: 5px solid #cc0000; margin-bottom: 15px; color: #333; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .u-tips { font-size: 0.85rem; color: #666; line-height: 1.5; }
        .stButton>button { width: 100%; border-radius: 8px; background: #800000; color: white; border: none; transition: 0.3s; }
        .stButton>button:hover { background: #ffd700; color: #000; }
    </style>
    """

# --- 2. 数据处理函数 ---
@st.cache_data(ttl=3600)
def fetch_market_snapshot():
    try:
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            # 确保代码格式为6位字符串
            df['代码'] = df['代码'].astype(str).str.zfill(6)
            return df.set_index('代码')[['名称', '换手率']].to_dict('index')
    except: pass
    return {}

def get_north_flow(symbol):
    try:
        hsgt_df = ak.stock_hsgt_individual_em(symbol=symbol)
        if not hsgt_df.empty:
            return round(hsgt_df['当日净买入额'].iloc[0] / 10000, 2)
    except: pass
    return 0.0

# --- 3. 核心诊断逻辑 (强化名字抓取与特征容错) ---
def diagnostic_core(ticker, risk_weight, snapshot, include_pro=False, manual_name=None):
    try:
        symbol_6digit = ticker.split('.')[0]
        
        # 1. 历史行情抓取
        if ".SS" in ticker or ".SZ" in ticker:
            # A股优先用ak获取含换手率的历史
            df_hist = ak.stock_zh_a_hist(symbol=symbol_6digit, period="daily", adjust="qfq")
            if df_hist.empty or len(df_hist) < 60: return None
            # 列名统一映射
            df = df_hist.rename(columns={'日期':'Date','开盘':'Open','收盘':'Close','最高':'High','最低':'Low','成交量':'Volume','换手率':'Turnover'})
            df.set_index('Date', inplace=True)
        else:
            # 美股或其它标的
            df = yf.download(ticker, period="250d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 60: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df['Turnover'] = (df['Volume'] / df['Volume'].rolling(20).mean()) * 2 # 模拟换手

        # 2. 特征计算
        df['Vol_Ratio'] = (df['Volume'] / df['Volume'].rolling(5).mean()).fillna(1.0)
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Bias'] = ((df['Close'] - df['MA20']) / df['MA20']).fillna(0.0)
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean().fillna(df['Close']*0.02)
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = (100 - (100 / (1 + gain/loss))).fillna(50.0)
        
        if 'Turnover' not in df.columns: df['Turnover'] = 0.0
        df['Turnover'] = df['Turnover'].fillna(0.0)

        # 3. 机器学习训练
        df['Target'] = (df['High'].shift(-5).rolling(5).max() > df['Close'] * 1.06).astype(int)
        feats = ['Vol_Ratio', 'Bias', 'RSI', 'Turnover']
        train_data = df[feats + ['Target']].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(train_data) > 30:
            rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
            rf.fit(train_data[feats].iloc[:-5].values, train_data['Target'].iloc[:-5].values)
            win_p = float(rf.predict_proba(df[feats].iloc[[-1]].fillna(0).values)[0][1])
        else:
            win_p = 0.5 # 兜底逻辑

        ev = (win_p * 0.08) - ((1 - win_p) * 0.04)
        
        # 4. 权重修正 (Pro)
        north_val = "跳过"
        north_multiplier = 1.0
        if include_pro:
            north_val = get_north_flow(symbol_6digit)
            if isinstance(north_val, (int, float)):
                if north_val > 500: north_multiplier = 1.15
                elif north_val < -500: north_multiplier = 0.85
        
        score = win_p * ev * risk_weight * north_multiplier * 1000
        
        # 5. 【关键修复】多级名称获取逻辑
        curr_price = df['Close'].iloc[-1]
        atr_now = df['ATR'].iloc[-1]
        
        # 优先级：手动输入 > 快照匹配 > 接口直查 > 代码备用
        name = manual_name
        if not name:
            name = snapshot.get(symbol_6digit, {}).get('名称')
        if not name:
            try:
                info_df = ak.stock_individual_info_em(symbol=symbol_6digit)
                name = info_df[info_df['item'] == '股票简称']['value'].values[0]
            except:
                name = ticker

        real_turnover = snapshot.get(symbol_6digit, {}).get('换手率', df['Turnover'].iloc[-1])

        return {
            '名称': name, '代码': ticker, '现价': round(curr_price, 2),
            '预测胜率': f"{win_p:.1%}", '期望值(EV)': f"{ev*100:+.2f}%", 
            '周期': "5-10交易日", '建议买入': round(curr_price * 0.99, 2),
            '止盈参考': round(curr_price + (atr_now * 2.5), 2), 
            '止损建议': round(curr_price - (atr_now * 1.5), 2),
            '换手率': f"{real_turnover:.2f}%", '北向资金(万)': north_val,
            '综合评分': round(score, 2), 'Score_Raw': score 
        }
    except:
        return None

# --- 4. 界面渲染 (完整保留手册与界面) ---
st.markdown(get_v24_css(), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🛡️ SENTINEL A-Share V24 PRO</h1><p>全市场量化扫描系统 • 2026 机器学习增强版</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🧬 系统逻辑简介")
    st.markdown('<div class="sidebar-box">本系统核心为 <b>Hybrid-RF (混合随机森林)</b> 模型，专为 A 股 T+1 环境定制。<br><br><b>核心计算因子：</b><li><b>Bias/RSI:</b> 价格动能与强弱偏离度。</li><li><b>Vol_Ratio/Turnover:</b> <b>(全引擎加入)</b> 量能修正与换手活跃度计算，过滤无量诱多。</li><li><b>North_Flow:</b> <b>(Pro版/单兵版)</b> 北向资金流向作为决策权重增益。</li></div>', unsafe_allow_html=True)
    st.markdown("### 🕒 最佳运行时间")
    st.markdown('<div class="sidebar-box"><b>1. 盘前 (09:15):</b> 定调。<br><b>2. 盘中 (14:00):</b> 捕捉趋势。<br><b>3. 尾盘 (14:45):</b> 锁定期望值。</div>', unsafe_allow_html=True)
    st.markdown("### 🛠️ 操作手册")
    st.markdown('<div class="u-tips"><li><b>评分 > 10:</b> 强信号。</li><li><b>评分 5-10:</b> 观察。</li><li><b>评分 < 0:</b> 避险区域。</li><br><i>*止盈止损根据个股 ATR 动态调整。</i></div>', unsafe_allow_html=True)

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
                <div class="stat-box"><small>全球联动</small><br><b>监测中</b></div>
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
    if st.button("开始全量量化扫描"):
        snapshot = fetch_market_snapshot()
        try:
            df_300 = ak.index_stock_cons_csindex(symbol="000300")
            pool = { (f"{row['成分券代码']}.SS" if row['成分券代码'].startswith('60') else f"{row['成分券代码']}.SZ"): row['成分券名称']
                    for _, row in df_300.head(50).iterrows() }
        except:
            pool = {"600519.SS": "贵州茅台", "300750.SZ": "宁德时代"}
            
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, (t, n) in enumerate(pool.items()):
            status_text.text(f"正在扫描: {n}...")
            res = diagnostic_core(t, risk_weight, snapshot, include_pro=("Pro" in mode), manual_name=n)
            if res: results.append(res)
            progress_bar.progress((i + 1) / len(pool))
        
        status_text.success("扫描完成！")
        if results:
            df_final = pd.DataFrame(results).sort_values('Score_Raw', ascending=False).head(15)
            st.dataframe(df_final[DISPLAY_COLS].style.background_gradient(subset=['综合评分'], cmap='RdYlGn'), width='stretch')

with tab2:
    user_input = st.text_input("输入代码（上限5个）", "600519.SS 300750.SZ 601318.SS")
    if st.button("执行精准诊断"):
        snapshot = fetch_market_snapshot()
        tickers = user_input.replace(',', ' ').split()[:5]
        results = []
        for t in tickers:
            with st.spinner(f"分析中 {t}..."):
                res = diagnostic_core(t, risk_weight, snapshot, include_pro=True)
                if res: results.append(res)
        
        if results:
            df_user = pd.DataFrame(results).sort_values('Score_Raw', ascending=False)
            st.dataframe(df_user[DISPLAY_COLS].style.background_gradient(subset=['综合评分'], cmap='RdYlGn'), width='stretch')
