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

# --- 2. 高效数据增强函数 ---
@st.cache_data(ttl=3600)
def fetch_market_snapshot():
    """带重试机制的快照抓取"""
    retry_count = 2
    for i in range(retry_count):
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                df['代码'] = df['代码'].astype(str).str.zfill(6)
                return df.set_index('代码')[['名称', '换手率']].to_dict('index')
        except:
            if i < retry_count - 1: time.sleep(1)
            continue
    return {}

def get_north_flow(symbol_6digit):
    """Pro版：获取北向资金"""
    try:
        hsgt_df = ak.stock_hsgt_individual_em(symbol=symbol_6digit)
        if not hsgt_df.empty:
            return round(hsgt_df['当日净买入额'].iloc[0] / 10000, 2)
    except:
        pass
    return 0.0

# --- 3. 核心诊断逻辑 ---
def diagnostic_core(ticker, risk_weight, snapshot, include_pro=False, manual_name=None):
    try:
        # 1. 基础行情下载
        time.sleep(0.1) 
        df = yf.download(ticker, period="250d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 2. 指标计算（原有因子）
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Bias'] = (df['Close'] - df['MA20']) / df['MA20']
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        atr_now = df['ATR'].iloc[-1]
        
        change = df['Close'].diff()
        gain = (change.where(change > 0, 0)).rolling(14).mean()
        loss = (-change.where(change < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        
        # --- 新增因子：换手率估算序列 ---
        symbol_6digit = ticker.split('.')[0]
        snap_data = snapshot.get(symbol_6digit, {})
        turnover_today = snap_data.get('换手率', 0.0)  # 当日换手率（百分比）
        
        # 用当日成交量作为基准，估算历史换手率（假设流通股本不变）
        latest_vol = df['Volume'].iloc[-1]
        if latest_vol > 0 and turnover_today > 0:
            df['Turnover'] = turnover_today * (df['Volume'] / latest_vol)
        else:
            df['Turnover'] = 0.0  # 若数据缺失，填充0
        
        # --- 新增因子：北向资金历史序列（Pro模式下启用）---
        if include_pro:
            north_today = get_north_flow(symbol_6digit)  # 当日净买入额（万元）
            # 由于获取完整历史序列耗时过长，此处简化：历史均填充为当日值
            # （实际应用中可替换为真实历史接口，此处保证特征列存在）
            df['North_Flow'] = north_today
        else:
            df['North_Flow'] = 0.0  # 非Pro模式用0填充
        
        # 3. 机器学习模型（动态构建特征列表）
        base_feats = ['Vol_Ratio', 'Bias', 'RSI']
        extra_feats = ['Turnover']  # 换手率始终加入（免费午餐）
        if include_pro:
            extra_feats.append('North_Flow')  # Pro模式增加北向资金因子
        
        feats = base_feats + extra_feats
        
        # 目标变量构建
        df['Target'] = (df['High'].shift(-5).rolling(5).max() > df['Close'] * 1.06).astype(int)
        train = df[feats + ['Target']].dropna()
        
        # 训练模型（注意排除最后5行避免标签泄露）
        rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        rf.fit(train[feats].iloc[:-5].values, train['Target'].iloc[:-5].values)
        
        # 预测最新一期的胜率
        win_p = float(rf.predict_proba(df[feats].iloc[[-1]].values)[0][1])
        ev = (win_p * 0.08) - ((1 - win_p) * 0.04)
        score = win_p * ev * risk_weight * 1000
        
        # 4. 价格建议
        curr_price = df['Close'].iloc[-1]
        tp_price = curr_price + (atr_now * 2.5)
        sl_price = curr_price - (atr_now * 1.5)
        
        # 5. 组装结果与多维名称修复
        # 名称判定逻辑
        name = "未知"
        if manual_name:
            name = manual_name
        elif snap_data.get('名称'):
            name = snap_data.get('名称')
        else:
            try:
                if ".SS" in ticker or ".SZ" in ticker:
                    info_df = ak.stock_individual_info_em(symbol=symbol_6digit)
                    name = info_df.iloc[0, 1]
            except:
                name = ticker
        
        turnover = snap_data.get('换手率', 0.0)
        north_val = "跳过"
        if include_pro:
            north_val = get_north_flow(symbol_6digit)

        return {
            '名称': name, '代码': ticker, '现价': round(curr_price, 2),
            '预测胜率': f"{win_p:.1%}", '期望值(EV)': f"{ev*100:+.2f}%", 
            '周期': "5-10交易日", '建议买入': round(curr_price * 0.99, 2),
            '止盈参考': round(tp_price, 2), '止损建议': round(sl_price, 2),
            '换手率': f"{turnover:.2f}%", '北向资金(万)': north_val,
            '综合评分': round(score, 2), 'Score_Raw': score 
        }
    except Exception as e:
        # 为调试方便可打印异常，生产环境保持静默
        # st.error(f"诊断 {ticker} 失败: {e}")
        return None

# --- 4. 界面渲染 ---
st.markdown(get_v24_css(), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🛡️ SENTINEL A-Share V24 PRO</h1><p>全市场量化扫描系统 • 2026 性能优化版</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🧬 系统逻辑简介")
    st.markdown('<div class="sidebar-box">本系统核心为 <b>Hybrid-RF (混合随机森林)</b> 模型，专为 A 股 T+1 环境定制。<br><br><b>核心因子：</b><li><b>Bias (量价乖离):</b> 监控价格回归动能。</li><li><b>RSI (强弱对比):</b> 评估超买超卖极端情绪。</li><li><b>Vol_Ratio (量能修正):</b> 过滤无量假拉升。</li><li><b>Turnover (换手率估算):</b> 捕捉流动性异常。</li><li><b>North_Flow (北向资金):</b> Pro版启用，跟踪聪明钱。</li><li><b>ATR (动态波幅):</b> 自动适配震荡空间。</div>', unsafe_allow_html=True)
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
            st.markdown(f"""
            <div class="env-card">
                <div class="grid-3">
                    <div class="stat-box"><small>沪深300</small><br><b>{m_close:.2f}</b></div>
                    <div class="stat-box"><small>全球联动</small><br><b>监测中</b></div>
                    <div class="stat-box"><small>风控乘数</small><br><b style="color:#ffd700;">x{risk_weight}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            return risk_weight
    except: pass
    return 0.8

risk_weight = get_market_env()
tab1, tab2 = st.tabs(["🚀 核心资产 Top 50 扫描", "🔍 跨市场标的单兵诊断"])
DISPLAY_COLS = ['名称', '代码', '现价', '预测胜率', '期望值(EV)', '周期', '建议买入', '止盈参考', '止损建议', '换手率', '北向资金(万)', '综合评分']

with tab1:
    mode = st.radio("选择扫描引擎", ["⚡ 极速轻量版 (核心系数 + ML)", "🧠 机器学习 Pro 版 (含北向资金深度扫描)"], horizontal=True)
    if st.button("开始全量量化扫描"):
        is_pro = "Pro" in mode
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
            status_text.text(f"正在扫描 ({i+1}/50): {n}...")
            res = diagnostic_core(t, risk_weight, snapshot, include_pro=is_pro, manual_name=n)
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
            with st.spinner(f"深度诊断 {t}..."):
                res = diagnostic_core(t, risk_weight, snapshot, include_pro=True)
                if res: results.append(res)
        
        if results:
            df_user = pd.DataFrame(results).sort_values('Score_Raw', ascending=False)
            st.dataframe(df_user[DISPLAY_COLS].style.background_gradient(subset=['综合评分'], cmap='RdYlGn'), width='stretch')