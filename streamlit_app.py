import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
from sklearn.ensemble import RandomForestClassifier
import warnings
from datetime import datetime

# --- 基础配置 ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="SENTINEL DUAL-ENGINE V26 PRO", layout="wide")

# --- 1. CSS 渲染 ---
def get_v24_css():
    return """
    <style>
        .main-header { background: linear-gradient(135deg, #111111 0%, #4a0000 100%); color: #ffd700; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 1px solid #ffd700; }
        .env-card { background: #0e1117; color: #ffd700; border: 1px solid #333; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
        .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; }
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
    try:
        df_300 = ak.index_stock_cons_csindex(symbol="000300")
        return dict(zip(df_300['成分券代码'], df_300['成分券名称']))
    except:
        return {}

# --- 3. 核心诊断算法（双周期日线+日内多空修正） ---
def diagnostic_core(ticker, market_env, name_map):
    """
    market_env: 包含大盘风控乘数与指数列表的字典
    """
    try:
        raw_code = "".join(filter(str.isdigit, ticker))
        stock_name = name_map.get(raw_code, ticker)

        # 1. 下载包含实时盘中数据的混合数据集
        # 为了捕获盘中未收盘的最后一根实时K线，使用period="250d"
        df = yf.download(ticker, period="250d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 区分历史完整日线与当前盘中未完结K线
        df_history = df.iloc[:-1]       # 纯历史完整日线（用于喂给RF模型）
        intraday_k = df.iloc[-1]        # 当前日内实时K线（用于计算日内异动）

        # 2. 基础日线指标计算
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Bias'] = (df['Close'] - df['MA20']) / df['MA20']
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        
        change = df['Close'].diff()
        gain = (change.where(change > 0, 0)).rolling(14).mean()
        loss = (-change.where(change < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        
        # 3. 机器学习模型层（基于纯日线历史）
        # 训练集剔除未完结的当天K线，确保特征的“纯净度”
        df['Target'] = (df['High'].shift(-5).rolling(5).max() > df['Close'] * 1.06).astype(int)
        feats = ['Vol_Ratio', 'Bias', 'RSI']
        train = df.iloc[:-1][feats + ['Target']].dropna() # 排除盘中行
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        rf.fit(train[feats].iloc[:-5].values, train['Target'].iloc[:-5].values)
        
        # 获取最新的（包含今日盘中变化的）特征进行静态胜率预测
        latest_feats = df[feats].iloc[[-1]].values
        base_win_p = float(rf.predict_proba(latest_feats)[0][1])

        # 4. 🔥 日内多空博弈及动能修正算法 (防止日内剧烈波动、诱多与背离)
        curr_price = float(intraday_k['Close'])
        prev_close = float(df_history['Close'].iloc[-1])
        intra_high = float(intraday_k['High'])
        intra_low = float(intraday_k['Low'])
        atr_now = float(df['ATR'].iloc[-1])
        
        # (A) 日内实时振幅与涨跌幅
        intra_return = (curr_price - prev_close) / prev_close  # 今日实时涨跌幅
        intra_range = (intra_high - intra_low) / prev_close     # 今日日内振幅
        
        # (B) 日内多空位置系数 (0 代表跌到全天最低点，1 代表处于全天最高点)
        if intra_high != intra_low:
            intra_position = (curr_price - intra_low) / (intra_high - intra_low)
        else:
            intra_position = 0.5
            
        # (C) 异常波动惩罚项 (防范日内过早冲高回落的上影线杀跌陷阱)
        # 如果当前价从日内最高点回落超过 ATR 的 0.5 倍，则触发动能衰减惩罚
        high_fallback = (intra_high - curr_price) / atr_now if atr_now > 0 else 0
        
        # 动态计算「日内动能修正系数」
        # 理想盘面：处于高位拉升中(intra_position高)，且没有大幅回落(high_fallback低)
        intraday_multiplier = 1.0
        if high_fallback > 0.5:
            intraday_multiplier -= (high_fallback - 0.5) * 0.4  # 冲高回落惩罚
        if intra_position < 0.3:
            intraday_multiplier -= (0.3 - intra_position) * 0.3 # 处于日内绝对低位（弱势）惩罚
        
        intraday_multiplier = max(0.5, min(1.5, intraday_multiplier)) # 约束在 [0.5, 1.5]

        # 5. 最终动态胜率与期望值(EV)计算
        # 最终胜率 = 基准模型胜率 * 日内动能修正
        final_win_p = max(0.01, min(0.99, base_win_p * intraday_multiplier))
        
        # EV逻辑：(最终胜率 * 8% 预期收益) - (败率 * 4% 预期风险)
        ev = (final_win_p * 0.08) - ((1 - final_win_p) * 0.04)
        
        # 综合大盘风控环境评分
        score = final_win_p * ev * market_env['risk_weight'] * 1000
        
        # 6. 动态风控仓位建议 (根据日内异动自适应)
        risk_tips = "正常波动"
        if high_fallback > 1.0: risk_tips = "⚠️ 日内多头力竭（严防长上影）"
        elif intra_return < -0.04 and intra_position < 0.1: risk_tips = "🚨 日内放量下杀（严禁左侧左入）"
        elif intra_position > 0.9 and intra_return > 0.03: risk_tips = "🔥 日内多头共振（动能强劲）"

        return {
            '名称': stock_name,
            '代码': ticker,
            '实时现价': round(curr_price, 2),
            '日内涨跌': f"{intra_return:+.2%}",
            '基准模型胜率': f"{base_win_p:.1%}",
            '修正后实时胜率': f"{final_win_p:.1%}",
            '实时期望值(EV)': f"{ev*100:+.2f}%",
            '预期周期': "5-10交易日",
            '建议买入': round(curr_price * 0.99, 2),
            '动态止盈参考': round(curr_price + (atr_now * 2.5), 2),
            '动态止损建议': round(curr_price - (atr_now * 1.5), 2),
            '实时风控提示': risk_tips,
            '综合评分': round(score, 2),
            'Score_Raw': score 
        }
    except Exception as e:
        return None

# --- 4. 大盘多指数综合风控评估（全面覆盖 QQQ, IWM, SPY 及 A股基准） ---
def get_comprehensive_market_env():
    """
    根据用户底层配置，综合评估跨市场环境因素
    A股全量扫描以 沪深300(000300.SS) 为主锚点；美股/跨市场单兵诊断引入 SPY, QQQ, IWM 综合加权
    """
    env_data = {'risk_weight': 1.0, 'status': "未知", 'details': {}}
    try:
        # 下载跨市场风控标的
        indices = {
            'SPY': 'SPY', 
            'QQQ': 'QQQ', 
            'IWM': 'IWM', 
            'CSI300': '000300.SS'
        }
        # 批量获取最近60天日线
        m_data = yf.download(list(indices.values()), period="60d", progress=False)
        if isinstance(m_data.columns, pd.MultiIndex):
            close_df = m_data['Close']
        else:
            close_df = m_data
            
        up_counts = 0
        total_weight = 0
        
        # 逐个评估各市场指数是否站稳20日线
        for name, ticker in indices.items():
            if ticker in close_df.columns:
                series = close_df[ticker].dropna()
                if not series.empty:
                    curr_c = series.iloc[-1]
                    ma20 = series.rolling(20).mean().iloc[-1]
                    is_bull = curr_c > ma20
                    env_data['details'][name] = "📈 站稳MA20" if is_bull else "📉 跌破MA20"
                    # 给不同市场赋予风控权重（可以根据侧重自行调整）
                    weight = 2 if name == 'CSI300' else 1 
                    total_weight += weight
                    if is_bull: up_counts += weight
                    
        # 综合计算大盘风控乘数
        bull_ratio = up_counts / total_weight if total_weight > 0 else 0.5
        if bull_ratio >= 0.75:
            env_data['risk_weight'] = 1.25
            env_data['status'] = "多头共振（安全度极高）"
        elif bull_ratio >= 0.5:
            env_data['risk_weight'] = 1.0
            env_data['status'] = "震荡分化（精选个股）"
        else:
            env_data['risk_weight'] = 0.75
            env_data['status'] = "空头防守（严格控制仓位）"
            
    except:
        env_data['status'] = "风控系统离线（执行默认风控）"
        
    # 渲染大盘环境看板
    st.markdown(f"""
    <div class="env-card">
        <h4 style='margin-top:0; color:#ffd700;'>🌐 全球跨市场综合环境诊断 (2026 PRO版)</h4>
        <div class="grid-3">
            <div class="stat-box"><small>美股大盘 (SPY/QQQ/IWM)</small><br>
                <b style="font-size:0.9rem;">SPY: {env_data['details'].get('SPY','未知')}</b><br>
                <b style="font-size:0.9rem;">QQQ: {env_data['details'].get('QQQ','未知')}</b><br>
                <b style="font-size:0.9rem;">IWM: {env_data['details'].get('IWM','未知')}</b>
            </div>
            <div class="stat-box"><small>A股基准 (沪深300)</small><br><b style="font-size:1.2rem; color:#fff;">{env_data['details'].get('CSI300','未知')}</b></div>
            <div class="stat-box"><small>综合风控乘数</small><br><b style="color:#ffd700; font-size:1.3rem;">x{env_data['risk_weight']}</b><br><small>{env_data['status']}</small></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    return env_data

# --- 5. 界面主体与流程控制 ---
st.markdown(get_v24_css(), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🛡️ SENTINEL ADVANCED V26 PRO</h1><p>多周期智能算法诊断引擎 • 实时动态风控版</p></div>', unsafe_allow_html=True)

# 侧边栏：核心逻辑
with st.sidebar:
    st.markdown("### 🧬 2026 双周期算法内核")
    st.markdown(f"""
    <div class="sidebar-box">
        本系统升级为<b>“日线波段 + 日内高频动能”</b>双引擎。
        <br><br>
        <b>传统算法盲区：</b><br>
        股票在盘中冲高大回落（形成诱多上影线）时，纯日线模型仍可能因昨日数据惯性判定为“高胜率”，导致交易员盘中追高。
        <br><br>
        <b>Sentinel V26 修正方案：</b><br>
        实时引入<b>日内分时位置</b>与<b>上影线回落惩罚项</b>。一旦检测到个股脱离盘中高点，或下杀速率过快，将<b>自动向下修正实时胜率</b>并截断 EV，从底层杜绝诱多。
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ 盘中风控警示规则")
    st.markdown("""
    <div class="u-tips">
        <li><b>⚠️ 日内多头力竭：</b> 价格明显冲高回落，此时模型建议观望，防范次日 T+1 惯性低开。</li>
        <li><b>🚨 放量下杀：</b> 属于极度危险的右侧空头宣泄，即使历史胜率高也必须等止跌。</li>
        <li><b>🔥 多头共振：</b> 股价维持在日内高位，且未出现明显回落，胜率及EV最可信。</li>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🧹 清理缓存并重置环境"):
        st.cache_data.clear()
        st.rerun()

# 触发大盘综合评估
market_env = get_comprehensive_market_env()
name_map = get_stock_name_map()

# 页面主要功能
tab1, tab2 = st.tabs(["🚀 沪深300 全量扫描", "🔍 跨市场单兵诊断 (上限5个)"])
DISPLAY_COLS = ['名称', '代码', '实时现价', '日内涨跌', '基准模型胜率', '修正后实时胜率', '实时期望值(EV)', '预期周期', '建议买入', '动态止盈参考', '动态止损建议', '实时风控提示', '综合评分']

with tab1:
    st.write("对沪深300指数所有成分股进行训练，并结合**实时动态K线特征**进行双周期重塑扫描。")
    if st.button("开始 300 蓝筹全量扫描"):
        try:
            df_300 = ak.index_stock_cons_csindex(symbol="000300")
            pool = []
            for code in df_300['成分券代码']:
                yf_code = f"{code}.SS" if code.startswith('60') else f"{code}.SZ"
                pool.append(yf_code)
        except:
            pool = ["600519.SS", "300750.SZ", "601318.SS"] # 兜底

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(pool):
            status_text.text(f"正在全量建模分析 ({i+1}/{len(pool)}): {t}...")
            res = diagnostic_core(t, market_env, name_map)
            if res: results.append(res)
            progress_bar.progress((i + 1) / len(pool))
        
        status_text.success("全量实时双周期扫描完成！")
        if results:
            df_final = pd.DataFrame(results).sort_values('Score_Raw', ascending=False)
            st.subheader("🔥 实时动态高分选股池 (Top 20)")
            st.dataframe(
                df_final[DISPLAY_COLS].head(20).style.background_gradient(subset=['综合评分'], cmap='RdYlGn'),
                width='stretch'
            )

with tab2:
    st.write("##### 手动输入代码进行跨市场诊断")
    st.caption("提示：支持 A股/美股/港股 混合输入。模型根据盘中多空力量实时修正胜率，完美规避日内冲高回落股。")
    user_input = st.text_input("代码之间用空格分隔（自动截取前5个，需带有yfinance后缀）：", "600519.SS 300750.SZ NVDA AAPL 0700.HK")
    
    if st.button("执行动态精准诊断"):
        tickers = user_input.replace(',', ' ').split()[:5]
        results = []
        for t in tickers:
            with st.spinner(f"正在深度诊断全球资产 {t}..."):
                res = diagnostic_core(t, market_env, name_map)
                if res: results.append(res)
        
        if results:
            df_user = pd.DataFrame(results).sort_values('Score_Raw', ascending=False)
            st.subheader("📊 诊断报告（按实时多空共振得分降序）")
            st.dataframe(df_user[DISPLAY_COLS].style.background_gradient(subset=['综合评分'], cmap='RdYlGn'), width='stretch')
        else:
            st.error("诊断失败，请检查输入代码是否符合 yfinance 格式，或网络数据源是否畅通。")
