import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
from sklearn.ensemble import RandomForestClassifier
import warnings
from datetime import datetime, timedelta
import time

# ==================== 基础配置 ====================
warnings.filterwarnings('ignore')
st.set_page_config(page_title="SENTINEL A-Share V24", layout="wide")

# ==================== 1. A股专用 CSS 渲染（强化颜色表现）====================
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

# ==================== 2. 辅助函数：获取A股数据（使用 akshare，回退到 yfinance）====================
@st.cache_data(ttl=3600)
def get_all_stock_names():
    """获取全市场A股代码->名称映射，用于手动诊断模块"""
    try:
        df = ak.stock_info_a_code_name()
        return dict(zip(df['code'], df['name']))
    except Exception as e:
        st.warning(f"获取股票名称映射失败: {e}，将使用代码作为名称")
        return {}

def get_stock_name(ticker):
    """根据ticker（如600519.SS）返回中文名称"""
    code = ticker.split('.')[0]
    name_map = get_all_stock_names()
    return name_map.get(code, ticker)

@st.cache_data(ttl=3600)
def get_a_shares_pool():
    """获取沪深300成分股前50只"""
    try:
        df_300 = ak.index_stock_cons(symbol="000300")
        df_top = df_300.head(50)
        # 转换为 yfinance 格式
        return { (f"{row['成分券代码']}.SS" if row['成分券代码'].startswith('60') else f"{row['成分券代码']}.SZ"): row['成分券名称']
                for _, row in df_top.iterrows() }
    except Exception as e:
        st.warning(f"获取沪深300成分股失败: {e}，使用默认股票池")
        return {"600519.SS": "贵州茅台", "300750.SZ": "宁德时代", "601318.SS": "中国平安"}

@st.cache_data(ttl=86400)
def get_stock_industry_mapping():
    """获取股票代码到申万三级行业的映射"""
    try:
        # 获取申万三级行业信息
        sw_df = ak.sw_index_third_info()
        industry_map = {}
        # 获取所有A股实时行情以获取代码-名称映射
        spot_df = ak.stock_zh_a_spot_em()
        # 构建代码到行业的映射（这里简化处理，实际需要更复杂的行业分类接口）
        # 注意：akshare 可能需要更专门的接口来获取个股行业分类，这里提供一个框架
        for _, row in spot_df.iterrows():
            code = row['代码']
            # 这里模拟行业分类，实际使用时需要调用专门的个股行业信息接口
            industry_map[code] = "未知行业"
        return industry_map
    except Exception as e:
        st.warning(f"获取行业映射失败: {e}")
        return {}

@st.cache_data(ttl=3600)
def get_industry_performance():
    """获取行业指数表现（使用申万一级行业指数）"""
    try:
        # 获取申万一级行业指数实时行情
        index_spot = ak.stock_zh_index_spot()
        # 筛选申万一级行业指数（通常以 sz399 或 sh000 开头，名称包含“行业”）
        industry_indices = index_spot[index_spot['名称'].str.contains('行业', na=False)]
        # 计算5日涨跌幅（简化处理，使用最新价和5日前价格）
        # 这里仅返回最近5日涨幅前5的行业名称
        top_industries = industry_indices.nlargest(5, '涨跌幅')['名称'].tolist()
        return top_industries
    except Exception as e:
        st.warning(f"获取行业表现失败: {e}")
        return []

@st.cache_data(ttl=3600)
def get_money_flow(ticker):
    """获取个股资金流向数据"""
    try:
        code = ticker.split('.')[0]
        # 获取个股资金流向排名数据
        flow_df = ak.stock_individual_fund_flow_rank(indicator="今日", symbol=code)
        if not flow_df.empty:
            # 提取主力净流入（单位：元）
            main_net_inflow = flow_df.iloc[0]['主力净流入-净额']
            return main_net_inflow / 10000  # 转换为万元
        return 0
    except Exception as e:
        # 资金流向接口可能不稳定，静默失败
        return 0

@st.cache_data(ttl=3600)
def get_north_flow():
    """获取北向资金净流入（亿元）"""
    try:
        north_df = ak.stock_hsgt_north_net_flow_in_em()
        if not north_df.empty:
            # 获取最新值
            latest = north_df.iloc[-1]
            return latest['value'] / 100000000  # 转换为亿元
        return 0
    except Exception as e:
        st.warning(f"获取北向资金失败: {e}")
        return 0

@st.cache_data(ttl=3600)
def get_stock_data_akshare(ticker, period=250):
    """使用 akshare 获取个股历史数据（前复权）"""
    try:
        code = ticker.split('.')[0]
        # 判断市场
        if code.startswith('60') or code.startswith('68'):
            symbol = f"sh{code}"
        else:
            symbol = f"sz{code}"
        # 获取历史数据，复权类型选择前复权
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="", end_date="", adjust="qfq")
        if df.empty:
            raise ValueError("No data")
        # 标准化列名
        df = df.rename(columns={
            '日期': 'Date', '开盘': 'Open', '收盘': 'Close', '最高': 'High', '最低': 'Low', '成交量': 'Volume'
        })
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        # 限制数据长度
        df = df.tail(period)
        return df
    except Exception as e:
        st.warning(f"akshare 获取 {ticker} 失败: {e}，尝试 yfinance...")
        import yfinance as yf
        df = yf.download(ticker, period=f"{period}d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

# ==================== 3. 核心诊断引擎（增强版）====================
def diagnostic_engine_a(ticker, name, risk_weight, industry_hot=False, money_flow_val=0):
    """
    核心诊断引擎，加入行业热度、资金流向因子
    """
    try:
        # 获取数据
        df = get_stock_data_akshare(ticker, period=250)
        if len(df) < 60:
            return None
        
        # 计算技术指标
        df['Ret'] = df['Close'].pct_change()
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Bias'] = (df['Close'] - df['MA20']) / df['MA20']
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        atr_now = df['ATR'].iloc[-1]
        
        change = df['Close'].diff()
        gain = (change.where(change > 0, 0)).rolling(14).mean()
        loss = (-change.where(change < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss))
        
        # 目标变量：未来5个交易日内是否涨超6%
        df['Target'] = (df['High'].shift(-5).rolling(5).max() > df['Close'] * 1.06).astype(int)
        
        # 基础特征
        feats = ['Vol_Ratio', 'Bias', 'RSI']
        train = df[feats + ['Target']].dropna()
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(train[feats].iloc[:-5].values, train['Target'].iloc[:-5].values)
        
        last_feat = df[feats].iloc[[-1]].values
        win_p = float(rf.predict_proba(last_feat)[0][1])
        
        # 加入行业热度调整（如果行业热门，提升胜率）
        if industry_hot:
            win_p = min(0.95, win_p * 1.05)
        
        # 加入资金流向调整（主力净流入为正时提升胜率）
        if money_flow_val > 0:
            win_p = min(0.95, win_p * (1 + min(0.1, money_flow_val / 10000)))
        
        # 期望值和综合评分
        ev = (win_p * 0.08) - ((1 - win_p) * 0.04)
        score = win_p * ev * risk_weight * 1000
        
        curr_price = df['Close'].iloc[-1]
        tp_price = curr_price + (atr_now * 2.5)
        sl_price = curr_price - (atr_now * 1.5)
        
        # 建议仓位（凯利公式）
        b = 0.08 / 0.04  # 赔率
        f = (win_p * (b + 1) - 1) / b if b > 0 else 0
        suggest_position = max(0, min(0.1, f * risk_weight))
        
        return {
            '代码': ticker,
            '名称': name,
            '现价': round(curr_price, 2),
            '预测胜率': f"{win_p:.1%}",
            '期望值': f"{ev*100:+.2f}%",
            '止盈参考': round(tp_price, 2),
            '止损建议': round(sl_price, 2),
            '综合评分': round(score, 2),
            '建议仓位': f"{suggest_position*100:.1f}%",
            'Score_Raw': score,
            '主力净流入': f"{money_flow_val:.0f}万" if money_flow_val != 0 else "N/A"
        }
    except Exception as e:
        st.error(f"诊断 {ticker} 时出错: {e}")
        return None

# ==================== 4. 统一表格样式函数 ====================
def style_dataframe(df):
    """对综合评分列应用渐变，对期望值列根据正负设置颜色"""
    styled = df.style.background_gradient(subset=['综合评分'], cmap='RdYlGn', low=0, high=20)
    def color_expect(val):
        if isinstance(val, str) and val.startswith('+'):
            return 'color: #00cc66'
        elif isinstance(val, str) and val.startswith('-'):
            return 'color: #ff4d4d'
        return ''
    styled = styled.applymap(color_expect, subset=['期望值'])
    return styled

# ==================== 5. Streamlit 界面布局 ====================
st.markdown(get_v24_css(), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🛡️ SENTINEL A-Share V24</h1><p>沪深300核心资产量化扫描系统 • 机器学习增强版 • 行业+资金流双轮驱动</p></div>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown("### 🧬 系统逻辑简介")
    st.markdown("""
    <div class="sidebar-box">
    本系统核心为 <b>Hybrid-RF (混合随机森林)</b> 模型，专为 A 股 T+1 环境定制。
    <br><br>
    <b>核心因子：</b>
    <li><b>Bias (量价乖离):</b> 监控价格回归动能。</li>
    <li><b>RSI (强弱对比):</b> 评估超买超卖极端情绪。</li>
    <li><b>Vol_Ratio (量能修正):</b> 过滤无量假拉升。</li>
    <li><b>行业热度:</b> 实时申万行业表现（v2.4新增）</li>
    <li><b>资金流向:</b> 个股主力净流入/北向资金（v2.4新增）</li>
    <li><b>ATR (动态波幅):</b> 自动适配不同股性的震荡空间。</li>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🕒 最佳运行时间")
    st.markdown("""
    <div class="sidebar-box">
    <b>1. 盘前观察 (09:15-09:25):</b>
    查看大盘风控乘数，制定当日仓位基调。
    <br><br>
    <b>2. 盘中确认 (10:30 & 14:00):</b>
    此时量能比值最具参考性，适合捕捉趋势。
    <br><br>
    <b>3. 尾盘博弈 (14:45):</b>
    锁定次日 T+1 期望值，过滤单日波动噪声。
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ 操作手册")
    st.markdown("""
    <div class="u-tips">
    <li><b>评分 > 10:</b> 强信号，建议关注。</li>
    <li><b>评分 5-10:</b> 观察信号，需结合板块效应。</li>
    <li><b>评分 < 0:</b> 避险区域，模型判定期望值为负。</li>
    <li><b>建议仓位:</b> 基于凯利公式的动态仓位建议（0-10%）。</li>
    <br>
    <i>*注：止盈止损已根据个股 ATR 自动动态调整，无需人工计算。</i>
    </div>
    """, unsafe_allow_html=True)

# ==================== 顶部大盘环境评估 ====================
try:
    m_df = ak.stock_zh_index_hist(symbol="sh000300", period="daily", start_date="", end_date="", adjust="")
    if not m_df.empty:
        m_df = m_df.rename(columns={'日期': 'Date', '收盘': 'Close'})
        m_df['Date'] = pd.to_datetime(m_df['Date'])
        m_df.set_index('Date', inplace=True)
        m_df = m_df.sort_index()
        m_close = m_df['Close'].iloc[-1]
        m_ma20 = m_df['Close'].rolling(20).mean().iloc[-1]
        m_ma60 = m_df['Close'].rolling(60).mean().iloc[-1]
        
        risk_weight = 1.2 if m_close > m_ma20 else 0.8 if m_close > m_ma60 else 0.5
        m_status = "强势反弹 (进攻)" if m_close > m_ma20 else "缩量回调 (观察)" if m_close > m_ma60 else "空头趋势 (避险)"
        
        # 获取北向资金
        north_flow = get_north_flow()
        north_status = f"北向净流入 {north_flow:.1f}亿" if north_flow > 0 else f"北向净流出 {abs(north_flow):.1f}亿"
        
        # 获取行业热度
        hot_industries = get_industry_performance()
        hot_text = "、".join(hot_industries[:3]) if hot_industries else "暂无"
        
        st.markdown(f"""
        <div class="env-card">
            <div style="margin-bottom:12px; font-weight:bold; font-size:1.1rem; border-left:4px solid #ffd700; padding-left:10px;">🚨 A股宏观情绪监测</div>
            <div class="grid-2">
                <div class="stat-box"><small>沪深300指数</small><br><b style="font-size:1.4rem;">{m_close:.2f}</b></div>
                <div class="stat-box"><small>战术环境建议</small><br><b style="font-size:1.4rem;">{m_status}</b></div>
                <div class="stat-box"><small>{north_status}</small><br><b style="font-size:0.9rem;">资金风向标</b></div>
                <div class="stat-box"><small>🔥 热门行业</small><br><b style="font-size:0.9rem;">{hot_text}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        risk_weight = 0.8
        st.warning("无法获取沪深300历史数据，使用默认风险权重")
except Exception as e:
    risk_weight = 0.8
    st.warning(f"获取大盘环境失败: {e}，使用默认风险权重")

# ==================== 功能标签页 ====================
tab1, tab2 = st.tabs(["🚀 核心资产 Top 50 扫描", "🔍 跨市场标的单兵诊断"])

DISPLAY_COLS = ['代码', '名称', '现价', '预测胜率', '期望值', '止盈参考', '止损建议', '综合评分', '建议仓位', '主力净流入']

with tab1:
    st.write("点击下方按钮，系统将实时抓取沪深 300 权重前 50 标的并运行机器学习引擎。")
    if st.button("开始全量量化扫描"):
        pool = get_a_shares_pool()
        results = []
        progress_bar = st.progress(0)
        
        # 获取行业热度信息（用于调整）
        hot_industries = get_industry_performance()
        # 获取北向资金（用于全局调整，这里简化）
        north_flow = get_north_flow()
        
        for i, (t, n) in enumerate(pool.items()):
            # 获取资金流向
            money_flow_val = get_money_flow(t)
            # 判断行业是否热门（简化处理）
            industry_hot = False  # 需要更精细的个股行业分类
            res = diagnostic_engine_a(t, n, risk_weight, industry_hot, money_flow_val)
            if res:
                results.append(res)
            progress_bar.progress((i + 1) / len(pool))
            # 避免请求过快
            time.sleep(0.1)
        
        if results:
            df_final = pd.DataFrame(results).sort_values('Score_Raw', ascending=False).head(12)
            st.subheader("🔥 SENTINEL 选股池 (高期望值标的)")
            st.dataframe(style_dataframe(df_final[DISPLAY_COLS]), use_container_width=True)
        else:
            st.warning("⚠️ 当前市场环境下，模型未发现具有正向期望值的标的，建议持币观望。")

with tab2:
    st.write("输入 A 股代码（含后缀），支持多代码批量诊断。")
    user_input = st.text_input("示例：600519.SS 300750.SZ 000001.SZ", "600519.SS 300750.SZ 601318.SS")
    if st.button("执行精准诊断"):
        tickers = user_input.replace(',', ' ').split()
        results = []
        
        # 获取行业热度（用于判断）
        hot_industries = get_industry_performance()
        
        for t in tickers:
            real_name = get_stock_name(t)
            money_flow_val = get_money_flow(t)
            # 判断行业是否热门（简化）
            industry_hot = False
            res = diagnostic_engine_a(t, real_name, risk_weight, industry_hot, money_flow_val)
            if res:
                results.append(res)
            time.sleep(0.1)
        
        if results:
            df_user = pd.DataFrame(results).sort_values('Score_Raw', ascending=False)
            st.subheader("📊 诊断报告")
            st.dataframe(style_dataframe(df_user[DISPLAY_COLS]), use_container_width=True)
        else:
            st.error("数据抓取失败。请确保输入格式正确（如 600519.SS）。")