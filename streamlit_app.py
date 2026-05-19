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
st.set_page_config(page_title="SENTINEL A-SHARE DUAL-ENGINE V26", layout="wide")

# --- 1. 赛博朋克专属尊享级 UI 样式表 ---
def get_v26_css():
    return """
    <style>
        .main-header { background: linear-gradient(135deg, #0f0f14 0%, #5a0000 100%); color: #ffd700; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 1px solid #ffd700; box-shadow: 0 4px 15px rgba(255,215,0,0.1); }
        .env-card { background: #0e1117; color: #ffd700; border: 1px solid #444; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
        .grid-4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 15px; }
        .stat-box { background: #161a22; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #333; }
        .sidebar-box { background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #800000; margin-bottom: 15px; color: #222; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .u-tips { font-size: 0.85rem; color: #444; line-height: 1.6; list-style-type: square; padding-left: 15px; }
        .stButton>button { width: 100%; border-radius: 8px; background: #800000; color: white; border: none; transition: 0.3s; font-weight: bold;}
        .stButton>button:hover { background: #ffd700; color: #000; box-shadow: 0 0 10px rgba(255,215,0,0.5); }
    </style>
    """

# --- 2. 境内代码规范化辅助工具 ---
def normalize_a_share_code(raw_input):
    """
    清洗用户输入，过滤非A股资产，自动纠错补全后缀
    """
    ticker = raw_input.strip().upper()
    ticker = "".join(c for c in ticker if c.isalnum() or c in ['.', '-'])
    
    if ticker.isdigit() and len(ticker) == 6:
        if ticker.startswith(('60', '68', '90')):
            return f"{ticker}.SS"
        elif ticker.startswith(('00', '30', '20', '15', '16', '18')):
            return f"{ticker}.SZ"
    
    if ticker.endswith(('.SS', '.SZ')):
        return ticker
        
    return None

@st.cache_data(ttl=86400)
def get_stock_name_map():
    try:
        df_300 = ak.index_stock_cons_csindex(symbol="000300")
        return dict(zip(df_300['成分券代码'], df_300['成分券名称']))
    except:
        return {}

# --- 3. 核心双周期算法引擎（含杠杆ETF过滤机制） ---
def diagnostic_core(ticker, market_env, name_map):
    try:
        raw_code = "".join(filter(str.isdigit, ticker))
        stock_name = name_map.get(raw_code, "A股目标资产")

        # 1. 下载混合数据集 (包含实时未完结K线)
        df = yf.download(ticker, period="260d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        
        df_history = df.iloc[:-1]       # 纯历史完整日线
        intraday_k = df.iloc[-1]        # 当前日内实时K线

        # 2. 特征工程深度扩展
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Bias'] = (df['Close'] - df['MA20']) / df['MA20']
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        df['ATR_Pct'] = df['ATR'] / df['Close']
        
        # 计算 RSI
        change = df['Close'].diff()
        gain = (change.where(change > 0, 0)).rolling(14).mean()
        loss = (-change.where(change < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-6)))
        
        # 3. 🛡️ 智能检测：杠杆资产及 ETF 特殊过滤单元
        is_etf = ticker.startswith(('51', '56', '58', '15', '16')) or "ETF" in stock_name
        is_leveraged = "杠杆" in stock_name or "两倍" in stock_name or "3倍" in stock_name or ticker.startswith('150')
        
        atr_multiplier_tp = 2.5
        atr_multiplier_sl = 1.5
        cycle_desc = "5-10 交易日"
        
        if is_leveraged:
            atr_multiplier_tp = 3.5  
            atr_multiplier_sl = 2.0  
            cycle_desc = "2-5 交易日 (高频杠杆监控)"
        elif is_etf:
            atr_multiplier_tp = 1.8  
            atr_multiplier_sl = 1.2 
            cycle_desc = "2-3 周 (指数趋势跟踪)"

        # 4. 机器学习模型层
        df['Target'] = (df['High'].shift(-5).rolling(5).max() > df['Close'] + (df['ATR'] * 1.5)).astype(int)
        
        feats = ['Vol_Ratio', 'Bias', 'RSI', 'ATR_Pct']
        train = df.iloc[:-1][feats + ['Target']].dropna()
        
        rf = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42)
        rf.fit(train[feats].iloc[:-5].values, train['Target'].iloc[:-5].values)
        
        latest_feats = df[feats].iloc[[-1]].values
        base_win_p = float(rf.predict_proba(latest_feats)[0][1])

        # 5. 日内高频多空博弈修正
        curr_price = float(intraday_k['Close'])
        prev_close = float(df_history['Close'].iloc[-1])
        intra_high = float(intraday_k['High'])
        intra_low = float(intraday_k['Low'])
        atr_now = float(df['ATR'].iloc[-1])
        
        intra_return = (curr_price - prev_close) / prev_close
        if intra_high != intra_low:
            intra_position = (curr_price - intra_low) / (intra_high - intra_low)
        else:
            intra_position = 0.5
            
        high_fallback = (intra_high - curr_price) / (atr_now + 1e-6)
        
        intraday_multiplier = 1.0
        if high_fallback > 0.4:
            intraday_multiplier -= (high_fallback - 0.4) * 0.4  
        if intra_position < 0.3:
            intraday_multiplier -= (0.3 - intra_position) * 0.3 
            
        intraday_multiplier = max(0.5, min(1.4, intraday_multiplier))
        final_win_p = max(0.01, min(0.99, base_win_p * intraday_multiplier))
        
        # 6. 📐 自适应动态期望值 (EV) 数学计算
        tp_price = round(curr_price + (atr_now * atr_multiplier_tp), 2)
        sl_price = round(curr_price - (atr_now * atr_multiplier_sl), 2)
        
        pot_gain_pct = (tp_price - curr_price) / curr_price
        pot_loss_pct = (curr_price - sl_price) / curr_price
        
        ev = (final_win_p * pot_gain_pct) - ((1 - final_win_p) * pot_loss_pct)
        
        # 结合大盘得分
        score = final_win_p * ev * market_env['risk_weight'] * 1000
        
        # 7. 实时风控标志生成
        risk_tips = "盘面良性波动"
        if is_leveraged:
            risk_tips = "⚡ 杠杆工具：严防耗损与双向杀多"
        elif is_etf:
            risk_tips = "📦 跟踪基金：关注成分股分化"
            
        if high_fallback > 0.8: 
            risk_tips = "⚠️ 盘中多头崩溃 (谨防长上影诱多)"
        elif intra_return < -0.05 and intra_position < 0.15: 
            risk_tips = "🚨 机构无底线杀跌 (严禁左侧入场)"

        # 🚀 【核心修复点】这里的 Key 必须与下方的 DISPLAY_COLS 完美对齐
        return {
            '名称': stock_name,
            '代码': ticker,
            '实时现价': round(curr_price, 2),
            '日内涨跌': f"{intra_return:+.2%}",
            '基准胜率': f"{base_win_p:.1%}",
            '动态修正胜率': f"{final_win_p:.1%}",
            '数学期望值(EV)': f"{ev*100:+.2f}%",
            '预期周期': cycle_desc,
            '建议买入价': round(curr_price * 0.992, 2),
            '推荐止盈点': tp_price,
            '推荐止损点': sl_price,
            '实时风险提示': risk_tips,
            '综合核心评分': round(score, 2),
            'Score_Raw': score 
        }
    except Exception as e:
        return None

# --- 4. 境内多维大盘宏观环境风控矩阵 ---
def get_mainland_market_env():
    env_data = {'risk_weight': 1.0, 'status': "未知", 'details': {}}
    try:
        indices = {
            '沪深300 (核心资产)': '000300.SS', 
            '上证指数 (大盘权重)': '000001.SS', 
            '创业板指 (科技成长)': '399006.SZ', 
            '中证500 (中盘标杆)': '000905.SS'
        }
        
        m_data = yf.download(list(indices.values()), period="50d", progress=False)
        close_df = m_data['Close'] if isinstance(m_data.columns, pd.MultiIndex) else m_data
            
        up_counts = 0
        total_weight = 0
        
        for name, ticker in indices.items():
            if ticker in close_df.columns:
                series = close_df[ticker].dropna()
                if not series.empty:
                    curr_c = series.iloc[-1]      
                    ma20 = series.rolling(20).mean().iloc[-1]
                    is_bull = curr_c > ma20
                    
                    trend_icon = "📈" if is_bull else "📉"
                    env_data['details'][name] = f"{trend_icon} {curr_c:.2f} ({'站稳' if is_bull else '跌破'}MA20)"
                    
                    weight = 2 if "300" in name or "上证" in name else 1
                    total_weight += weight
                    if is_bull: up_counts += weight
                        
        bull_ratio = up_counts / total_weight if total_weight > 0 else 0.5
        
        if bull_ratio >= 0.8:
            env_data['risk_weight'] = 1.30
            env_data['status'] = "四盘多头共振（全多头环境，可积极主攻）"
        elif bull_ratio >= 0.5:
            env_data['risk_weight'] = 1.00
            env_data['status'] = "指数结构分化（震荡市，需精选个股风格）"
        else:
            env_data['risk_weight'] = 0.70
            env_data['status'] = "系统性多头退潮（严格控制总仓位，防范破位）"
            
    except:
        env_data['status'] = "境内风控墙离线（执行默认风控乘数）"
        
    st.markdown(f"""
    <div class="env-card">
        <h4 style='margin-top:0; color:#ffd700;'>🇨🇳 SENTINEL 境内多维宏观环境风控墙 (2026 实时点位版)</h4>
        <div class="grid-4">
            <div class="stat-box"><small>沪深300</small><br><b style="font-size:0.9rem; color:#fff;">{env_data['details'].get('沪深300 (核心资产)','数据维护中')}</b></div>
            <div class="stat-box"><small>上证指数</small><br><b style="font-size:0.9rem; color:#fff;">{env_data['details'].get('上证指数 (大盘权重)','数据维护中')}</b></div>
            <div class="stat-box"><small>创业板指</small><br><b style="font-size:0.9rem; color:#fff;">{env_data['details'].get('创业板指 (科技成长)','数据维护中')}</b></div>
            <div class="stat-box"><small>中证500</small><br><b style="font-size:0.9rem; color:#fff;">{env_data['details'].get('中证500 (中盘标杆)','数据维护中')}</b></div>
        </div>
        <div style="margin-top: 15px; text-align: center; border-top: 1px solid #333; padding-top: 10px;">
            <span>环境风控乘数：<b style="color:#ffd700; font-size:1.2rem;">x{env_data['risk_weight']}</b></span>
            <span style="margin-left:25px;">系统研判：<b style="color:#ff4b4b;">{env_data['status']}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    return env_data

# --- 5. 交互界面主体 ---
st.markdown(get_v26_css(), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>🛡️ SENTINEL A-SHARE ADVANCED V26</h1><p>A 股智能多周期算法引擎 • 期望值自适应版</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🧬 V26 境内自适应内核")
    st.markdown(f"""
    <div class="sidebar-box">
        <b>1. 摒弃外部噪音</b><br>
        本系统已完全阻断美股指数及港股的交叉干扰，风险乘数百分之百基于境内四大风格指数的趋势共振生成。
        <br><br>
        <b>2. 杠杆/ETF 特殊识别</b><br>
        代码一旦触发 15/16/51/56 等开头，模型将自动开启<b>“基金风控过滤法”</b>。
        <br><br>
        <b>3. 动态盈亏比数学期望</b><br>
        模型读取个股近期的平均真实波幅（ATR），自适应推演合乎个股基因的止盈止损点。
    </div>
    """, unsafe_allow_html=True)

    if st.button("🧹 清理底层缓存并重置风控墙"):
        st.cache_data.clear()
        st.rerun()

market_env = get_mainland_market_env()
name_map = get_stock_name_map()

tab1, tab2 = st.tabs(["🚀 沪深300 成分全量扫描", "🔍 A股单兵精准诊断 (上限5个)"])

# 🚀 【核心修复点】前端展示字段与核心计算字典 Key 必须严格一致
DISPLAY_COLS = [
    '名称', '代码', '实时现价', '日内涨跌', 
    '基准胜率', '动态修正胜率', 
    '数学期望值(EV)', '预期周期', '建议买入价', 
    '推荐止盈点', '推荐止损点', '实时风险提示', '综合核心评分'
]

with tab1:
    st.write("对境内核心资产沪深300全量成分股进行动态建模，自动重组高频动能得分池。")
    if st.button("启动 300 蓝筹全量量化扫描"):
        try:
            df_300 = ak.index_stock_cons_csindex(symbol="000300")
            pool = []
            for code in df_300['成分券代码']:
                yf_code = f"{code}.SS" if code.startswith('60') else f"{code}.SZ"
                pool.append(yf_code)
        except:
            pool = ["600519.SS", "300750.SZ", "601318.SS", "000001.SZ"] 

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(pool):
            status_text.text(f"正在全量建模分析 ({i+1}/{len(pool)}): {t}...")
            res = diagnostic_core(t, market_env, name_map)
            if res: results.append(res)
            progress_bar.progress((i + 1) / len(pool))
        
        status_text.success("A股核心资产全量实时双周期扫描完成！")
        if results:
            df_final = pd.DataFrame(results).sort_values('Score_Raw', ascending=False)
            st.subheader("🔥 实时动态高期望值选股池 (Top 20)")
            st.dataframe(
                df_final[DISPLAY_COLS].head(20).style.background_gradient(subset=['综合核心评分'], cmap='RdYlGn'),
                width='stretch'
            )

with tab2:
    st.write("##### 手动输入中国 A 股代码进行精准深度诊断")
    user_input = st.text_input(
        "请输入代码（空格分隔，最多支持5个）：", 
        "600519 300750 000001.SZ 159915 510300"
    )
    
    if st.button("执行动态自适应诊断"):
        raw_tickers = user_input.replace(',', ' ').split()
        cleaned_tickers = []
        
        for r_t in raw_tickers:
            normed = normalize_a_share_code(r_t)
            if normed:
                cleaned_tickers.append(normed)
        
        final_tickers = cleaned_tickers[:5]
        
        if not final_tickers:
            st.error("请输入有效的 A 股股票代码（如 600519 或 000001.SZ）。已拦截非 A 股资产。")
        else:
            results = []
            for t in final_tickers:
                with st.spinner(f"正在深度诊断境内资产 {t}..."):
                    res = diagnostic_core(t, market_env, name_map)
                    if res: results.append(res)
            
            if results:
                df_user = pd.DataFrame(results).sort_values('Score_Raw', ascending=False)
                st.subheader("📊 A 股多周期量化诊断报告（按实时多空得分降序）")
                st.dataframe(
                    df_user[DISPLAY_COLS].style.background_gradient(subset=['综合核心评分'], cmap='RdYlGn'), 
                    width='stretch'
                )
            else:
                st.error("诊断失败：未能成功获取对应资产数据。")
