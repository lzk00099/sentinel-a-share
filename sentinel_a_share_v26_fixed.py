import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
from sklearn.ensemble import RandomForestClassifier
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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
    ticker = str(raw_input).strip().upper()
    ticker = ticker.replace(".SH", ".SS")
    if ticker.endswith(".0") and ticker[:-2].isdigit():
        ticker = ticker[:-2]
    ticker = "".join(c for c in ticker if c.isalnum() or c in ['.', '-'])
    
    if ticker.isdigit() and len(ticker) <= 6:
        ticker = ticker.zfill(6)
        # 沪市：主板/科创板/B股/常见沪市 ETF
        if ticker.startswith(('60', '68', '90', '50', '51', '52', '56', '58')):
            return f"{ticker}.SS"
        # 深市：主板/创业板/B股/常见深市 ETF、LOF
        elif ticker.startswith(('00', '30', '20', '15', '16', '18')):
            return f"{ticker}.SZ"
    
    if ticker.endswith(('.SS', '.SZ')):
        return ticker
        
    return None

def run_with_timeout(func, timeout_seconds=25):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func)
    try:
        return future.result(timeout=timeout_seconds)
    except (TimeoutError, Exception):
        return None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

def get_constituent_columns(df):
    # 增加英文键名 'code', 'name' 以应对 API 上游暗改
    code_candidates = ['成分券代码', '证券代码', '品种代码', '股票代码', '代码', 'code']
    name_candidates = ['成分券名称', '证券简称', '品种名称', '股票简称', '名称', 'name']
    
    code_col = next((col for col in code_candidates if col in df.columns), None)
    name_col = next((col for col in name_candidates if col in df.columns), None)
    
    return code_col, name_col

def normalize_constituent_frame(df):
    code_col, name_col = get_constituent_columns(df)
    if df is None or df.empty or code_col is None:
        return pd.DataFrame(columns=['成分券代码', '成分券名称'])
    
    out = pd.DataFrame()
    out['成分券代码'] = (
        df[code_col]
        .astype(str)
        .str.replace(r'\.0$', '', regex=True)
        .str.extract(r'(\d{1,6})', expand=False)
        .fillna('')
        .str.zfill(6)
    )
    if name_col:
        out['成分券名称'] = df[name_col].astype(str)
    else:
        out['成分券名称'] = "A股目标资产"

    out = out[out['成分券代码'].map(lambda code: normalize_a_share_code(code) is not None)]
    return out.drop_duplicates('成分券代码').reset_index(drop=True)

@st.cache_data(ttl=86400, show_spinner=False)
def get_hs300_constituents():
    providers = [
        lambda: ak.index_stock_cons_csindex(symbol="000300"),
    ]
    if hasattr(ak, "index_stock_cons_sina"):
        providers.append(lambda: ak.index_stock_cons_sina(symbol="000300"))

    for provider in providers:
        df = run_with_timeout(provider, timeout_seconds=25)
        normalized = normalize_constituent_frame(df)
        if len(normalized) >= 50:
            return normalized

    return pd.DataFrame({
        '成分券代码': ['600519', '300750', '601318', '000001'],
        '成分券名称': ['贵州茅台', '宁德时代', '中国平安', '平安银行']
    })

@st.cache_data(ttl=86400, show_spinner=False)
def get_stock_name_map():
    try:
        # 【升级】直接拉取 A 股全市场股票字典，覆盖5000+只股票
        df_all = ak.stock_info_a_code_name()
        
        # akshare 全市场接口默认包含 'code' 和 'name' 列
        if 'code' in df_all.columns and 'name' in df_all.columns:
            return dict(zip(df_all['code'].astype(str), df_all['name'].astype(str)))
    except Exception as e:
        pass
        
    # 如果全市场接口意外断联，退化到原有的沪深300兜底策略
    try:
        df_300 = get_hs300_constituents()
        return dict(zip(df_300['成分券代码'], df_300['成分券名称']))
    except:
        return {
            '600519': '贵州茅台', '300750': '宁德时代',
            '601318': '中国平安', '000001': '平安银行'
        }

# --- 3. 核心双周期算法引擎 ---
def diagnostic_core(ticker, market_env, name_map):
    try:
        raw_code = "".join(filter(str.isdigit, ticker))
        stock_name = name_map.get(raw_code, "A股目标资产")

        # 1. 下载混合数据集 (包含实时未完结K线)
        df = yf.download(
            ticker,
            period="260d",
            progress=False,
            auto_adjust=True,
            timeout=15,
            threads=False
        )
        if df.empty or len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
            
        # [修复2] 盘后数据暴恐清洗：干掉空行，填充断点，修正Volume为0导致的除零溢出
        df = df.dropna(how='all') 
        df = df.ffill()           
        df['Volume'] = df['Volume'].replace(0, np.nan).ffill().fillna(1)
        
        df_history = df.iloc[:-1]       
        intraday_k = df.iloc[-1]        

        # 2. 个股基础特征工程
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Bias'] = (df['Close'] - df['MA20']) / df['MA20']
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        df['ATR_Pct'] = df['ATR'] / df['Close']
        
        change = df['Close'].diff()
        gain = (change.where(change > 0, 0)).rolling(14).mean()
        loss = (-change.where(change < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-6)))

        # ================= 新增：大盘共振与相对强弱特征注入 =================
        feats = ['Vol_Ratio', 'Bias', 'RSI', 'ATR_Pct'] # 初始化特征列表
        
        index_df = market_env.get('index_df')
        if index_df is not None and not index_df.empty:
            bm_ticker = '000300.SS' # 以沪深300作为基准水位锚点
            if bm_ticker in index_df.columns:
                # 按日期左连接大盘数据，并前向填充防止节假日错位
                df = df.join(index_df[[bm_ticker]].rename(columns={bm_ticker: 'BM_Close'}), how='left')
                df['BM_Close'] = df['BM_Close'].ffill()
                
                # 特征A: 10日相对强弱 (Alpha韧性)。正数代表跑赢大盘，负数代表跟跌不跟涨
                df['RS_10'] = df['Close'].pct_change(10) - df['BM_Close'].pct_change(10)
                
                # 特征B: 20日滚动相关性 (共振度)。判断该股是随波逐流还是走独立逻辑
                df['Corr_20'] = df['Close'].rolling(20).corr(df['BM_Close']).fillna(0)
                
                # 特征C: 大盘乖离率 (宏观水位)。让树模型学会“覆巢之下无完卵”
                df['BM_MA20'] = df['BM_Close'].rolling(20).mean()
                df['BM_Bias'] = (df['BM_Close'] - df['BM_MA20']) / (df['BM_MA20'] + 1e-6)
                
                feats.extend(['RS_10', 'Corr_20', 'BM_Bias'])
        
        # 兜底机制：若网络抖动导致大盘数据确实，以 0 填充防报错
        for f in ['RS_10', 'Corr_20', 'BM_Bias']:
            if f not in df.columns:
                df[f] = 0.0
                if f not in feats: feats.append(f)
        # ==============================================================

        # 3. 智能检测：杠杆资产及 ETF 特殊过滤单元
        # (保留你原有的过滤逻辑...)
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
        
        # 注意：这里直接使用上面动态生成的 feats 列表
        train = df.iloc[:-1][feats + ['Target']].dropna() 
        
        rf = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42)
        rf.fit(train[feats].values, train['Target'].values)
        
        latest_feats = df[feats].iloc[[-1]].values
        if np.isnan(latest_feats).any(): return None 
        
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
        
        # 6. 自适应动态期望值 (EV) 数学计算
        tp_price = round(curr_price + (atr_now * atr_multiplier_tp), 2)
        sl_price = round(curr_price - (atr_now * atr_multiplier_sl), 2)
        
        pot_gain_pct = (tp_price - curr_price) / curr_price
        pot_loss_pct = (curr_price - sl_price) / curr_price
        ev = (final_win_p * pot_gain_pct) - ((1 - final_win_p) * pot_loss_pct)
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

        # [修复3] 输出底层原生浮点数（Float），舍弃字符映射，为UI层的完美排序打好地基
        return {
            '名称': stock_name,
            '代码': ticker,
            '实时现价': round(curr_price, 2),
            '日内涨跌': intra_return,
            '基准胜率': base_win_p,
            '动态修正胜率': final_win_p,
            '数学期望值(EV)': ev,
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
    env_data = {'risk_weight': 1.0, 'status': "未知", 'details': {}, 'index_df': None} # 新增 index_df 键
    try:
        indices = {
            '沪深300 (核心资产)': '000300.SS', 
            '上证指数 (大盘权重)': '000001.SS', 
            '创业板指 (科技成长)': '399006.SZ', 
            '中证500 (中盘标杆)': '000905.SS'
        }
        
        # 【修改1】将 period="50d" 改为 "260d"，与个股特征工程的时间轴严格对齐
        m_data = yf.download(
            list(indices.values()),
            period="260d",
            progress=False,
            timeout=15,
            threads=False
        )
        close_df = m_data['Close'] if isinstance(m_data.columns, pd.MultiIndex) else m_data
        
        # 【修改2】将整张大盘数据表压入字典，供下游模型提取
        env_data['index_df'] = close_df
        
        up_counts = 0
        total_weight = 0
        # ...(保留原有的大盘胜率判断逻辑不变)
        
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
    st.markdown("### 🧬 SENTINEL V26 决策面板")
    
    # 1. 架构与模型简介
    st.markdown("""
    <div class="sidebar-box" style="border-left-color: #800000; background: #11141a; color: #e0e6ed; padding: 12px; border-radius: 6px;">
        <b style="color: #ffd700; font-size: 0.95rem;">📋 模型架构简介 (Model Architecture)</b><br>
        <p style="font-size: 0.82rem; margin-top: 6px; line-height: 1.5; color: #b4c6d8;">
        SENTINEL V26 是一款基于<b>随机森林分类器 (Random Forest)</b> 与 <b>自适应动态期望值 (Expected Value)</b> 双引擎驱动的 A 股动能博弈诊断系统。系统融合多维宏观风控墙，旨在通过数理统计优势实现仓位优选与回撤控制。
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 2. 核心量化算法
    st.markdown("""
    <div class="sidebar-box" style="border-left-color: #00875a; background: #11141a; color: #e0e6ed; padding: 12px; border-radius: 6px;">
        <b style="color: #00ffaa; font-size: 0.95rem;">🔬 核心量化算法 (Core Quant Logic)</b><br>
        <ol style="font-size: 0.8rem; margin-top: 6px; padding-left: 15px; color: #b4c6d8; line-height: 1.6;">
            <li><b>自适应期望值 (Adaptive EV)：</b>利用 14 日平均真实波幅 (ATR) 动态推演盈亏比空间，取代传统固定比例风控。</li>
            <li><b>多特征收敛 (RF Convergence)：</b>集成【量比 Volume Ratio】、【均线乖离 Bias】、【相对强弱 RSI】以及【历史波动率 ATR_Pct】作为特征向量，交叉校准未来 5 交易日的破位或突破概率。</li>
            <li><b>高频形态修正 (Intraday Correction)：</b>深度解构日内 K 线结构，基于盘中长上影线冲高回落幅度及尾盘异动对基础胜率进行动态 alpha 纠偏。</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # 3. 推荐运行时间
    st.markdown("""
    <div class="sidebar-box" style="border-left-color: #ff9900; background: #11141a; color: #e0e6ed; padding: 12px; border-radius: 6px;">
        <b style="color: #ffaa00; font-size: 0.95rem;">⏱️ 推荐运行时间 (Execution Window)</b><br>
        <p style="font-size: 0.82rem; margin-top: 6px; line-height: 1.5; color: #b4c6d8;">
        ⚠️ <b>强烈推荐运行时间：每个交易日 <span style="color:#ff4d4d; font-weight:bold;">09:35 — 15:00</span></b><br>
        <span style="color: #94a3b8;">*模型引入了日内实时高频 K 线流。09:30-09:35 刚开盘时，集合竞价导致的极端噪声和跳空极易引发模型胜率钝化。建议推迟 5 分钟运行，等待市场完成首轮流动性冷却，此时计算出的 EV 值与买入挂单点最符合统计学规律。</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4. 简明操作手册
    st.markdown("""
    <div class="sidebar-box" style="border-left-color: #0052cc; background: #11141a; color: #e0e6ed; padding: 12px; border-radius: 6px;">
        <b style="color: #3399ff; font-size: 0.95rem;">🛠️ 标准操作手册 (Operation Manual)</b><br>
        <ul style="font-size: 0.8rem; margin-top: 6px; padding-left: 15px; color: #b4c6d8; line-height: 1.5;">
            <li><b>全量扫描：</b>点击 <i>[启动 300 蓝筹扫描]</i>，系统将一键重组当前大盘成分股的高得分池，用于多头共振期的选股。</li>
            <li><b>单兵诊断：</b>在精准诊断页签输入目标资产代码（最多 5 个，例如 <code>600519 300750</code>），系统会自动执行清洗和智能补全。</li>
            <li><b>挂单执行：</b>建议严格参考系统输出的 <b>[建议买入价]</b> 采用下摆网格或分批分仓挂单，切勿盲目以市价左侧追高。</li>
            <li><b>止盈止损：</b>当价格触及 <b>[推荐止盈点]</b> 或 <b>[推荐止损点]</b> 时，应严格执行交易纪律。针对杠杆/ETF资产，系统已完成底层阈值压缩。</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🧹 清理底层缓存并重置风控墙"):
        st.cache_data.clear()
        st.rerun()

market_env = get_mainland_market_env()
name_map = get_stock_name_map()

tab1, tab2 = st.tabs(["🚀 沪深300 成分全量扫描", "🔍 A股单兵精准诊断 (上限5个)"])

DISPLAY_COLS = [
    '名称', '代码', '实时现价', '日内涨跌', 
    '基准胜率', '动态修正胜率', 
    '数学期望值(EV)', '预期周期', '建议买入价', 
    '推荐止盈点', '推荐止损点', '实时风险提示', '综合核心评分'
]

# 前端浮点数样式映射矩阵 (解决 Streamlit 字符串错乱排序的核心)
format_dict = {
    '日内涨跌': '{:+.2%}',
    '基准胜率': '{:.1%}',
    '动态修正胜率': '{:.1%}',
    '数学期望值(EV)': '{:+.2%}'
}

with tab1:
    st.write("对境内核心资产沪深300全量成分股进行动态建模，自动重组高频动能得分池。")
    if st.button("启动 300 蓝筹全量量化扫描"):
        try:
            df_300 = get_hs300_constituents()
            pool = []
            for code in df_300['成分券代码']:
                yf_code = normalize_a_share_code(code)
                if yf_code:
                    pool.append(yf_code)
            pool = list(dict.fromkeys(pool))
            if not pool:
                raise ValueError("沪深300成分代码池为空")
            if len(pool) < 50:
                st.warning("沪深300成分接口暂时不可用，已启用备用小样本池，避免页面卡死。")
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
            # 默认继续执行你要求的 EV 期望模型降序排序
            df_final = pd.DataFrame(results).sort_values('Score_Raw', ascending=False)
            st.subheader("🔥 实时动态高期望值选股池 (Top 20)")
            st.caption("💡 **排序提示**：为追求最优盈亏比，默认按 **综合核心评分（EV）** 降序。如需按胜率排序，请直接点击下表的 **【动态修正胜率】** 表头。")
            st.dataframe(
                df_final[DISPLAY_COLS].head(20).style
                .format(format_dict)
                .background_gradient(subset=['综合核心评分'], cmap='RdYlGn'),
                width='stretch'
            )

with tab2:
    st.write("##### 手动输入中国 A 股代码进行精准深度诊断")
    user_input = st.text_input(
        "请输入代码（空格分隔，最多支持5个）：", 
        "000807.SZ 002463.SZ 600183 002384.SZ 000630.SZ"
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
                st.subheader("📊 A 股多周期量化诊断报告")
                st.caption("💡 提示：点击任意表头字段即可完成智能升/降序。")
                st.dataframe(
                    df_user[DISPLAY_COLS].style
                    .format(format_dict)
                    .background_gradient(subset=['综合核心评分'], cmap='RdYlGn'), 
                    width='stretch'
                )
            else:
                st.error("诊断失败：未能成功获取对应资产数据（非交易时间或接口断联）。")
