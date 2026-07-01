import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
from sklearn.ensemble import RandomForestClassifier
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

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
        # [修复1] 补充了 51, 56, 58 等沪市 ETF 前缀，防止 510300 等被静默拦截
        if ticker.startswith(('60', '68', '90', '51', '56', '58')):
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

# --- 3. 核心双周期算法引擎 ---
def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "").strip()
            if value in ("", "-", "--", "None", "nan"):
                return default
        if pd.isna(value):
            return default
        return float(value)
    except:
        return default

def clean_numeric_series(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "-": np.nan, "--": np.nan, "None": np.nan, "nan": np.nan}),
        errors='coerce'
    )

def fetch_with_timeout(callable_fn, seconds=3, default=None):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(callable_fn)
    try:
        return future.result(timeout=seconds)
    except (FuturesTimeout, Exception):
        return default
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

@st.cache_data(ttl=300)
def get_a_share_spot_map():
    df_spot = fetch_with_timeout(lambda: ak.stock_zh_a_spot_em(), seconds=5, default=pd.DataFrame())
    if df_spot is None or df_spot.empty or '代码' not in df_spot.columns:
        return {}

    df_spot = df_spot.copy()
    df_spot['代码'] = df_spot['代码'].astype(str).str.zfill(6)
    useful_cols = ['代码', '换手率', '量比', '成交额', '流通市值', '总市值', '涨跌幅']
    useful_cols = [c for c in useful_cols if c in df_spot.columns]
    return df_spot[useful_cols].set_index('代码').to_dict('index')

@st.cache_data(ttl=1800)
def get_stock_hist_features(raw_code):
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - pd.Timedelta(days=520)).strftime("%Y%m%d")

    def loader():
        try:
            return ak.stock_zh_a_hist(
                symbol=raw_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
                timeout=3
            )
        except TypeError:
            return ak.stock_zh_a_hist(
                symbol=raw_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )

    df_hist = fetch_with_timeout(loader, seconds=4, default=pd.DataFrame())
    if df_hist is None or df_hist.empty or '日期' not in df_hist.columns:
        return pd.DataFrame()

    df_hist = df_hist.copy()
    df_hist['Trade_Date'] = pd.to_datetime(df_hist['日期'], errors='coerce').dt.normalize()
    if '换手率' in df_hist.columns:
        df_hist['Turnover'] = clean_numeric_series(df_hist['换手率']) / 100.0
    else:
        df_hist['Turnover'] = np.nan
    if '成交额' in df_hist.columns:
        df_hist['Amount_AK'] = clean_numeric_series(df_hist['成交额'])
    else:
        df_hist['Amount_AK'] = np.nan

    return df_hist[['Trade_Date', 'Turnover', 'Amount_AK']].dropna(subset=['Trade_Date'])

def infer_limit_pct(raw_code, stock_name):
    upper_name = str(stock_name).upper()
    if 'ST' in upper_name:
        return 0.05
    if raw_code.startswith(('30', '68')):
        return 0.20
    if raw_code.startswith(('8', '4')):
        return 0.30
    return 0.10

def diagnostic_core(ticker, market_env, name_map, enable_slow_features=True):
    try:
        raw_code = "".join(filter(str.isdigit, ticker))
        stock_name = name_map.get(raw_code, "A股目标资产")

        # 1. 下载混合数据集 (包含实时未完结K线)
        df = yf.download(ticker, period="260d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
            
        # [修复2] 盘后数据暴恐清洗：干掉空行，填充断点，修正Volume为0导致的除零溢出
        df = df.dropna(how='all') 
        df = df.ffill()           
        df['Volume'] = df['Volume'].replace(0, np.nan).ffill().fillna(1)
        
        df_history = df.iloc[:-1]       
        intraday_k = df.iloc[-1]        

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

        trade_dates = pd.to_datetime(df.index)
        if getattr(trade_dates, "tz", None) is not None:
            trade_dates = trade_dates.tz_convert(None)
        df['Trade_Date'] = trade_dates.normalize()

        spot_row = get_a_share_spot_map().get(raw_code, {})
        spot_turnover_raw = safe_float(spot_row.get('换手率'), np.nan)
        spot_turnover = spot_turnover_raw / 100.0 if not np.isnan(spot_turnover_raw) else np.nan
        spot_vol_ratio = safe_float(spot_row.get('量比'), np.nan)
        spot_amount = safe_float(spot_row.get('成交额'), np.nan)

        hist_features = get_stock_hist_features(raw_code) if enable_slow_features else pd.DataFrame()
        hist_available = hist_features is not None and not hist_features.empty and hist_features['Turnover'].notna().any()
        if hist_available:
            hist_idx = hist_features.drop_duplicates('Trade_Date').set_index('Trade_Date')
            df['Turnover'] = df['Trade_Date'].map(hist_idx['Turnover'])
            df['Amount_AK'] = df['Trade_Date'].map(hist_idx['Amount_AK'])
            if not np.isnan(spot_turnover):
                df.at[df.index[-1], 'Turnover'] = spot_turnover
        else:
            df['Turnover'] = 0.0
            df['Amount_AK'] = np.nan

        df['Amount_Est'] = df['Close'] * df['Volume']
        df['Amount'] = df['Amount_AK'].fillna(df['Amount_Est'])
        if not np.isnan(spot_amount):
            df.at[df.index[-1], 'Amount'] = spot_amount

        if hist_available:
            df['Turnover'] = df['Turnover'].ffill().fillna(df['Turnover'].median()).fillna(0.0)
            df['Turnover_Ratio5'] = df['Turnover'] / (df['Turnover'].rolling(5).mean() + 1e-6)
            df['Turnover_Z20'] = (df['Turnover'] - df['Turnover'].rolling(20).mean()) / (df['Turnover'].rolling(20).std() + 1e-6)
        else:
            df['Turnover_Ratio5'] = 1.0
            df['Turnover_Z20'] = 0.0

        df['Amount_Ratio'] = df['Amount'] / (df['Amount'].rolling(20).mean() + 1e-6)
        df['Liquidity_Amihud'] = df['Close'].pct_change().abs() / ((df['Amount'] / 1e8) + 1e-6)

        limit_pct = infer_limit_pct(raw_code, stock_name)
        limit_up_price = df['Close'].shift(1) * (1 + limit_pct)
        df['Limit_Distance'] = (limit_up_price - df['Close']) / (df['Close'] + 1e-6)
        df['Hit_Limit_Up'] = (df['High'] >= limit_up_price * 0.997).astype(float)
        df['Limit_Break'] = ((df['High'] >= limit_up_price * 0.997) & (df['Close'] < limit_up_price * 0.985)).astype(float)
        df['Close_Position'] = (df['Close'] - df['Low']) / ((df['High'] - df['Low']) + 1e-6)
        df['Upper_Shadow_ATR'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['ATR'] + 1e-6)

        latest_turnover_display = spot_turnover if not np.isnan(spot_turnover) else float(df['Turnover'].iloc[-1])
        latest_turnover_strength = spot_vol_ratio if not np.isnan(spot_vol_ratio) else float(df['Turnover_Ratio5'].iloc[-1])
        data_quality_note = "换手率OK" if hist_available else ("实时换手OK" if not np.isnan(spot_turnover) else "换手率降级")
        
        # 3. 智能检测：杠杆资产及 ETF 特殊过滤单元
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
        future_high = df['High'].shift(-1).rolling(5).max().shift(-4)
        valid_target = future_high.notna() & df['ATR'].notna()
        df['Target'] = np.nan
        df.loc[valid_target, 'Target'] = (
            future_high[valid_target] > df.loc[valid_target, 'Close'] + (df.loc[valid_target, 'ATR'] * 1.5)
        ).astype(int)

        feats = [
            'Vol_Ratio', 'Bias', 'RSI', 'ATR_Pct',
            'Turnover', 'Turnover_Ratio5', 'Turnover_Z20',
            'Amount_Ratio', 'Liquidity_Amihud',
            'Limit_Distance', 'Hit_Limit_Up', 'Limit_Break',
            'Close_Position', 'Upper_Shadow_ATR'
        ]
        model_frame = df[feats + ['Target']].replace([np.inf, -np.inf], np.nan)
        train = model_frame.dropna()
        
        latest_frame = df[feats].replace([np.inf, -np.inf], np.nan).iloc[[-1]]
        core_latest = latest_frame[['Vol_Ratio', 'Bias', 'RSI', 'ATR_Pct']]
        if core_latest.isna().any().any():
            return None
        latest_frame = latest_frame.fillna(0.0)

        if len(train) >= 50 and train['Target'].nunique() >= 2:
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=4, random_state=42)
            rf.fit(train[feats].values, train['Target'].values)
            base_win_p = float(rf.predict_proba(latest_frame.values)[0][1])
        else:
            base_win_p = 0.50

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

        latest_limit_break = float(df['Limit_Break'].iloc[-1])
        latest_hit_limit = float(df['Hit_Limit_Up'].iloc[-1])
        latest_close_position = float(df['Close_Position'].iloc[-1])

        a_share_multiplier = 1.0
        if latest_turnover_strength > 1.8 and latest_close_position > 0.55:
            a_share_multiplier += 0.07
        if latest_turnover_strength > 3.0 and high_fallback > 0.5:
            a_share_multiplier -= 0.12
        if latest_limit_break > 0.5:
            a_share_multiplier -= 0.18
        elif latest_hit_limit > 0.5 and latest_close_position > 0.75:
            a_share_multiplier += 0.08

        a_share_multiplier = max(0.75, min(1.25, a_share_multiplier))
        final_win_p = max(0.01, min(0.99, base_win_p * intraday_multiplier * a_share_multiplier))
        
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
            
        if latest_limit_break > 0.5:
            risk_tips = "⚠️ 涨停炸板回落 (短线分歧显著放大)"
        elif high_fallback > 0.8: 
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
            '换手率': latest_turnover_display,
            '换手强度': latest_turnover_strength,
            'A股特征修正': a_share_multiplier,
            '预期周期': cycle_desc,
            '建议买入价': round(curr_price * 0.992, 2),
            '推荐止盈点': tp_price,
            '推荐止损点': sl_price,
            '实时风险提示': risk_tips,
            '数据质量': data_quality_note,
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
    '数学期望值(EV)', '换手率', '换手强度', 'A股特征修正', '预期周期', '建议买入价', 
    '推荐止盈点', '推荐止损点', '实时风险提示', '数据质量', '综合核心评分'
]

# 前端浮点数样式映射矩阵 (解决 Streamlit 字符串错乱排序的核心)
format_dict = {
    '日内涨跌': '{:+.2%}',
    '基准胜率': '{:.1%}',
    '动态修正胜率': '{:.1%}',
    '数学期望值(EV)': '{:+.2%}',
    '换手率': '{:.2%}',
    '换手强度': '{:.2f}x',
    'A股特征修正': '{:.2f}x'
}

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
            res = diagnostic_core(t, market_env, name_map, enable_slow_features=False)
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
                    res = diagnostic_core(t, market_env, name_map, enable_slow_features=True)
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
