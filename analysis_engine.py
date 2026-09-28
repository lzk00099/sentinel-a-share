"""Pure daily-bar analysis, independent of network and Streamlit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from market_data import History, now_cn


class AnalysisError(ValueError):
    """Valid data that cannot produce a usable analysis result."""


def completed_bars(frame, now=None):
    now = now or now_cn()
    # Conservative cutoff: after 15:15 Shanghai time allow today's daily bar.
    if now.hour * 60 + now.minute < 15 * 60 + 15:
        return frame[frame.index.date < now.date()].copy()
    return frame.copy()


def build_features(frame, benchmark=None):
    data = frame.copy()
    previous = data["Close"].shift(1)
    true_range = pd.concat([data["High"] - data["Low"],
                            (data["High"] - previous).abs(),
                            (data["Low"] - previous).abs()], axis=1).max(axis=1)
    data["ATR"] = true_range.rolling(14).mean()
    data["ATR_Pct"] = data["ATR"] / data["Close"]
    data["Vol_Ratio"] = data["Volume"] / data["Volume"].rolling(5).mean().replace(0, np.nan)
    data["Bias"] = data["Close"] / data["Close"].rolling(20).mean() - 1
    change = data["Close"].diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = (-change.clip(upper=0)).rolling(14).mean()
    data["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    data.loc[(loss == 0) & (gain > 0), "RSI"] = 100
    data.loc[(loss == 0) & (gain == 0), "RSI"] = 50
    features = ["Vol_Ratio", "Bias", "RSI", "ATR_Pct"]
    if benchmark is not None and not benchmark.empty:
        aligned = benchmark["Close"].reindex(data.index)
        relative = data["Close"].pct_change(10, fill_method=None) - aligned.pct_change(10, fill_method=None)
        correlation = data["Close"].rolling(20).corr(aligned)
        bias = aligned / aligned.rolling(20).mean() - 1
        macro = pd.DataFrame({"RS_10": relative, "Corr_20": correlation, "BM_Bias": bias})
        macro = macro.replace([np.inf, -np.inf], np.nan)
        # Use macro features only if dates align and the latest complete bar is usable.
        if len(macro.dropna()) >= 65 and macro.iloc[-1].notna().all():
            data = data.join(macro)
            features += list(macro.columns)
    return data.replace([np.inf, -np.inf], np.nan), features


def future_target(data, horizon=5):
    future = pd.concat([data["High"].shift(-offset) for offset in range(1, horizon + 1)], axis=1)
    target = (future.max(axis=1) > data["Close"] + 1.5 * data["ATR"]).astype(float)
    # Unknown future outcomes must stay NaN rather than becoming negative examples.
    target[future.isna().any(axis=1) | data["ATR"].isna()] = np.nan
    return target


def positive_probability(model, sample):
    probabilities = model.predict_proba(sample)[0]
    positions = np.flatnonzero(model.classes_ == 1)
    return float(probabilities[positions[0]]) if len(positions) else 0.0


def analyze(history: History, ticker, benchmark=None, risk_weight=1.0):
    bars = completed_bars(history.frame)
    if len(bars) < 85:
        raise AnalysisError(f"仅有 {len(bars)} 根完整日线，建模至少需要 85 根")
    data, features = build_features(bars, completed_bars(benchmark) if benchmark is not None else None)
    data["Target"] = future_target(data)
    train = data[features + ["Target"]].dropna()
    if len(train) < 60:
        raise AnalysisError(f"有效训练样本不足：{len(train)} / 60")
    latest = data[features].tail(1)
    if not np.isfinite(latest.to_numpy()).all():
        raise AnalysisError("最新完整日线的指标无效（可能停牌或成交量不足）")
    age = (now_cn().date() - bars.index[-1].date()).days
    latest_bar = bars.iloc[-1]
    if age > 7:
        raise AnalysisError(f"数据停留在 {bars.index[-1].date()}，已停用模型评分")
    if latest_bar["Volume"] <= 0:
        raise AnalysisError("最近完整日线成交量为零，暂停评分")
    atr = float(data["ATR"].iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        raise AnalysisError("最近波动率为零或无效，无法计算区间")
    model = RandomForestClassifier(n_estimators=60, max_depth=4, min_samples_leaf=5,
                                   random_state=42, n_jobs=1)
    model.fit(train[features], train["Target"].astype(int))
    probability = positive_probability(model, latest)
    close = float(latest_bar["Close"])
    last_return = close / float(bars["Close"].iloc[-2]) - 1
    position = ((close - latest_bar["Low"]) / (latest_bar["High"] - latest_bar["Low"])
                if latest_bar["High"] > latest_bar["Low"] else 0.5)
    fallback = (latest_bar["High"] - close) / atr
    multiplier = np.clip(1 - max(0, fallback - 0.4) * 0.4 - max(0, 0.3 - position) * 0.3, 0.5, 1.4)
    adjusted = float(np.clip(probability * multiplier, 0, 1))
    is_fund = ticker.startswith(("50", "51", "52", "56", "58", "15", "16", "18"))
    upper = close + atr * (1.8 if is_fund else 2.5)
    lower = max(close * 0.01, close - atr * (1.2 if is_fund else 1.5))
    # Preserved ranking heuristic, not an estimated strategy payoff / calibrated win rate.
    heuristic = adjusted * (upper / close - 1) - (1 - adjusted) * (1 - lower / close)
    score = adjusted * heuristic * risk_weight * 1000
    notes = list(history.notes)
    if len(features) == 4:
        notes.append("基准不可用或日期未对齐，使用个股特征")
    if train["Target"].nunique() < 2:
        notes.append("训练样本只有一类标签，模型参考性有限")
    if fallback > 0.8:
        notes.append("最近完整日线存在较长上影线")
    if history.frame.index[-1] != bars.index[-1]:
        notes.append("当日日线尚未收盘，评分使用上一根完整日线")
    latest_price = float(history.frame["Close"].iloc[-1])
    return {"名称": history.name, "代码": ticker, "最近价": round(latest_price, 3),
            "行情日期": str(history.frame.index[-1].date()), "评分日期": str(bars.index[-1].date()),
            "评分基准价": round(close, 3), "最近完整日涨跌(%)": last_return * 100,
            "模型突破概率(%)": probability * 100, "形态修正值(%)": adjusted * 100,
            "启发式盈亏分值(%)": heuristic * 100, "综合评分": round(score, 3),
            "上方参考位": round(upper, 3), "下方参考位": round(lower, 3),
            "训练样本": len(train), "数据源": history.source, "复权": history.adjustment,
            "提示": "；".join(notes) or "使用完整日线，未来突破观察窗口为 5 个交易日"}
