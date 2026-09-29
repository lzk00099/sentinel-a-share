from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from analysis_engine import (AnalysisError, analyze, build_features, completed_bars,
                             future_target, positive_probability)
from market_data import CN, History, now_cn
from test_data import bars


def history(frame=None):
    return History(bars(180) if frame is None else frame, "测试", "测试资产", now_cn().isoformat(), "前复权")


def test_unknown_last_five_targets_are_excluded():
    data, _ = build_features(bars())
    target = future_target(data)
    assert target.tail(5).isna().all()
    for index in range(20, len(data) - 5):
        expected = data["High"].iloc[index + 1:index + 6].max() > data["Close"].iloc[index] + 1.5 * data["ATR"].iloc[index]
        assert target.iloc[index] == int(expected)


@pytest.mark.parametrize("label", [0, 1])
def test_one_class_model_does_not_index_missing_class(label):
    x = np.arange(80).reshape(-1, 1)
    model = RandomForestClassifier(n_estimators=3, random_state=1).fit(x, np.full(80, label))
    assert positive_probability(model, x[-1:]) == float(label)


def test_atr_accounts_for_overnight_gap():
    data = bars(40)
    data.loc[:, ["Open", "Close"]] = 100.0
    data["High"], data["Low"] = 101.0, 99.0
    data.iloc[-1, :4] = [120, 121, 119, 120]
    featured, _ = build_features(data)
    assert featured["ATR"].iloc[-1] == pytest.approx((13 * 2 + 21) / 14)


def test_intraday_is_excluded_and_finished_day_retained():
    data = bars(20)
    today = pd.Timestamp("2026-09-21")
    data.index = pd.bdate_range(end=today, periods=20)
    assert len(completed_bars(data, datetime(2026, 9, 21, 10, tzinfo=CN))) == 19
    assert len(completed_bars(data, datetime(2026, 9, 21, 16, tzinfo=CN))) == 20


def test_missing_benchmark_falls_back_without_fabricating_zero_features():
    _, features = build_features(bars(), pd.DataFrame())
    assert len(features) == 4
    output = analyze(history(), "600519.SS")
    assert "个股特征" in output["提示"]
    assert np.isfinite(output["综合评分"])


def test_aligned_benchmark_used_and_no_leak_from_future():
    original = bars()
    changed = original.copy()
    changed.iloc[-1, changed.columns.get_loc("Close")] += 10
    before, features = build_features(original, original)
    after, _ = build_features(changed, original)
    assert len(features) == 7
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])


def test_suspended_and_stale_data_do_not_produce_score():
    data = bars()
    data.iloc[-1, data.columns.get_loc("Volume")] = 0
    with pytest.raises(AnalysisError, match="成交量为零"):
        analyze(history(data), "600519.SS")
    data = bars()
    data.index -= pd.Timedelta(days=40)
    with pytest.raises(AnalysisError, match="停用"):
        analyze(history(data), "600519.SS")


def test_end_to_end_analysis_has_numeric_columns():
    report = analyze(history(), "600519.SS", bars(180))
    assert report["训练样本"] >= 60
    for key in ("最近价", "模型突破概率(%)", "综合评分", "启发式盈亏分值(%)"):
        assert isinstance(report[key], (float, int))
        assert np.isfinite(report[key])
    assert report["上方参考位"] > report["评分基准价"] > report["下方参考位"] > 0
