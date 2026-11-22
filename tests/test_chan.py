import pandas as pd

from chan_addons import cd_rsi_signals, sea_bottom_moon
from chan_engine import (AUTO_PRESETS, add_indicators, build_centers, build_pivots,
                         build_segments, first_second_signals, plot_chart,
                         third_signals, unique_levels)
from test_data import bars


def test_chan_indicators_keep_daily_index_and_expected_columns():
    data = add_indicators(bars(180))
    sea = sea_bottom_moon(data, "1d")
    cd = cd_rsi_signals(data)

    assert sea.index.equals(data.index)
    assert {"tide", "signal", "rsi", "stoch", "tier"}.issubset(sea.columns)
    assert cd.index.equals(data.index)
    assert {"diff", "dea", "hist", "rsi", "bull_div", "bottom", "top"}.issubset(cd.columns)


def test_chan_structure_and_plot_build_from_a_share_daily_bars():
    data = add_indicators(bars(240))
    preset = AUTO_PRESETS["1d"]
    minor = build_pivots(data, preset["radius"], preset["min_bars"])
    major = build_pivots(data, preset["radius"] * preset["scale"],
                         preset["min_bars"] * preset["scale"])
    minor_centers = build_centers(minor, "minor")
    major_centers = build_centers(major, "major")
    centers = minor_centers + major_centers
    segments = build_segments(minor)
    signals = (first_second_signals(data, minor, preset["force"])
               + third_signals(minor, minor_centers, True)
               + third_signals(major, major_centers, True))
    sea = sea_bottom_moon(data, "1d")
    cd = cd_rsi_signals(data)
    levels = unique_levels(data, centers, minor + major, preset["levels"])
    figure = plot_chart(data, minor, major, segments, centers, signals, levels,
                        sea, cd, True, True, preset["large_at"])

    assert len(minor) > 0
    assert len(figure.data) >= 6
    assert len(figure.layout.xaxis.matches) > 0
