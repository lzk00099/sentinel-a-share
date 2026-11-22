"""Chan-style charting and signal engine adapted for A-share daily bars.

Structure rules are transparent approximations and are intended for research.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from chan_addons import cd_rsi_signals, sea_bottom_moon

Kind = Literal["H", "L"]

AUTO_PRESETS = {
    "15m": {"radius": 5, "min_bars": 8, "scale": 2, "large_at": 9, "force": .65, "levels": 6},
    "30m": {"radius": 4, "min_bars": 7, "scale": 2, "large_at": 9, "force": .68, "levels": 7},
    "1h": {"radius": 3, "min_bars": 6, "scale": 2, "large_at": 9, "force": .72, "levels": 8},
    "1d": {"radius": 3, "min_bars": 4, "scale": 3, "large_at": 9, "force": .75, "levels": 8},
    "1wk": {"radius": 2, "min_bars": 3, "scale": 2, "large_at": 9, "force": .80, "levels": 6},
}

@dataclass(frozen=True)
class Pivot:
    i: int
    t: pd.Timestamp
    price: float
    kind: Kind
    confirmed_i: int

@dataclass(frozen=True)
class Center:
    start_i: int
    end_i: int
    low: float
    high: float
    strokes: int
    direction: Literal['up', 'down']
    tier: Literal['major', 'minor']
    start_pivot: int
    end_pivot: int

@dataclass(frozen=True)
class Signal:
    i: int
    t: pd.Timestamp
    price: float
    text: str
    side: Literal['buy', 'sell']
    confidence: str

@dataclass(frozen=True)
class Segment:
    """A confirmed, simplified directional segment made from consecutive pens."""
    start_i: int
    end_i: int
    start_t: pd.Timestamp
    end_t: pd.Timestamp
    start_price: float
    end_price: float
    direction: Literal['up', 'down']
    strokes: int

def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    fast = result.Close.ewm(span=12, adjust=False).mean()
    slow = result.Close.ewm(span=26, adjust=False).mean()
    result['macd_hist'] = fast - slow - (fast - slow).ewm(span=9, adjust=False).mean()
    tr = pd.concat([result.High - result.Low, (result.High - result.Close.shift()).abs(), (result.Low - result.Close.shift()).abs()], axis=1).max(axis=1)
    result['atr'] = tr.rolling(14, min_periods=1).mean()
    return result

def recent_primary_buys(data: pd.DataFrame, radius: int, min_bars: int, scale: int, force_ratio: float, strict_three: bool, lookback_bars: int) -> list[str]:
    """Return confirmed Chan-style buy labels in the most recent scan window."""
    d = add_indicators(data)
    minor = build_pivots(d, radius, min_bars)
    major = build_pivots(d, radius * scale, min_bars * scale)
    minor_centers, major_centers = (build_centers(minor, 'minor'), build_centers(major, 'major'))
    signals = first_second_signals(d, minor, force_ratio) + third_signals(minor, minor_centers, strict_three) + third_signals(major, major_centers, strict_three)
    return sorted({x.text for x in signals if x.side == 'buy' and x.i >= len(d) - lookback_bars})

def remove_inclusion(data: pd.DataFrame) -> pd.DataFrame:
    """Merge containing K-bars before identifying fractals.

    In an upward relation the merged bar keeps the higher high and higher low;
    in a downward relation it keeps the lower high and lower low. Each merged
    record retains the original end-bar index so plots and confirmations still
    align with the raw price chart.
    """
    bars: list[dict[str, float | int | pd.Timestamp]] = []
    for raw_i, (stamp, row) in enumerate(data.iterrows()):
        current = {'start_i': raw_i, 'end_i': raw_i, 't': stamp, 'open': float(row.Open), 'high': float(row.High), 'low': float(row.Low), 'close': float(row.Close)}
        if not bars:
            bars.append(current)
            continue
        last = bars[-1]
        contained = current['high'] <= last['high'] and current['low'] >= last['low'] or (current['high'] >= last['high'] and current['low'] <= last['low'])
        if not contained:
            bars.append(current)
            continue
        if len(bars) >= 2:
            prior = bars[-2]
            rising = last['high'] >= prior['high'] and last['low'] >= prior['low']
            falling = last['high'] <= prior['high'] and last['low'] <= prior['low']
            up = rising if rising != falling else last['close'] >= prior['close']
        else:
            up = last['close'] >= last['open']
        merged = dict(last)
        merged['end_i'], merged['t'], merged['close'] = (current['end_i'], current['t'], current['close'])
        if up:
            merged['high'], merged['low'] = (max(last['high'], current['high']), max(last['low'], current['low']))
        else:
            merged['high'], merged['low'] = (min(last['high'], current['high']), min(last['low'], current['low']))
        bars[-1] = merged
    return pd.DataFrame(bars)

def raw_fractals(data: pd.DataFrame, radius: int) -> list[Pivot]:
    """Strict top/bottom fractals calculated after containment processing."""
    bars = remove_inclusion(data)
    out: list[Pivot] = []
    for i in range(radius, len(bars) - radius):
        before, current, after = (bars.iloc[i - radius:i], bars.iloc[i], bars.iloc[i + 1:i + radius + 1])
        is_top = current.high > before.high.max() and current.high > after.high.max() and (current.low > before.low.max()) and (current.low > after.low.max())
        is_bottom = current.low < before.low.min() and current.low < after.low.min() and (current.high < before.high.min()) and (current.high < after.high.min())
        raw_i = int(current.end_i)
        confirmed_i = int(bars.iloc[i + radius].end_i)
        if is_top:
            out.append(Pivot(raw_i, data.index[raw_i], float(current.high), 'H', confirmed_i))
        if is_bottom:
            out.append(Pivot(raw_i, data.index[raw_i], float(current.low), 'L', confirmed_i))
    return sorted(out, key=lambda x: (x.i, x.kind))

def build_pivots(data: pd.DataFrame, radius: int, min_bars: int) -> list[Pivot]:
    """Simplified pen construction: alternate tops/bottoms; retain the more
    extreme candidate of the same type; require min_bars between opposite ends.
    It is deliberately deterministic rather than claiming any one school is canonical.
    """
    kept: list[Pivot] = []
    for p in raw_fractals(data, radius):
        if not kept:
            kept.append(p)
            continue
        prev = kept[-1]
        if p.kind == prev.kind:
            better = p.price > prev.price if p.kind == 'H' else p.price < prev.price
            if better:
                kept[-1] = p
        elif p.i - prev.i >= min_bars:
            kept.append(p)
    return kept

def build_segments(pivots: list[Pivot]) -> list[Segment]:
    """Build conservative directional segments from a sequence of pens.

    A rising segment needs successive higher low / higher high structure; a
    falling segment needs lower high / lower low. It is intentionally labelled
    'simplified segment' because it does not yet implement every feature-sequence
    exception used by strict Chan line-segment schools.
    """
    out: list[Segment] = []
    start = 0
    while start + 3 < len(pivots):
        a, b, c, d = pivots[start:start + 4]
        up = a.kind == 'L' and b.kind == 'H' and (c.kind == 'L') and (d.kind == 'H') and (c.price > a.price) and (d.price > b.price)
        down = a.kind == 'H' and b.kind == 'L' and (c.kind == 'H') and (d.kind == 'L') and (c.price < a.price) and (d.price < b.price)
        if not (up or down):
            start += 1
            continue
        direction: Literal['up', 'down'] = 'up' if up else 'down'
        end = start + 3
        while end + 2 < len(pivots):
            retrace, extension = (pivots[end + 1], pivots[end + 2])
            if direction == 'up':
                continues = retrace.kind == 'L' and extension.kind == 'H' and (retrace.price > pivots[end - 1].price) and (extension.price > pivots[end].price)
            else:
                continues = retrace.kind == 'H' and extension.kind == 'L' and (retrace.price < pivots[end - 1].price) and (extension.price < pivots[end].price)
            if not continues:
                break
            end += 2
        first, last = (pivots[start], pivots[end])
        out.append(Segment(first.i, last.i, first.t, last.t, first.price, last.price, direction, end - start))
        start = end
    return out

def stroke_range(a: Pivot, b: Pivot) -> tuple[float, float]:
    return (min(a.price, b.price), max(a.price, b.price))

def build_centers(pivots: list[Pivot], tier: Literal['major', 'minor']) -> list[Center]:
    """Build non-overlapping three-stroke overlaps and extend them while each
    following stroke overlaps the original [ZD, ZG]. The persisted stroke count
    is what makes 7/9-stroke zones visible.
    """
    centers: list[Center] = []
    stroke = 0
    while stroke + 3 < len(pivots):
        ranges = [stroke_range(pivots[k], pivots[k + 1]) for k in range(stroke, stroke + 3)]
        zd, zg = (max((x[0] for x in ranges)), min((x[1] for x in ranges)))
        if zd >= zg:
            stroke += 1
            continue
        end_stroke = stroke + 2
        while end_stroke + 2 < len(pivots):
            nzd, nzg = stroke_range(pivots[end_stroke + 1], pivots[end_stroke + 2])
            if nzg <= zd or nzd >= zg:
                break
            end_stroke += 1
        end_pivot = end_stroke + 1
        direction = 'up' if pivots[end_pivot].price >= pivots[stroke].price else 'down'
        centers.append(Center(pivots[stroke].i, pivots[end_pivot].i, zd, zg, end_stroke - stroke + 1, direction, tier, stroke, end_pivot))
        stroke = end_stroke + 1
    return centers

def segment_force(data: pd.DataFrame, start: int, end: int) -> float:
    """A reproducible proxy for '力度': absolute MACD histogram area."""
    lo, hi = sorted((start, end))
    return float(data.macd_hist.iloc[lo:hi + 1].abs().sum())

def first_second_signals(data: pd.DataFrame, pivots: list[Pivot], force_ratio: float) -> list[Signal]:
    """Candidate 1/2 points. A 1 point needs a new price extreme plus weaker
    same-direction MACD force. A 2 point needs a later non-breaking retest.
    All are labelled 'candidate' because lower-level confirmation is not inferred.
    """
    signals: list[Signal] = []
    last_one_buy: Pivot | None = None
    last_one_sell: Pivot | None = None
    prior_low: Pivot | None = None
    prior_high: Pivot | None = None
    prior_low_force: float | None = None
    prior_high_force: float | None = None
    for n, p in enumerate(pivots):
        if n < 1:
            continue
        leg_force = segment_force(data, pivots[n - 1].i, p.i)
        if p.kind == 'L':
            if prior_low is not None and p.price < prior_low.price and (prior_low_force is not None) and (leg_force < prior_low_force * force_ratio):
                signals.append(Signal(p.i, p.t, p.price, '1买', 'buy', '背驰候选'))
                last_one_buy = p
            elif last_one_buy is not None and p.i > last_one_buy.i and (p.price > last_one_buy.price):
                signals.append(Signal(p.i, p.t, p.price, '2买', 'buy', '回踩候选'))
                last_one_buy = None
            prior_low, prior_low_force = (p, leg_force)
        else:
            if prior_high is not None and p.price > prior_high.price and (prior_high_force is not None) and (leg_force < prior_high_force * force_ratio):
                signals.append(Signal(p.i, p.t, p.price, '1卖', 'sell', '背驰候选'))
                last_one_sell = p
            elif last_one_sell is not None and p.i > last_one_sell.i and (p.price < last_one_sell.price):
                signals.append(Signal(p.i, p.t, p.price, '2卖', 'sell', '回抽候选'))
                last_one_sell = None
            prior_high, prior_high_force = (p, leg_force)
    return signals

def third_signals(pivots: list[Pivot], centers: list[Center], strict: bool) -> list[Signal]:
    """For each center: detect an exit, then only the first opposite pivot.
    3buy: first pullback low stays at/above ZG after upward departure.
    3sell: first rebound high stays at/below ZD after downward departure.
    """
    signals: list[Signal] = []
    for c in centers:
        after = pivots[c.end_pivot + 1:]
        state: str | None = None
        for p in after:
            if state is None:
                if p.kind == 'H' and p.price > c.high:
                    state = 'up'
                elif p.kind == 'L' and p.price < c.low:
                    state = 'down'
            elif state == 'up' and p.kind == 'L':
                valid = p.price > c.high if strict else p.price >= c.high
                if valid:
                    signals.append(Signal(p.i, p.t, p.price, '3买', 'buy', '中枢外回踩'))
                break
            elif state == 'down' and p.kind == 'H':
                valid = p.price < c.low if strict else p.price <= c.low
                if valid:
                    signals.append(Signal(p.i, p.t, p.price, '3卖', 'sell', '中枢外回抽'))
                break
    return list({(x.i, x.text): x for x in signals}.values())

def zone_fill(c: Center, large_at: int) -> tuple[str, str]:
    """Four palette based on trend and actual center extension length.
    A zone becomes 'large' at large_at strokes (9 by default), irrespective of
    which fractal scale discovered it. This keeps the colour meaning identical
    in Streamlit and Colab.
    """
    is_large = c.strokes >= large_at
    if is_large and c.direction == 'up':
        return ('rgba(174, 255, 0, .28)', '#AFFF00')
    if not is_large and c.direction == 'up':
        return ('rgba(0, 210, 128, .24)', '#00D280')
    if not is_large and c.direction == 'down':
        return ('rgba(174, 45, 182, .24)', '#AE2DB6')
    return ('rgba(231, 76, 60, .25)', '#E74C3C')

def unique_levels(data: pd.DataFrame, centers: list[Center], pivots: list[Pivot], limit: int) -> list[tuple[float, str]]:
    """Current support/resistance candidates from recent center edges and pivots."""
    current, atr = (float(data.Close.iloc[-1]), float(data.atr.iloc[-1]))
    candidates: list[float] = []
    for c in centers[-8:]:
        candidates.extend([c.low, c.high])
    candidates.extend((p.price for p in pivots[-12:]))
    distinct: list[float] = []
    for x in sorted(candidates, key=lambda v: abs(v - current)):
        if not any((abs(x - y) <= max(atr * 0.16, current * 0.001) for y in distinct)):
            distinct.append(x)
    result = [(x, '阻力' if x > current else '支撑') for x in distinct[:limit]]
    return sorted(result, key=lambda z: z[0])

def plot_chart(data: pd.DataFrame, minor: list[Pivot], major: list[Pivot], segments: list[Segment], centers: list[Center], signals: list[Signal], levels: list[tuple[float, str]], sea: pd.DataFrame, cd: pd.DataFrame, show_minor: bool, show_segments: bool, large_at: int) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.012, row_heights=[0.64, 0.18, 0.18], subplot_titles=('', '', ''))
    fig.add_trace(go.Candlestick(x=data.index, open=data.Open, high=data.High, low=data.Low, close=data.Close, name='价格', increasing_line_color='#33d69f', increasing_fillcolor='#33d69f', decreasing_line_color='#ff5364', decreasing_fillcolor='#ff5364'), row=1, col=1)
    for c in sorted(centers, key=lambda x: 0 if x.tier == 'major' else 1):
        fill, edge = zone_fill(c, large_at)
        fig.add_shape(type='rect', x0=data.index[c.start_i], x1=data.index[c.end_i], y0=c.low, y1=c.high, fillcolor=fill, line={'color': edge, 'width': 1.3}, layer='below', row=1, col=1)
        label = f"{('大' if c.strokes >= large_at else '小')}中枢 {c.strokes}笔"
        if c.strokes >= 7:
            label += ' ★'
        if c.strokes >= 9:
            label += '★'
        fig.add_annotation(x=data.index[c.start_i], y=c.high, text=label, showarrow=False, font={'color': edge, 'size': 10}, xanchor='left', yanchor='bottom', row=1, col=1)
    for pivs, width, name in ((major, 2.3, '大级别结构'), (minor, 1.25, '小级别结构')):
        if pivs is minor and (not show_minor):
            continue
        for a, b in zip(pivs, pivs[1:]):
            fig.add_shape(type='line', x0=a.t, y0=a.price, x1=b.t, y1=b.price, line={'color': '#00d68f' if b.price > a.price else '#ff4d5a', 'width': width}, row=1, col=1)
    if show_segments:
        for s in segments:
            fig.add_shape(type='line', x0=s.start_t, y0=s.start_price, x1=s.end_t, y1=s.end_price, line={'color': '#F1C40F', 'width': 3, 'dash': 'dash'}, row=1, col=1)
            fig.add_annotation(x=s.end_t, y=s.end_price, text=f'简化线段 {s.strokes}笔', showarrow=False, font={'color': '#F1C40F', 'size': 9}, yshift=12, row=1, col=1)
    for side, color, symbol in (('buy', '#00C98D', 'triangle-up'), ('sell', '#FF5A5F', 'triangle-down')):
        s = [x for x in signals if x.side == side]
        if s:
            fig.add_trace(go.Scatter(x=[x.t for x in s], y=[x.price for x in s], mode='markers+text', text=[x.text for x in s], textposition='bottom center' if side == 'buy' else 'top center', marker={'size': 17, 'color': color, 'symbol': symbol, 'line': {'color': 'white', 'width': 1.5}}, customdata=[[x.confidence] for x in s], hovertemplate='%{text}<br>%{customdata[0]}<extra></extra>', name='买卖点候选'), row=1, col=1)
    for value, name in levels:
        color = '#00a9e8' if name == '阻力' else '#f5a623'
        fig.add_hline(y=value, line={'color': color, 'dash': 'dot', 'width': 1}, annotation_text=f'{name} {value:.2f}', annotation_position='right', row=1, col=1)
    fig.add_trace(go.Bar(x=sea.index, y=sea.pressure, name='底部压力', marker_color='#ff5ca8', opacity=0.65), row=2, col=1)
    fig.add_trace(go.Scatter(x=sea.index, y=sea.tide, name='Tide', line={'color': '#ff0070', 'width': 2}), row=2, col=1)
    fig.add_trace(go.Scatter(x=sea.index, y=sea.signal, name='Signal', line={'color': '#e7e7e7', 'width': 1.6}), row=2, col=1)
    tier_style = {'小底': ('#ff94be', 22), '中底': ('#ff5ca8', 25), '大底': ('#ff1493', 29), '超大底': ('#ffe600', 33)}
    for tier, (color, size) in tier_style.items():
        point = sea.tier == tier
        if point.any():
            fig.add_trace(go.Scatter(x=sea.index[point], y=sea.tide[point], mode='markers+text', text=[tier] * int(point.sum()), textposition='top center', name=tier, marker={'color': color, 'size': size, 'symbol': 'triangle-up', 'line': {'color': 'white', 'width': 1}}, textfont={'color': color}), row=2, col=1)
    fig.add_hline(y=20, line={'color': '#ff94be', 'dash': 'dot'}, row=2, col=1)
    fig.add_hline(y=12, line={'color': '#ff1493', 'dash': 'dot'}, row=2, col=1)
    fig.update_yaxes(range=[0, 100], title_text='底部振荡', row=2, col=1)
    hist = cd['hist']
    hist_colors = np.where((hist >= 0) & (hist >= hist.shift(1)), '#00d658', np.where(hist >= 0, '#007f42', np.where(hist < hist.shift(1), '#ff405a', '#ff9d00')))
    fig.add_trace(go.Bar(x=cd.index, y=hist, name='MACD 柱', marker_color=hist_colors), row=3, col=1)
    diff_line = cd['diff']
    dea_line = cd['dea']
    fig.add_trace(go.Scatter(x=cd.index, y=diff_line, name='DIFF', line={'color': '#00dcff', 'width': 1.8}), row=3, col=1)
    fig.add_trace(go.Scatter(x=cd.index, y=dea_line, name='DEA', line={'color': '#ffd200', 'width': 1.8}), row=3, col=1)
    for column, label, color, marker, position in (('bottom', 'BUY', '#00ff80', 'triangle-up', 'bottom center'), ('top', 'SELL', '#ff405a', 'triangle-down', 'top center')):
        point = cd[column]
        if point.any():
            multiplier = 0.81 if column == 'bottom' else 1.21
            fig.add_trace(go.Scatter(x=cd.index[point], y=diff_line[point] * multiplier, mode='markers+text', text=[label] * int(point.sum()), textposition=position, name=f'CD+RSI {label}', marker={'color': color, 'size': 15, 'symbol': marker, 'line': {'color': 'white', 'width': 1}}), row=3, col=1)
    fig.add_hline(y=0, line={'color': '#8c8c8c', 'width': 1}, row=3, col=1)
    fig.update_yaxes(title_text='CD+RSI', row=3, col=1)
    # Keep one true date axis across the candlesticks, shapes, annotations and
    # indicator traces. A category axis converts datetimes into per-trace
    # category positions; shapes can then land on a different category index
    # and appear detached from the candles. Hide non-trading weekends while
    # retaining date coordinates so every overlay shares the same x mapping.
    fig.update_xaxes(showgrid=False, showline=False, zeroline=False, type='date',
                     rangebreaks=[{'bounds': ['sat', 'mon']}], nticks=10)
    fig.update_yaxes(gridcolor='#243244', gridwidth=1, zerolinecolor='#62738a', tickfont={'color': '#b9c5d8'}, title_font={'color': '#b9c5d8'})
    fig.update_layout(template='plotly_dark', height=940, xaxis_rangeslider_visible=False, paper_bgcolor='#0e131c', plot_bgcolor='#0e131c', font={'color': '#d9e2ee'}, margin={'l': 12, 'r': 135, 't': 12, 'b': 12}, showlegend=False)
    return fig
