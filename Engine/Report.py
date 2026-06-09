import numpy as np
import pandas as pd
import plotly.graph_objects as go
import webbrowser
from pathlib import Path

from Engine.Utils import confidence_interval, confidence_interval_mean


def generate_report(bt, output_path: str = 'backtest_report.html', open_browser: bool = True):
    assert bt.backtest_ran, "Run full_backtest() before generating report"

    # --- Compute stats ---
    trade_groups = bt.trade_indicies.cumsum()
    trade_returns = pd.Series(bt.candle_pnl).groupby(trade_groups).sum().values

    # Win rate CI (Wilson score)
    _, wr_lb_95, wr_ub_95 = confidence_interval(
        bt.winning_trades, bt.total_trades, 0.05)
    _, wr_lb_997, wr_ub_997 = confidence_interval(
        bt.winning_trades, bt.total_trades, 0.003)

    # EV confidence intervals (t-distribution)
    ev_mean = trade_returns.mean()
    ev_std = trade_returns.std()
    n_trades = len(trade_returns)
    _, ev_lb_95, ev_ub_95 = confidence_interval_mean(
        ev_mean, ev_std, n_trades, alpha=0.05)
    _, ev_lb_997, ev_ub_997 = confidence_interval_mean(
        ev_mean, ev_std, n_trades, alpha=0.003)

    # Buy & hold equity curve
    shares = bt.initial_equity / bt.close_prices[0]
    buyhold_equity = shares * bt.close_prices

    # Build per-trade equity/drawdown by summing at trade boundaries
    boundaries = np.flatnonzero(bt.trade_indicies)
    trade_returns_cum = np.add.reduceat(bt.candle_pnl, boundaries).cumsum()
    trade_equity = bt.initial_equity + trade_returns_cum
    trade_drawdown = trade_equity - np.maximum.accumulate(trade_equity)
    trade_x = np.arange(1, len(trade_equity) + 1)

    # Buy & hold sampled at trade boundaries
    trade_buyhold = buyhold_equity[boundaries]

    # Trade dates (timestamp at each trade boundary)
    trade_dates = bt.indicies[boundaries]

    # Downsample for plotting (by trade number)
    step = max(1, len(trade_equity) // 2000)
    plot_x = trade_x[::step]
    plot_equity = trade_equity[::step]
    plot_buyhold = trade_buyhold[::step]
    plot_drawdown = trade_drawdown[::step]
    plot_dates = trade_dates[::step]

    # Date range string
    start_str = bt.start_date.strftime('%Y-%m-%d')
    end_str = bt.end_date.strftime('%Y-%m-%d')
    date_range_str = f'{start_str}  —  {end_str}'

    # --- Build Plotly figures ---

    # 1. Main equity chart
    fig_equity = go.Figure()
    plot_date_strs = [d.strftime('%Y-%m-%d %H:%M') for d in plot_dates]
    fig_equity.add_trace(go.Scatter(
        x=plot_x, y=plot_equity,
        customdata=plot_date_strs,
        name='Strategy', line=dict(color='#2196F3', width=1.5),
        hovertemplate='Trade #%{x} | %{customdata}<br>Equity: $%{y:,.2f}<extra></extra>'
    ))
    fig_equity.add_trace(go.Scatter(
        x=plot_x, y=plot_buyhold,
        customdata=plot_date_strs,
        name='Buy & Hold', line=dict(color='#9E9E9E', width=1, dash='dot'),
        hovertemplate='Trade #%{x} | %{customdata}<br>B&H: $%{y:,.2f}<extra></extra>'
    ))
    fig_equity.update_layout(
        margin=dict(l=50, r=20, t=30, b=30),
        legend=dict(x=0.5, y=1.15, xanchor='center', yanchor='top',
                    orientation='h', bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified',
        xaxis=dict(title='Trade #', showgrid=True, gridcolor='#2a2a2a'),
        yaxis=dict(showgrid=True, gridcolor='#2a2a2a', tickprefix='$'),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        font=dict(color='#e0e0e0'),
        annotations=[dict(
            text=date_range_str,
            xref='paper', yref='paper',
            x=0.0, y=1.02,
            showarrow=False,
            font=dict(size=12, color='#aaa'),
            xanchor='left', yanchor='bottom',
        )],
    )

    # Drawdown duration formatting
    dd_duration = bt.max_drawdown_duration
    if hasattr(dd_duration, 'days'):
        dd_dur_str = f'{dd_duration.days}d {dd_duration.seconds // 3600}h {(dd_duration.seconds % 3600) // 60}m'
    else:
        dd_dur_str = str(dd_duration)

    # 2. Enhanced Drawdown chart — highlight max drawdown period
    # Find max drawdown period in trade-level data
    dd_end_idx = np.argmin(trade_drawdown)
    dd_start_idx = np.argmax(
        trade_equity[:dd_end_idx]) if dd_end_idx > 0 else 0
    dd_duration_trades = dd_end_idx - dd_start_idx

    fig_drawdown = go.Figure()
    fig_drawdown.add_trace(go.Scatter(
        x=plot_x, y=plot_drawdown,
        fill='tozeroy', fillcolor='rgba(244,67,54,0.15)',
        line=dict(color='#f44336', width=1),
        hovertemplate='Trade #%{x}<br>Drawdown: $%{y:,.2f}<extra></extra>',
        cliponaxis=False
    ))
    # Highlight max drawdown region
    fig_drawdown.add_vrect(
        x0=dd_start_idx + 1, x1=dd_end_idx + 1,
        fillcolor='rgba(244,67,54,0.12)', line_width=0,
    )
    # Annotation for max drawdown — position at 40% of the way from trough to 0
    dd_annotation_y = trade_drawdown[dd_end_idx] * 0.6
    dd_min = trade_drawdown[dd_end_idx]
    fig_drawdown.add_annotation(
        x=(dd_start_idx + dd_end_idx) / 2 + 1,
        y=dd_annotation_y,
        text=f'Max DD: ${dd_min:,.0f}<br>{dd_duration_trades} trades | {dd_dur_str}',
        showarrow=True, arrowhead=2, arrowcolor='#f44336',
        ax=0, ay=0,
        yanchor='bottom',
        font=dict(size=11, color='#f44336'),
        bgcolor='rgba(26,26,26,0.8)', bordercolor='#f44336', borderwidth=1,
    )
    # Pad y-axis so trough + annotation always visible
    fig_drawdown.update_layout(
        margin=dict(l=50, r=20, t=50, b=30),
        hovermode='x unified',
        xaxis=dict(title=dict(text='Trade #', standoff=5), showgrid=True,
                   gridcolor='#2a2a2a', side='top'),
        yaxis=dict(showgrid=True, gridcolor='#2a2a2a',
                   tickprefix='$'),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        font=dict(color='#e0e0e0'),
    )

    # 2b. Bootstrap MC Drawdown
    mc_dd_data = bt.bootstrap_mc_drawdown(
        iterations=10000, max_paths=100, max_points=100)

    # Left panel: drawdown paths
    fig_mc_dd_paths = go.Figure()
    for i in range(mc_dd_data['dd_paths'].shape[0]):
        fig_mc_dd_paths.add_trace(go.Scatter(
            x=mc_dd_data['path_x'], y=mc_dd_data['dd_paths'][i],
            mode='lines', line=dict(color='rgba(100,100,100,0.25)', width=0.7),
            hoverinfo='skip', showlegend=False,
        ))
    fig_mc_dd_paths.add_trace(go.Scatter(
        x=mc_dd_data['path_x'], y=mc_dd_data['original_dd_path'],
        mode='lines', line=dict(color='#f44336', width=2.5),
        name='Original', hovertemplate='Trade %{x}<br>DD: $%{y:,.2f}<extra></extra>',
    ))
    fig_mc_dd_paths.update_layout(
        margin=dict(l=50, r=20, t=50, b=20),
        hovermode='x unified',
        xaxis=dict(title=dict(text='Trade #', standoff=5), showgrid=True,
                   gridcolor='#2a2a2a', side='top'),
        yaxis=dict(showgrid=True, gridcolor='#2a2a2a', tickprefix='$'),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        font=dict(color='#e0e0e0'),
        legend=dict(x=0.5, y=1.12, xanchor='center', yanchor='bottom',
                    orientation='h', bgcolor='rgba(0,0,0,0)'),
    )

    # Middle panel: histogram of max drawdowns
    fig_mc_dd_hist = go.Figure()
    fig_mc_dd_hist.add_trace(go.Histogram(
        x=mc_dd_data['max_drawdowns'], nbinsx=80,
        marker=dict(color='rgba(244, 67, 54, 0.6)', line=dict(
            color='rgba(244, 67, 54, 0.9)', width=0.5)),
        hovertemplate='Max DD: $%{x:,.0f}<br>Count: %{y}<extra></extra>',
    ))
    fig_mc_dd_hist.add_vline(x=mc_dd_data['original_max_dd'],
                             line=dict(color='#f44336', width=2.5),
                             annotation_text='Original', annotation_position='top')
    fig_mc_dd_hist.add_vrect(x0=mc_dd_data['ci_95'][0], x1=mc_dd_data['ci_95'][1],
                             fillcolor='rgba(255, 152, 0, 0.12)', line_width=0,
                             annotation_text='95% CI', annotation_position='top left')
    fig_mc_dd_hist.add_vline(x=mc_dd_data['ci_95'][0], line=dict(
        color='#FF9800', width=1.5, dash='dot'))
    fig_mc_dd_hist.add_vline(x=mc_dd_data['ci_95'][1], line=dict(
        color='#FF9800', width=1.5, dash='dot'))
    fig_mc_dd_hist.add_vline(x=mc_dd_data['ci_997'][0], line=dict(
        color='#9C27B0', width=1, dash='dashdot'))
    fig_mc_dd_hist.add_vline(x=mc_dd_data['ci_997'][1], line=dict(
        color='#9C27B0', width=1, dash='dashdot'))
    fig_mc_dd_hist.update_layout(
        margin=dict(l=50, r=20, t=30, b=95),
        xaxis=dict(title='Max Drawdown ($)', showgrid=True,
                   gridcolor='#2a2a2a', tickprefix='$', tickformat=',.0f',
                   showticklabels=True, tickangle=0, tickfont=dict(size=11)),
        yaxis=dict(title='Frequency', showgrid=True, gridcolor='#2a2a2a'),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        font=dict(color='#e0e0e0'),
        showlegend=False,
    )

    # 3. Bootstrap MC Resampling
    mc_data = bt.bootstrap_mc_resampling(
        iterations=10000, max_paths=100, max_points=100)

    # Left panel: equity paths
    fig_mc_paths = go.Figure()
    for i in range(mc_data['paths'].shape[0]):
        fig_mc_paths.add_trace(go.Scatter(
            x=mc_data['path_x'], y=mc_data['paths'][i],
            mode='lines', line=dict(color='rgba(100,100,100,0.25)', width=0.7),
            hoverinfo='skip', showlegend=False,
        ))
    fig_mc_paths.add_trace(go.Scatter(
        x=mc_data['path_x'], y=mc_data['original_path'],
        mode='lines', line=dict(color='#2196F3', width=2.5),
        name='Original', hovertemplate='Trade %{x}<br>PnL: $%{y:,.2f}<extra></extra>',
    ))
    fig_mc_paths.update_layout(
        margin=dict(l=50, r=20, t=0, b=50),
        hovermode='x unified',
        xaxis=dict(title='Trade #', showgrid=True, gridcolor='#2a2a2a'),
        yaxis=dict(showgrid=True, gridcolor='#2a2a2a', tickprefix='$'),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        font=dict(color='#e0e0e0'),
        legend=dict(x=0.5, y=1.12, xanchor='center', yanchor='top',
                    orientation='h', bgcolor='rgba(0,0,0,0)'),
    )

    # Right panel: histogram of final PnLs
    fig_mc_hist = go.Figure()
    fig_mc_hist.add_trace(go.Histogram(
        x=mc_data['final_pnls'], nbinsx=80,
        marker=dict(color='rgba(156, 39, 176, 0.6)', line=dict(
            color='rgba(156, 39, 176, 0.9)', width=0.5)),
        hovertemplate='PnL: $%{x:,.0f}<br>Count: %{y}<extra></extra>',
    ))
    # Original PnL line
    fig_mc_hist.add_vline(x=mc_data['original_final_pnl'], line=dict(color='#2196F3', width=2.5),
                          annotation_text='Original', annotation_position='top')
    # Mean line
    # fig_mc_hist.add_vline(x=mc_data['mean_pnl'], line=dict(color='#FF9800', width=2, dash='dash'),
    #                       annotation_text='Mean', annotation_position='top')
    # 95% CI
    fig_mc_hist.add_vrect(x0=mc_data['ci_95'][0], x1=mc_data['ci_95'][1],
                          fillcolor='rgba(76, 175, 80, 0.12)', line_width=0,
                          annotation_text='95% CI', annotation_position='top left')
    fig_mc_hist.add_vline(x=mc_data['ci_95'][0], line=dict(
        color='#4CAF50', width=1.5, dash='dot'))
    fig_mc_hist.add_vline(x=mc_data['ci_95'][1], line=dict(
        color='#4CAF50', width=1.5, dash='dot'))
    # 99.7% CI
    fig_mc_hist.add_vline(x=mc_data['ci_997'][0], line=dict(
        color='#f44336', width=1, dash='dashdot'))
    fig_mc_hist.add_vline(x=mc_data['ci_997'][1], line=dict(
        color='#f44336', width=1, dash='dashdot'))

    fig_mc_hist.update_layout(
        margin=dict(l=50, r=20, t=30, b=95),
        xaxis=dict(title='Final PnL ($)', showgrid=True,
                   gridcolor='#2a2a2a', tickprefix='$', tickformat=',.0f',
                   showticklabels=True, tickangle=0, tickfont=dict(size=11)),
        yaxis=dict(title='Frequency', showgrid=True, gridcolor='#2a2a2a'),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        font=dict(color='#e0e0e0'),
        showlegend=False,
    )

    # --- Helpers ---
    def _fmt_dollar(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 'N/A'
        return f'${v:,.2f}'

    def _fmt_pct(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 'N/A'
        return f'{v * 100:.2f}%'

    # MC CI text panel
    mc_ci_html = f'''
    <div class="mc-ci-section">
        <h3>Bootstrap Confidence Interval</h3>
        <div class="mc-ci-big {'positive' if mc_data['mean_pnl'] >= 0 else 'negative'}">{_fmt_dollar(mc_data['mean_pnl'])}</div>
        <div class="mc-ci-block">
            <div class="mc-ci-label">95% Confidence Interval</div>
            <div class="mc-ci-range ci-95-color">{_fmt_dollar(mc_data['ci_95'][0])} — {_fmt_dollar(mc_data['ci_95'][1])}</div>
            <div class="mc-ci-bar">
                <div class="mc-ci-fill-95"></div>
                <div class="mc-ci-marker"></div>
            </div>
        </div>
        <div class="mc-ci-block">
            <div class="mc-ci-label">99.7% Confidence Interval</div>
            <div class="mc-ci-range ci-997-color">{_fmt_dollar(mc_data['ci_997'][0])} — {_fmt_dollar(mc_data['ci_997'][1])}</div>
            <div class="mc-ci-bar">
                <div class="mc-ci-fill-997"></div>
                <div class="mc-ci-marker"></div>
            </div>
        </div>
        <div class="mc-ci-substat">
            <span>Median: {_fmt_dollar(np.median(mc_data['final_pnls']))}</span>
            <span>{mc_data['iterations']:,} iterations</span>
        </div>
    </div>
    '''

    # 4. Kelly Criterion — Equity projection chart
    b_kelly = bt.kelly_payoff_ratio if bt.kelly_payoff_ratio > 0 else 1.0
    kelly_fracs = [1, 1/2, 1/4, 1/8, 1/16]
    kelly_labels = ['Full Kelly', '1/2 Kelly',
                    '1/4 Kelly', '1/8 Kelly', '1/16 Kelly']
    kelly_colors = ['#FF9800', '#4CAF50', '#2196F3', '#9C27B0', '#607D8B']

    # Per-trade returns as fraction of equity (normalized to 1 contract flat bet)
    boundaries_k = np.flatnonzero(bt.trade_indicies)
    per_trade_pnl = np.add.reduceat(bt.candle_pnl, boundaries_k)
    n_kelly_trades = len(per_trade_pnl)

    # Current strategy: fixed 1 contract (flat sizing)
    current_equity = bt.initial_equity + per_trade_pnl.cumsum()

    # Simulate geometric Kelly sizing for each fraction
    kelly_equity_paths = {}
    kelly_final = {}
    for frac, label in zip(kelly_fracs, kelly_labels):
        f_val = bt.kelly_fraction * frac
        if f_val <= 0 or f_val >= 1:
            kelly_equity_paths[label] = current_equity.copy()
            kelly_final[label] = current_equity[-1]
            continue
        eq = np.empty(n_kelly_trades)
        bal = bt.initial_equity
        for i in range(n_kelly_trades):
            # Risk f_val of current bankroll per trade
            # pnl scaled: original pnl per trade assumes 1 contract
            # Kelly bet: risk fraction * bankroll, pnl proportional
            trade_return_pct = per_trade_pnl[i] / bt.initial_equity
            bal = bal * (1 + f_val * trade_return_pct * n_kelly_trades /
                         bt.total_trades * bt.total_trades / n_kelly_trades)
            eq[i] = bal
        # Simpler: geometric compounding with Kelly fraction
        # Each trade's return as pct of initial -> scale by fraction of current bankroll
        eq2 = np.empty(n_kelly_trades)
        bal2 = bt.initial_equity
        for i in range(n_kelly_trades):
            ret_pct = per_trade_pnl[i] / bt.initial_equity
            bal2 = bal2 * (1 + f_val * ret_pct)
            eq2[i] = bal2
        kelly_equity_paths[label] = eq2
        kelly_final[label] = eq2[-1]

    # Downsample for plotting
    step_k = max(1, n_kelly_trades // 2000)
    kelly_x = np.arange(1, n_kelly_trades + 1)[::step_k]

    fig_kelly = go.Figure()
    # Current flat sizing
    fig_kelly.add_trace(go.Scatter(
        x=kelly_x, y=current_equity[::step_k],
        name='Current (1 contract)', line=dict(color='#e0e0e0', width=2),
        hovertemplate='Trade #%{x}<br>Equity: $%{y:,.0f}<extra></extra>',
    ))
    # Kelly paths
    for label, color in zip(kelly_labels, kelly_colors):
        fig_kelly.add_trace(go.Scatter(
            x=kelly_x, y=kelly_equity_paths[label][::step_k],
            name=label, line=dict(color=color, width=1.5),
            hovertemplate='Trade #%{x}<br>Equity: $%{y:,.0f}<extra></extra>',
        ))

    fig_kelly.update_layout(
        margin=dict(l=50, r=20, t=30, b=30),
        legend=dict(x=0.5, y=1.15, xanchor='center', yanchor='top',
                    orientation='h', bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified',
        xaxis=dict(title='Trade #', showgrid=True, gridcolor='#2a2a2a'),
        yaxis=dict(showgrid=True, gridcolor='#2a2a2a',
                   tickprefix='$', type='log'),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        font=dict(color='#e0e0e0'),
    )

    # Kelly table panel
    kelly_table_rows = []
    for frac, label in zip(kelly_fracs, kelly_labels):
        f_val = bt.kelly_fraction * frac
        if f_val > 0 and b_kelly > 0 and f_val < 1:
            g_val = bt.win_rate * np.log(1 + f_val * b_kelly) + \
                (1 - bt.win_rate) * np.log(1 - f_val)
        else:
            g_val = 0.0
        final_eq = kelly_final.get(label, bt.initial_equity)
        kelly_table_rows.append((label, f_val, g_val, final_eq))

    kelly_table_html = '<div class="kelly-table-section">'
    kelly_table_html += '<h3>Kelly Criterion</h3>'
    kelly_table_html += f'<div class="kelly-big">{bt.kelly_fraction:.2%}</div>'
    kelly_table_html += f'<div class="kelly-sub">Payoff Ratio: {b_kelly:.2f} | Win Rate: {bt.win_rate:.2%}</div>'
    kelly_table_html += '<table class="kelly-table"><thead><tr><th>Fraction</th><th>Bet Size</th><th>Final Equity</th><th>Growth</th></tr></thead><tbody>'
    # Current row
    kelly_table_html += f'<tr><td>Current (flat)</td><td>—</td><td>${current_equity[-1]:,.0f}</td><td>—</td></tr>'
    for label, f_val, g_val, final_eq in kelly_table_rows:
        highlight = ' class="kelly-highlight"' if label == 'Full Kelly' else ''
        kelly_table_html += f'<tr{highlight}><td>{label}</td><td>{f_val:.2%}</td><td>${final_eq:,.0f}</td><td>{g_val:.5f}</td></tr>'
    kelly_table_html += '</tbody></table></div>'

    # MC Drawdown CI text panel
    mc_dd_ci_html = f'''
    <div class="mc-ci-section">
        <h3>Bootstrap Max Drawdown</h3>
        <div class="mc-ci-big negative">{_fmt_dollar(mc_dd_data['mean_max_dd'])}</div>
        <div class="mc-ci-block">
            <div class="mc-ci-label">95% Confidence Interval</div>
            <div class="mc-ci-range ci-95-color">{_fmt_dollar(mc_dd_data['ci_95'][0])} — {_fmt_dollar(mc_dd_data['ci_95'][1])}</div>
            <div class="mc-ci-bar">
                <div class="mc-ci-fill-95"></div>
                <div class="mc-ci-marker"></div>
            </div>
        </div>
        <div class="mc-ci-block">
            <div class="mc-ci-label">99.7% Confidence Interval</div>
            <div class="mc-ci-range ci-997-color">{_fmt_dollar(mc_dd_data['ci_997'][0])} — {_fmt_dollar(mc_dd_data['ci_997'][1])}</div>
            <div class="mc-ci-bar">
                <div class="mc-ci-fill-997"></div>
                <div class="mc-ci-marker"></div>
            </div>
        </div>
        <div class="mc-ci-substat">
            <span>Median: {_fmt_dollar(mc_dd_data['median_max_dd'])}</span>
            <span>{mc_dd_data['iterations']:,} iterations</span>
        </div>
    </div>
    '''

    # 5. Trade Visualization — Price chart with position highlighting + mini equity
    from plotly.subplots import make_subplots

    # Sequential integer x-axis eliminates weekend gaps; datetime shown via tick labels + hover
    n_total = len(bt.close_prices)
    tv_x = np.arange(n_total)
    tv_open = bt.ohlc[:, 0]
    tv_high = bt.ohlc[:, 1]
    tv_low = bt.ohlc[:, 2]
    tv_close = bt.ohlc[:, 3]
    tv_signal = bt.signal
    tv_equity = bt.equity_curve
    tv_date_strs = [d.strftime('%Y-%m-%d %H:%M') for d in bt.indicies]

    # Build tick labels: evenly spaced datetime labels on the integer axis
    n_ticks = 10
    tick_step = max(1, n_total // n_ticks)
    tv_tickvals = list(range(0, n_total, tick_step))
    tv_ticktext = [bt.indicies[k].strftime('%Y-%m-%d') for k in tv_tickvals]

    fig_trades = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )

    # Candlestick on top panel
    fig_trades.add_trace(go.Candlestick(
        x=tv_x, open=tv_open, high=tv_high, low=tv_low, close=tv_close,
        increasing_line_color='#4CAF50', decreasing_line_color='#f44336',
        increasing_fillcolor='#4CAF50', decreasing_fillcolor='#f44336',
        name='Price', showlegend=False,
        text=[f'{tv_date_strs[k]}<br>O:{tv_open[k]:.2f} H:{tv_high[k]:.2f} L:{tv_low[k]:.2f} C:{tv_close[k]:.2f}' for k in range(
            n_total)],
        hoverinfo='text',
    ), row=1, col=1)

    # Build position regions (consolidate consecutive same-signal bars into rectangles)
    shapes = []
    i = 0
    while i < n_total:
        sig = tv_signal[i]
        if sig == 0:
            i += 1
            continue
        j = i
        while j < n_total and tv_signal[j] == sig:
            j += 1
        color = 'rgba(76, 175, 80, 0.08)' if sig == 1 else 'rgba(244, 67, 54, 0.08)'
        shapes.append(dict(
            type='rect', xref='x', yref='y domain',
            x0=i, x1=j - 1,
            y0=0, y1=1,
            fillcolor=color, line_width=0, layer='below',
        ))
        i = j

    # Equity curve on bottom panel
    fig_trades.add_trace(go.Scatter(
        x=tv_x, y=tv_equity,
        customdata=tv_date_strs,
        line=dict(color='#2196F3', width=1.5),
        name='Equity', showlegend=False,
        hovertemplate='%{customdata}<br>Equity: $%{y:,.2f}<extra></extra>',
    ), row=2, col=1)

    fig_trades.update_layout(
        margin=dict(l=50, r=20, t=30, b=30),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        font=dict(color='#e0e0e0'),
        hovermode='x unified',
        xaxis=dict(showgrid=True, gridcolor='#2a2a2a', rangeslider_visible=False,
                   tickvals=tv_tickvals, ticktext=tv_ticktext),
        yaxis=dict(showgrid=True, gridcolor='#2a2a2a', title='Price'),
        xaxis2=dict(showgrid=True, gridcolor='#2a2a2a',
                    tickvals=tv_tickvals, ticktext=tv_ticktext),
        yaxis2=dict(showgrid=True, gridcolor='#2a2a2a',
                    title='Equity', tickprefix='$'),
        shapes=shapes,
        legend=dict(x=0.5, y=1.05, xanchor='center', orientation='h',
                    bgcolor='rgba(0,0,0,0)'),
    )

    trades_chart_html = fig_trades.to_html(
        full_html=False, include_plotlyjs=False, div_id='trades-chart',
        config={'responsive': True})

    # --- Convert figures to HTML divs ---
    equity_html = fig_equity.to_html(
        full_html=False, include_plotlyjs=False, div_id='equity-chart',
        config={'responsive': True})
    drawdown_html = fig_drawdown.to_html(
        full_html=False, include_plotlyjs=False, div_id='drawdown-chart',
        config={'responsive': True})
    mc_dd_paths_html = fig_mc_dd_paths.to_html(
        full_html=False, include_plotlyjs=False, div_id='mc-dd-paths-chart',
        config={'responsive': True})
    mc_dd_hist_html = fig_mc_dd_hist.to_html(
        full_html=False, include_plotlyjs=False, div_id='mc-dd-hist-chart',
        config={'responsive': True})
    mc_paths_html = fig_mc_paths.to_html(
        full_html=False, include_plotlyjs=False, div_id='mc-paths-chart',
        config={'responsive': True})
    mc_hist_html = fig_mc_hist.to_html(
        full_html=False, include_plotlyjs=False, div_id='mc-hist-chart',
        config={'responsive': True})
    kelly_chart_html = fig_kelly.to_html(
        full_html=False, include_plotlyjs=False, div_id='kelly-chart',
        config={'responsive': True})

    # --- EV box (top right) ---
    ev_html = f'''
    <div class="ev-section">
        <h3>Expected Value per Trade</h3>
        <div class="ev-big {'positive' if ev_mean >= 0 else 'negative'}">{_fmt_dollar(ev_mean)}</div>
        <div class="ci-block">
            <div class="ci-label">95% Confidence Interval</div>
            <div class="ci-range">{_fmt_dollar(ev_lb_95)} — {_fmt_dollar(ev_ub_95)}</div>
            <div class="ci-bar">
                <div class="ci-fill ci-95"></div>
                <div class="ci-marker"></div>
            </div>
        </div>
        <div class="ci-block">
            <div class="ci-label">99.7% Confidence Interval</div>
            <div class="ci-range">{_fmt_dollar(ev_lb_997)} — {_fmt_dollar(ev_ub_997)}</div>
            <div class="ci-bar">
                <div class="ci-fill ci-997"></div>
                <div class="ci-marker"></div>
            </div>
        </div>
        <div class="ev-substat">
            <span class="stat-label">n = {n_trades:,} trades</span>
            <span class="stat-label">std = {_fmt_dollar(ev_std)}</span>
        </div>
    </div>
    '''

    # --- Stats panel ---
    stats_html = f'''
    <div class="stats-grid">
        <div class="stat-section">
            <h3>Equity</h3>
            <div class="stat-row">
                <span class="stat-label">Initial</span>
                <span class="stat-value">{_fmt_dollar(bt.initial_equity)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Final</span>
                <span class="stat-value {'positive' if bt.final_equity >= bt.initial_equity else 'negative'}">{_fmt_dollar(bt.final_equity)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Strategy Return</span>
                <span class="stat-value {'positive' if bt.total_pct_return >= 0 else 'negative'}">{_fmt_pct(bt.total_pct_return)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Buy & Hold Return</span>
                <span class="stat-value">{_fmt_pct(bt.buyhold_pct_return)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">CAGR</span>
                <span class="stat-value">{_fmt_pct(bt.CAGR)}</span>
            </div>
        </div>

        <div class="stat-section">
            <h3>Trades</h3>
            <div class="stat-row">
                <span class="stat-label">Total Trades</span>
                <span class="stat-value">{bt.total_trades:,}</span>
            </div>
            <div class="stat-row highlight">
                <span class="stat-label">Win Rate</span>
                <span class="stat-value">{_fmt_pct(bt.win_rate)}</span>
            </div>
            <div class="stat-row ci-row">
                <span class="stat-label">95% CI</span>
                <span class="stat-value ci">{_fmt_pct(wr_lb_95)} — {_fmt_pct(wr_ub_95)}</span>
            </div>
            <div class="stat-row ci-row">
                <span class="stat-label">99.7% CI</span>
                <span class="stat-value ci">{_fmt_pct(wr_lb_997)} — {_fmt_pct(wr_ub_997)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Avg Win</span>
                <span class="stat-value positive">{_fmt_dollar(bt.average_win)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Avg Loss</span>
                <span class="stat-value negative">{_fmt_dollar(bt.average_loss)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Win/Loss Ratio</span>
                <span class="stat-value">{bt.ratio_avg_win_loss:.2f}</span>
            </div>
        </div>

        <div class="stat-section">
            <h3>Long Trades</h3>
            <div class="stat-row">
                <span class="stat-label">Total</span>
                <span class="stat-value">{bt.long_total_trades:,}</span>
            </div>
            <div class="stat-row highlight">
                <span class="stat-label">Win Rate</span>
                <span class="stat-value">{_fmt_pct(bt.long_win_rate)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Avg Win</span>
                <span class="stat-value positive">{_fmt_dollar(bt.long_average_win)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Avg Loss</span>
                <span class="stat-value negative">{_fmt_dollar(bt.long_average_loss)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Avg PnL / Trade</span>
                <span class="stat-value {'positive' if (bt.long_average_pnl or 0) >= 0 else 'negative'}">{_fmt_dollar(bt.long_average_pnl)}</span>
            </div>
            <div class="stat-row highlight">
                <span class="stat-label">Cumulative PnL</span>
                <span class="stat-value {'positive' if (bt.long_cumulative_pnl or 0) >= 0 else 'negative'}">{_fmt_dollar(bt.long_cumulative_pnl)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Largest Win</span>
                <span class="stat-value positive">{_fmt_dollar(bt.long_largest_win)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Largest Loss</span>
                <span class="stat-value negative">{_fmt_dollar(bt.long_largest_loss)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Profit Factor</span>
                <span class="stat-value">{bt.long_profit_factor:.2f}</span>
            </div>
        </div>

        <div class="stat-section">
            <h3>Short Trades</h3>
            <div class="stat-row">
                <span class="stat-label">Total</span>
                <span class="stat-value">{bt.short_total_trades:,}</span>
            </div>
            <div class="stat-row highlight">
                <span class="stat-label">Win Rate</span>
                <span class="stat-value">{_fmt_pct(bt.short_win_rate)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Avg Win</span>
                <span class="stat-value positive">{_fmt_dollar(bt.short_average_win)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Avg Loss</span>
                <span class="stat-value negative">{_fmt_dollar(bt.short_average_loss)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Avg PnL / Trade</span>
                <span class="stat-value {'positive' if (bt.short_average_pnl or 0) >= 0 else 'negative'}">{_fmt_dollar(bt.short_average_pnl)}</span>
            </div>
            <div class="stat-row highlight">
                <span class="stat-label">Cumulative PnL</span>
                <span class="stat-value {'positive' if (bt.short_cumulative_pnl or 0) >= 0 else 'negative'}">{_fmt_dollar(bt.short_cumulative_pnl)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Largest Win</span>
                <span class="stat-value positive">{_fmt_dollar(bt.short_largest_win)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Largest Loss</span>
                <span class="stat-value negative">{_fmt_dollar(bt.short_largest_loss)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Profit Factor</span>
                <span class="stat-value">{bt.short_profit_factor:.2f}</span>
            </div>
        </div>

        <div class="stat-section">
            <h3>Max Drawdown</h3>
            <div class="stat-row">
                <span class="stat-label">Drawdown</span>
                <span class="stat-value negative">{_fmt_dollar(bt.max_drawdown)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Drawdown %</span>
                <span class="stat-value negative">{_fmt_pct(bt.max_drawdown_pct)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Duration</span>
                <span class="stat-value">{dd_dur_str}</span>
            </div>
        </div>

        <div class="stat-section">
            <h3>Extremes</h3>
            <div class="stat-row">
                <span class="stat-label">Largest Win</span>
                <span class="stat-value positive">{_fmt_dollar(bt.largest_win)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Largest Loss</span>
                <span class="stat-value negative">{_fmt_dollar(bt.largest_loss)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Max Consec Wins</span>
                <span class="stat-value">{bt.max_consec_wins}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Max Consec Losses</span>
                <span class="stat-value">{bt.max_consec_losses}</span>
            </div>
        </div>

        <div class="stat-section">
            <h3>Performance</h3>
            <div class="stat-row">
                <span class="stat-label">Sharpe Ratio</span>
                <span class="stat-value">{bt.sharpe_ratio:.2f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Sortino Ratio</span>
                <span class="stat-value">{bt.sortino_ratio:.2f}</span>
            </div>
            <div class="stat-row highlight">
                <span class="stat-label">Profit Factor</span>
                <span class="stat-value">{bt.profit_factor:.2f}</span>
            </div>
            <div class="stat-row ci-row">
                <span class="stat-label">Granular (bar-level)</span>
                <span class="stat-value ci">{bt.profit_factor_granular:.2f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Alpha (ann.)</span>
                <span class="stat-value">{_fmt_pct(bt.alpha)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Beta</span>
                <span class="stat-value">{bt.beta:.2f}</span>
            </div>
        </div>

        <div class="stat-section">
            <h3>Costs</h3>
            <div class="stat-row">
                <span class="stat-label">Slippage / Trade</span>
                <span class="stat-value">{bt.slippage_ticks} ticks</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Commission / Trade</span>
                <span class="stat-value">{_fmt_dollar(bt.commission)}</span>
            </div>
            <div class="stat-row highlight">
                <span class="stat-label">Friction / Avg Trade</span>
                <span class="stat-value negative">{_fmt_pct(float(bt.transaction_costs.sum()) / bt.total_trades / abs(trade_returns).mean() if abs(trade_returns).mean() > 0 else 0)}</span>
            </div>
            <div class="stat-row highlight">
                <span class="stat-label">Total Costs</span>
                <span class="stat-value negative">{_fmt_dollar(float(bt.transaction_costs.sum()))}</span>
            </div>
            <div class="stat-row ci-row">
                <span class="stat-label">Slippage</span>
                <span class="stat-value ci">{_fmt_dollar(float(bt.slippage_costs.sum()))}</span>
            </div>
            <div class="stat-row ci-row">
                <span class="stat-label">Commission</span>
                <span class="stat-value ci">{_fmt_dollar(float(bt.commission_costs.sum()))}</span>
            </div>
        </div>
    </div>
    '''

    # --- Assemble full HTML ---
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report</title>
    <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #0d0d0d;
            color: #e0e0e0;
            font-family: 'Segoe UI', -apple-system, sans-serif;
            padding: 20px 24px;
        }}

        .container {{
            width: 100%;
            max-width: 100%;
        }}

        .top-row {{
            display: grid;
            grid-template-columns: 3fr 1fr;
            gap: 12px;
            margin-bottom: 12px;
            height: 460px;
        }}

        .panel {{
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            padding: 12px;
            overflow: hidden;
        }}

        .panel-title {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid #2a2a2a;
        }}

        /* Make plotly charts fill their panel */
        .chart-panel {{
            display: flex;
            flex-direction: column;
        }}
        .chart-panel .panel-title {{
            flex-shrink: 0;
        }}
        .chart-panel .js-plotly-plot,
        .chart-panel .plotly-graph-div {{
            width: 100% !important;
            height: 100% !important;
            flex: 1;
        }}

        /* Stats panel */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 10px;
        }}

        .stat-section h3 {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #666;
            margin-bottom: 10px;
            padding-bottom: 6px;
            border-bottom: 1px solid #2a2a2a;
        }}

        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            font-size: 13px;
        }}

        .stat-label {{ color: #999; }}
        .stat-value {{ font-weight: 500; font-variant-numeric: tabular-nums; }}
        .stat-value.positive {{ color: #4CAF50; }}
        .stat-value.negative {{ color: #f44336; }}
        .stat-value.ci {{ color: #FF9800; font-size: 12px; }}

        .stat-row.highlight {{
            background: #222;
            padding: 6px 8px;
            border-radius: 4px;
            margin: 4px -8px;
        }}

        .stat-row.ci-row {{
            padding-left: 12px;
        }}

        /* EV section */
        .ev-section {{
            background: #111;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        .ev-section h3 {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            margin-bottom: 12px;
            text-align: center;
        }}

        .ev-big {{
            font-size: 36px;
            font-weight: 700;
            text-align: center;
            margin: 8px 0 24px;
        }}
        .ev-big.positive {{ color: #4CAF50; }}
        .ev-big.negative {{ color: #f44336; }}

        .ev-substat {{
            display: flex;
            justify-content: space-between;
            margin-top: 16px;
            padding-top: 12px;
            border-top: 1px solid #2a2a2a;
            font-size: 12px;
        }}

        .ci-block {{
            margin: 10px 0;
            padding: 10px;
            background: #1a1a1a;
            border-radius: 6px;
        }}

        .ci-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #888;
            margin-bottom: 4px;
        }}

        .ci-range {{
            font-size: 15px;
            font-weight: 500;
            color: #FF9800;
            margin-bottom: 8px;
        }}

        .ci-bar {{
            height: 6px;
            background: #333;
            border-radius: 3px;
            position: relative;
            overflow: hidden;
        }}

        .ci-fill {{
            position: absolute;
            height: 100%;
            border-radius: 3px;
        }}

        .ci-fill.ci-95 {{
            background: rgba(255, 152, 0, 0.5);
            left: 15%;
            right: 15%;
        }}

        .ci-fill.ci-997 {{
            background: rgba(255, 152, 0, 0.3);
            left: 5%;
            right: 5%;
        }}

        .ci-marker {{
            position: absolute;
            top: -2px;
            left: 50%;
            width: 3px;
            height: 10px;
            background: #FF9800;
            border-radius: 2px;
            transform: translateX(-50%);
        }}

        /* Below fold */
        .below-fold {{
            margin-top: 12px;
            display: grid;
            grid-template-columns: 1fr;
            gap: 12px;
        }}

        .below-fold .panel {{
            height: 480px;
        }}

        /* Monte Carlo section */
        .mc-row {{
            margin-top: 12px;
            display: grid;
            grid-template-columns: 2fr 2fr 1fr;
            gap: 12px;
        }}

        .mc-row .panel {{
            height: 450px;
        }}

        /* MC CI text panel */
        .mc-ci-section {{
            background: #111;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        .mc-ci-section h3 {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            margin-bottom: 12px;
            text-align: center;
        }}

        .mc-ci-big {{
            font-size: 28px;
            font-weight: 700;
            text-align: center;
            margin: 6px 0 20px;
        }}
        .mc-ci-big.positive {{ color: #4CAF50; }}
        .mc-ci-big.negative {{ color: #f44336; }}

        .mc-ci-block {{
            margin: 8px 0;
            padding: 10px;
            background: #1a1a1a;
            border-radius: 6px;
        }}

        .mc-ci-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #888;
            margin-bottom: 4px;
        }}

        .mc-ci-range {{
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 6px;
        }}
        .mc-ci-range.ci-95-color {{ color: #4CAF50; }}
        .mc-ci-range.ci-997-color {{ color: #f44336; }}

        .mc-ci-bar {{
            height: 6px;
            background: #333;
            border-radius: 3px;
            position: relative;
            overflow: hidden;
        }}

        .mc-ci-fill-95 {{
            position: absolute;
            height: 100%;
            border-radius: 3px;
            background: rgba(76, 175, 80, 0.5);
            left: 15%;
            right: 15%;
        }}

        .mc-ci-fill-997 {{
            position: absolute;
            height: 100%;
            border-radius: 3px;
            background: rgba(244, 67, 54, 0.3);
            left: 5%;
            right: 5%;
        }}

        .mc-ci-marker {{
            position: absolute;
            top: -2px;
            left: 50%;
            width: 3px;
            height: 10px;
            background: #2196F3;
            border-radius: 2px;
            transform: translateX(-50%);
        }}

        .mc-ci-substat {{
            display: flex;
            justify-content: space-between;
            margin-top: 14px;
            padding-top: 10px;
            border-top: 1px solid #2a2a2a;
            font-size: 11px;
            color: #999;
        }}

        /* Kelly Criterion section */
        .kelly-row {{
            margin-top: 12px;
            display: grid;
            grid-template-columns: 2fr 2fr 1fr;
            gap: 12px;
        }}

        .kelly-row .panel {{
            height: 520px;
        }}

        .kelly-table-section {{
            background: #111;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        .kelly-table-section h3 {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            margin-bottom: 12px;
            text-align: center;
        }}

        .kelly-big {{
            font-size: 32px;
            font-weight: 700;
            text-align: center;
            color: #2196F3;
            margin: 4px 0 8px;
        }}

        .kelly-sub {{
            font-size: 12px;
            color: #999;
            text-align: center;
            margin-bottom: 16px;
        }}

        .kelly-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}

        .kelly-table th {{
            text-align: left;
            color: #888;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 6px 8px;
            border-bottom: 1px solid #2a2a2a;
        }}

        .kelly-table td {{
            padding: 6px 8px;
            color: #e0e0e0;
            font-variant-numeric: tabular-nums;
        }}

        .kelly-table tr.kelly-highlight {{
            background: #222;
            border-radius: 4px;
        }}

        .kelly-table tr.kelly-highlight td {{
            color: #2196F3;
            font-weight: 600;
        }}

        /* Streak probability section */
        .streak-section {{
            margin-top: 12px;
        }}

        .streak-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .streak-header-left {{
            display: flex;
            align-items: baseline;
            gap: 12px;
        }}

        .streak-header-left h2 {{
            font-size: 16px;
            font-weight: 600;
            color: #e0e0e0;
            margin: 0;
        }}

        .streak-header-left .streak-winrate {{
            font-size: 13px;
            color: #888;
        }}

        .streak-slider-wrap {{
            display: flex;
            align-items: center;
            gap: 12px;
            background: #111;
            border: 1px solid #2a2a2a;
            border-radius: 24px;
            padding: 6px 18px;
        }}

        .streak-slider-wrap label {{
            font-size: 12px;
            color: #999;
            white-space: nowrap;
        }}

        #streak-n-display {{
            font-weight: 700;
            color: #2196F3;
            font-size: 14px;
            min-width: 52px;
            display: inline-block;
            text-align: right;
        }}

        #streak-slider {{
            width: 220px;
            height: 4px;
            -webkit-appearance: none;
            appearance: none;
            background: linear-gradient(to right, #2a2a2a, #2196F3);
            border-radius: 2px;
            outline: none;
            cursor: pointer;
        }}

        #streak-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #2196F3;
            box-shadow: 0 0 8px rgba(33, 150, 243, 0.5);
            cursor: pointer;
        }}

        #streak-slider::-moz-range-thumb {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #2196F3;
            box-shadow: 0 0 8px rgba(33, 150, 243, 0.5);
            border: none;
            cursor: pointer;
        }}

        .streak-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }}

        .streak-col {{
            background: #111;
            border: 1px solid #2a2a2a;
            border-radius: 10px;
            padding: 16px;
        }}

        .streak-col-title {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #2a2a2a;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .streak-col-title.win {{ color: #4CAF50; }}
        .streak-col-title.loss {{ color: #f44336; }}

        .streak-col-title::before {{
            content: '';
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }}

        .streak-col-title.win::before {{ background: #4CAF50; box-shadow: 0 0 6px #4CAF50; }}
        .streak-col-title.loss::before {{ background: #f44336; box-shadow: 0 0 6px #f44336; }}

        .streak-heatmap {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(70px, 1fr));
            gap: 6px;
        }}

        .streak-cell {{
            border-radius: 6px;
            padding: 10px 6px;
            text-align: center;
            position: relative;
            transition: transform 0.1s, box-shadow 0.1s;
            border: 1px solid rgba(255,255,255,0.04);
        }}

        .streak-cell:hover {{
            transform: scale(1.05);
            z-index: 1;
        }}

        .streak-cell .streak-k {{
            font-size: 18px;
            font-weight: 700;
            color: #fff;
            display: block;
            margin-bottom: 2px;
        }}

        .streak-cell .streak-dollar {{
            font-size: 11px;
            font-weight: 600;
            color: rgba(255,255,255,0.95);
            display: block;
            margin-bottom: 1px;
        }}

        .streak-cell .streak-pct {{
            font-size: 10px;
            font-weight: 500;
            color: rgba(255,255,255,0.6);
        }}

        .streak-legend {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin-top: 16px;
            font-size: 11px;
            color: #666;
        }}

        .streak-legend-bar {{
            width: 160px;
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(to right, rgba(255,255,255,0.03), #4CAF50);
        }}

        .streak-legend-bar.loss-bar {{
            background: linear-gradient(to right, rgba(255,255,255,0.03), #f44336);
        }}

        @media (max-width: 1000px) {{
            .top-row {{
                grid-template-columns: 1fr;
                height: auto;
            }}
            .top-row .chart-panel {{
                height: 330px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Top row: equity chart left, EV box right -->
        <div class="top-row">
            <div class="panel chart-panel">
                <div class="panel-title">Strategy vs Buy & Hold</div>
                {equity_html}
            </div>
            <div class="panel" style="padding: 0;">
                {ev_html}
            </div>
        </div>

        <!-- Stats panel -->
        <div class="panel" style="margin-bottom: 12px;">
            <div class="panel-title">Backtest Statistics</div>
            {stats_html}
        </div>

        <!-- Trade Visualization: Price + Position Highlighting + Equity -->
        <div class="panel chart-panel" style="height: 520px; margin-bottom: 12px;">
            <div class="panel-title">Trade Visualization</div>
            {trades_chart_html}
        </div>

        <!-- Below fold charts -->
        <div class="below-fold">
            <div class="panel chart-panel">
                <div class="panel-title">Drawdown</div>
                {drawdown_html}
            </div>
        </div>

        <!-- Monte Carlo Drawdown Bootstrap Section -->
        <div class="mc-row">
            <div class="panel chart-panel">
                <div class="panel-title">Bootstrapped MC — Drawdown Paths</div>
                {mc_dd_paths_html}
            </div>
            <div class="panel chart-panel">
                <div class="panel-title">Bootstrapped MC — Max Drawdown Distribution</div>
                {mc_dd_hist_html}
            </div>
            <div class="panel" style="padding: 0;">
                {mc_dd_ci_html}
            </div>
        </div>

        <!-- Monte Carlo Bootstrap Section -->
        <div class="mc-row">
            <div class="panel chart-panel">
                <div class="panel-title">Bootstrapped MC Resampling — Equity Paths</div>
                {mc_paths_html}
            </div>
            <div class="panel chart-panel">
                <div class="panel-title">Bootstrapped MC Resampling — Distribution</div>
                {mc_hist_html}
            </div>
            <div class="panel" style="padding: 0;">
                {mc_ci_html}
            </div>
        </div>

        <!-- Kelly Criterion Section -->
        <div class="kelly-row">
            <div class="panel chart-panel">
                <div class="panel-title">Kelly Criterion — Equity Projection</div>
                {kelly_chart_html}
            </div>
            <div class="panel" style="padding: 0;">
                {kelly_table_html}
            </div>
        </div>

        <!-- Streak Probability Section -->
        <div class="panel streak-section" style="padding: 20px;">
            <div class="streak-header">
                <div class="streak-header-left">
                    <h2>Streak Probabilities</h2>
                    <span class="streak-winrate">Win Rate: {_fmt_pct(bt.win_rate)}</span>
                </div>
                <div class="streak-slider-wrap">
                    <label for="streak-slider">Trades:</label>
                    <input type="range" id="streak-slider" min="1" max="100000" value="{bt.total_trades}" step="1">
                    <span id="streak-n-display">{bt.total_trades:,}</span>
                </div>
            </div>
            <div class="streak-grid">
                <div class="streak-col">
                    <div class="streak-col-title win">Winning Streaks</div>
                    <div class="streak-heatmap" id="streak-win-grid"></div>
                    <div class="streak-legend"><span>0%</span><div class="streak-legend-bar"></div><span>100%</span></div>
                </div>
                <div class="streak-col">
                    <div class="streak-col-title loss">Losing Streaks</div>
                    <div class="streak-heatmap" id="streak-loss-grid"></div>
                    <div class="streak-legend"><span>0%</span><div class="streak-legend-bar loss-bar"></div><span>100%</span></div>
                </div>
            </div>
        </div>

    </div>

    <script>
        window.addEventListener('resize', function() {{
            document.querySelectorAll('.js-plotly-plot').forEach(function(plot) {{
                Plotly.Plots.resize(plot);
            }});
        }});
        setTimeout(function() {{
            document.querySelectorAll('.js-plotly-plot').forEach(function(plot) {{
                Plotly.Plots.resize(plot);
            }});
        }}, 100);

        // Auto-rescale y-axis on x-zoom for specified charts
        function attachAutoRescaleY(divId, yAxes) {{
            var el = document.getElementById(divId);
            if (!el || !el.on) return false;
            var busy = false;
            el.on('plotly_relayout', function(ed) {{
                if (busy) return;
                if (ed['xaxis.autorange'] || ed['autosize']) {{
                    busy = true;
                    var update = {{}};
                    yAxes.forEach(function(ya) {{ update[ya.yAxisKey.replace('.range', '.autorange')] = true; }});
                    Plotly.relayout(el, update).then(function() {{ busy = false; }});
                    return;
                }}
                var xMin = ed['xaxis.range[0]'];
                var xMax = ed['xaxis.range[1]'];
                if (xMin == null || xMax == null) return;
                xMin = parseFloat(xMin);
                xMax = parseFloat(xMax);
                var data = el.data;
                var update = {{}};
                yAxes.forEach(function(ya) {{
                    var yMin = Infinity, yMax = -Infinity;
                    ya.traceIndices.forEach(function(ti) {{
                        var trace = data[ti];
                        if (!trace) return;
                        var xs = trace.x;
                        if (!xs) return;
                        for (var i = 0; i < xs.length; i++) {{
                            var xv = xs[i];
                            if (xv < xMin || xv > xMax) continue;
                            if (trace.type === 'candlestick') {{
                                if (trace.high && trace.high[i] != null) {{
                                    if (trace.high[i] > yMax) yMax = trace.high[i];
                                }}
                                if (trace.low && trace.low[i] != null) {{
                                    if (trace.low[i] < yMin) yMin = trace.low[i];
                                }}
                            }} else {{
                                var yv = trace.y[i];
                                if (yv == null) continue;
                                if (yv < yMin) yMin = yv;
                                if (yv > yMax) yMax = yv;
                            }}
                        }}
                    }});
                    if (yMin !== Infinity && yMax !== -Infinity) {{
                        var pad = (yMax - yMin) * 0.05 || 1;
                        update[ya.yAxisKey] = [yMin - pad, yMax + pad];
                    }}
                }});
                if (Object.keys(update).length > 0) {{
                    busy = true;
                    Plotly.relayout(el, update).then(function() {{ busy = false; }});
                }}
            }});
            return true;
        }}

        // Attach after Plotly fully initializes the divs
        setTimeout(function() {{
            attachAutoRescaleY('equity-chart', [
                {{traceIndices: [0, 1], yAxisKey: 'yaxis.range'}}
            ]);
            attachAutoRescaleY('trades-chart', [
                {{traceIndices: [0], yAxisKey: 'yaxis.range'}},
                {{traceIndices: [1], yAxisKey: 'yaxis2.range'}}
            ]);
            attachAutoRescaleY('drawdown-chart', [
                {{traceIndices: [0], yAxisKey: 'yaxis.range'}}
            ]);
        }}, 500);

        // Streak probability computation (Feller's recurrence)
        (function() {{
            const WIN_RATE = {bt.win_rate};
            const AVG_WIN = {abs(bt.average_win) if bt.average_win else 0};
            const AVG_LOSS = {abs(bt.average_loss) if bt.average_loss else 0};
            const STREAK_LENGTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 30];

            function probStreakAtLeast(p, k, N) {{
                if (k <= 0 || N <= 0) return 0;
                if (k === 1) return 1 - Math.pow(1 - p, N);
                if (N < k) return Math.pow(p, N);
                var q = 1 - p;
                var f = new Float64Array(N + 1);
                for (var i = 0; i < k; i++) f[i] = 1.0;
                for (var n = k; n <= N; n++) {{
                    var s = 0;
                    var pk = 1;
                    for (var j = 0; j < k; j++) {{
                        s += pk * f[n - 1 - j];
                        pk *= p;
                    }}
                    f[n] = q * s;
                }}
                return 1 - f[N];
            }}

            function formatDollar(v) {{
                if (v >= 1000) return '$' + (v / 1000).toFixed(1) + 'k';
                return '$' + v.toFixed(0);
            }}

            function makeCell(k, prob, type) {{
                var pct = (prob * 100).toFixed(1);
                var intensity = prob;
                var bg, shadow, dollar;
                if (type === 'win') {{
                    bg = 'rgba(76, 175, 80, ' + (intensity * 0.55 + 0.05).toFixed(3) + ')';
                    shadow = prob > 0.5 ? '0 0 12px rgba(76, 175, 80, ' + (intensity * 0.4).toFixed(2) + ')' : 'none';
                    dollar = k * AVG_WIN;
                }} else {{
                    bg = 'rgba(244, 67, 54, ' + (intensity * 0.55 + 0.05).toFixed(3) + ')';
                    shadow = prob > 0.5 ? '0 0 12px rgba(244, 67, 54, ' + (intensity * 0.4).toFixed(2) + ')' : 'none';
                    dollar = k * AVG_LOSS;
                }}
                return '<div class="streak-cell" style="background:' + bg + ';box-shadow:' + shadow + '">'
                     + '<span class="streak-k">' + k + '</span>'
                     + '<span class="streak-dollar">' + formatDollar(dollar) + '</span>'
                     + '<span class="streak-pct">' + pct + '%</span></div>';
            }}

            function buildGrid(N) {{
                var winGrid = document.getElementById('streak-win-grid');
                var lossGrid = document.getElementById('streak-loss-grid');
                var winHtml = '', lossHtml = '';
                for (var i = 0; i < STREAK_LENGTHS.length; i++) {{
                    var k = STREAK_LENGTHS[i];
                    var pWin = probStreakAtLeast(WIN_RATE, k, N);
                    var pLoss = probStreakAtLeast(1 - WIN_RATE, k, N);
                    winHtml += makeCell(k, pWin, 'win');
                    lossHtml += makeCell(k, pLoss, 'loss');
                }}
                winGrid.innerHTML = winHtml;
                lossGrid.innerHTML = lossHtml;
            }}

            var slider = document.getElementById('streak-slider');
            var display = document.getElementById('streak-n-display');
            var debounceTimer = null;

            function onSliderChange() {{
                var N = parseInt(slider.value);
                display.textContent = N.toLocaleString();
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(function() {{ buildGrid(N); }}, 30);
            }}

            slider.addEventListener('input', onSliderChange);
            buildGrid({bt.total_trades});
        }})();
    </script>
</body>
</html>'''

    output_path = str(Path(output_path).resolve())
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f'Report saved: {output_path}')

    if open_browser:
        webbrowser.open(f'file:///{output_path}')

    return output_path
