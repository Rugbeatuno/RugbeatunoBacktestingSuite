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

    # Downsample for plotting (by trade number)
    step = max(1, len(trade_equity) // 2000)
    plot_x = trade_x[::step]
    plot_equity = trade_equity[::step]
    plot_buyhold = np.linspace(
        buyhold_equity[0], buyhold_equity[-1], len(plot_x))
    plot_drawdown = trade_drawdown[::step]

    # Date range string
    start_str = bt.start_date.strftime('%Y-%m-%d')
    end_str = bt.end_date.strftime('%Y-%m-%d')
    date_range_str = f'{start_str}  —  {end_str}'

    # --- Build Plotly figures ---

    # 1. Main equity chart
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=plot_x, y=plot_equity,
        name='Strategy', line=dict(color='#2196F3', width=1.5),
        hovertemplate='Trade #%{x}<br>Equity: $%{y:,.2f}<extra></extra>'
    ))
    fig_equity.add_trace(go.Scatter(
        x=plot_x, y=plot_buyhold,
        name='Buy & Hold', line=dict(color='#9E9E9E', width=1, dash='dot'),
        hovertemplate='Trade #%{x}<br>B&H: $%{y:,.2f}<extra></extra>'
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

    # 2. Drawdown chart
    fig_drawdown = go.Figure()
    fig_drawdown.add_trace(go.Scatter(
        x=plot_x, y=plot_drawdown,
        fill='tozeroy', fillcolor='rgba(244,67,54,0.3)',
        line=dict(color='#f44336', width=1),
        hovertemplate='Trade #%{x}<br>Drawdown: $%{y:,.2f}<extra></extra>'
    ))
    fig_drawdown.update_layout(
        margin=dict(l=50, r=20, t=30, b=50),
        hovermode='x unified',
        xaxis=dict(title='Trade #', showgrid=True, gridcolor='#2a2a2a'),
        yaxis=dict(showgrid=True, gridcolor='#2a2a2a', tickprefix='$'),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        font=dict(color='#e0e0e0'),
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
        margin=dict(l=50, r=20, t=30, b=20),
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
            <span>{mc_data['iterations']:,} iterations</span>
            <span>{mc_data['n_trades']:,} trades</span>
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

    # --- Convert figures to HTML divs ---
    equity_html = fig_equity.to_html(
        full_html=False, include_plotlyjs=False, div_id='equity-chart',
        config={'responsive': True})
    drawdown_html = fig_drawdown.to_html(
        full_html=False, include_plotlyjs=False, div_id='drawdown-chart',
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

    # Drawdown duration formatting
    dd_duration = bt.max_drawdown_duration
    if hasattr(dd_duration, 'days'):
        dd_dur_str = f'{dd_duration.days}d {dd_duration.seconds // 3600}h {(dd_duration.seconds % 3600) // 60}m'
    else:
        dd_dur_str = str(dd_duration)

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
            <h3>Performance</h3>
            <div class="stat-row">
                <span class="stat-label">Sharpe Ratio</span>
                <span class="stat-value">{bt.sharpe_ratio:.2f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Sortino Ratio</span>
                <span class="stat-value">{bt.sortino_ratio:.2f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Profit Factor</span>
                <span class="stat-value">{bt.profit_factor:.2f}</span>
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
            height: 330px;
        }}

        /* Monte Carlo section */
        .mc-row {{
            margin-top: 12px;
            display: grid;
            grid-template-columns: 2fr 2fr 1fr;
            gap: 12px;
        }}

        .mc-row .panel {{
            height: 420px;
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

        @media (max-width: 1000px) {{
            .top-row {{
                grid-template-columns: 1fr;
                height: auto;
            }}
            .top-row .chart-panel {{
                height: 350px;
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

        <!-- Below fold charts -->
        <div class="below-fold">
            <div class="panel chart-panel">
                <div class="panel-title">Drawdown</div>
                {drawdown_html}
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
