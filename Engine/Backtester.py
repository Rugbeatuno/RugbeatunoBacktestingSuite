import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from math import sqrt
import matplotlib.dates as mdates

from Engine.DataTimeseries import DataTimeseries
from Engine.EquityCurve import EquityCurve
from Engine.PnLDistribution import SkewCurve
from Engine.Utils import dollar_format, percentage_format, number_format, confidence_interval, bars_per_year, p_value_one_tailed, get_max_consecutive_streak, confidence_interval_mean

# equity curves
# trade win distribution, skew analysis
# optimal trade times etc
# monte carlo simulations
# monte carlo permulation test
# hyperparameter stationary testing


class Backtester:

    def __init__(
            self,
            ohlc: np.ndarray,
            indicies: pd.DatetimeIndex,
            signal: np.ndarray,
            timeframe: str,
            tick_value: float,
            contract_multiplier: float,
            initial_equity: float = 100_000,
            slippage_ticks: float = 0,
            commission: float = 0
    ):

        assert len(ohlc) == len(
            signal), "OHLC data and signal length must match."
        assert len(indicies) == len(
            ohlc), "Indicies length must match OHLC data length."
        assert contract_multiplier / \
            4 == tick_value, "Tick value must be one-fourth of contract multiplier for quarter-point ticks."
        assert ohlc.shape[1] == 4, "OHLC data must have 4 columns (Open, High, Low, Close)."

        # DataTimeseries initialization, closing prices, start, end
        self.ohlc: np.ndarray = ohlc
        self.indicies: pd.DatetimeIndex = indicies
        self.timeframe: str = timeframe
        self.close_prices: np.ndarray = self.ohlc[:, 3]
        self.start_date: pd.Timestamp = indicies[0]
        self.end_date: pd.Timestamp = indicies[-1]
        self.duration: pd.Timestamp = self.end_date - self.start_date

        # Backtesting hyperparameter metrics
        self.tick_value: float = tick_value  # dollar equivalent for 1 tick move
        self.contract_multiplier: float = contract_multiplier  # $ equivalent for 1 point
        self.slippage_ticks: float = slippage_ticks
        self.commission: float = commission

        # equity metrics
        self.initial_equity: float = initial_equity
        self.final_equity: float = None

        # Signal metrics
        self.signal: np.ndarray = signal
        self.trade_indicies: np.ndarray = np.insert(
            np.abs(np.diff(self.signal)) / 2, 0, 0.0)

        # Return metrics
        self.trade_pnl: np.ndarray = None  # PnL of each candle
        self.cum_trade_pnl: np.ndarray = None  # cumulative PnL
        self.equity_curve: np.ndarray = None  # dollar equity curve
        self.drawdown_curve: np.ndarray = None  # dollar amounts
        self.transaction_costs: np.ndarray = None  # transaction costs per bar

        self.total_pct_return: float = None  # (PnL ÷ initial equity)
        # Total PnL in dollars (only PnL not initial equity)
        self.total_pnl: float = None
        self.buyhold_pct_return: float = self.calculate_raw_buy_hold_return()
        self.buyhold_pnl: float = self.buyhold_pct_return * self.initial_equity

        self.max_drawdown: float = None
        self.max_drawdown_pct: float = None
        self.max_drawdown_start: pd.Timestamp = None
        self.max_drawdown_end: pd.Timestamp = None
        self.max_drawdown_duration: pd.Timestamp = None

        # winrate metrics
        self.total_trades: int = None
        self.winning_trades: int = None
        self.losing_trades: int = None
        self.win_rate: float = None
        self.average_win: float = None
        self.average_loss: float = None
        self.ratio_avg_win_loss: float = None
        self.max_consec_wins: int = None
        self.max_consec_losses: int = None
        self.largest_win: float = None
        self.largest_loss: float = None
        self.expected_value: float = None

        # Short and long trade metrics
        self.short_trades_won: int = None
        self.long_trades_won: int = None
        self.short_trades_winrate: float = None
        self.long_trades_winrate: float = None

        self.CAGR: float = None
        self.sharpe_ratio: float = None
        self.sortino_ratio: float = None
        self.profit_factor: float = None

        # self.alpha: float = None
        # self.beta: float = None

    def __str__(self):
        mu, lb, ub = confidence_interval(
            self.winning_trades, self.total_trades, 0.05)
        return f'''
        Backtest Results from {self.start_date} to {self.end_date} ({self.duration})
        Equity:     {dollar_format(self.initial_equity)} -> {dollar_format(self.final_equity)}
        Buy & Hold: {dollar_format(self.initial_equity)} -> {dollar_format(self.initial_equity + self.buyhold_pnl)}
        Buy & Hold Return (%): {percentage_format(self.buyhold_pct_return)}
        Strategy Return (%):   {percentage_format(self.total_pct_return)}

        Total Trades:     {number_format(self.total_trades)}
        Win Rate:         {percentage_format(self.win_rate)}
        95% CI:           {percentage_format(lb)} - {percentage_format(ub)}
        Avg. Win:         {dollar_format(self.average_win)}
        Avg. Loss:        {dollar_format(self.average_loss)}
        Ratio Win/Loss:   {number_format(self.ratio_avg_win_loss)}
        Expected Value:   {dollar_format(self.expected_value)}

        Largest Win:      {dollar_format(self.largest_win)}
        Largest Loss:     {dollar_format(self.largest_loss)}
        Max Consec. Wins: {self.max_consec_wins}
        Max Consec. Loss: {self.max_consec_losses}

        Max Drawdown from {self.max_drawdown_start} to {self.max_drawdown_end} ({self.max_drawdown_duration})
        Max Drawdown:     {dollar_format(self.max_drawdown)}
        Max Drawdown (%): {percentage_format(self.max_drawdown_pct)}

        Slippage:   {self.slippage_ticks} ticks
        Commission: {dollar_format(self.commission)}
        Contract Multiplier: {dollar_format(self.contract_multiplier)}
        Transaction Costs:   {dollar_format(self.transaction_costs.sum())}

        CAGR:              {percentage_format(self.CAGR)}
        Sharpe Ratio:      {number_format(self.sharpe_ratio)}
        Sortino Ratio:     {number_format(self.sortino_ratio)}
        Profit Factor:     {number_format(self.profit_factor)}
        '''
        # Short Trades Won:     {self.short_trades_won}
        # Long Trades Won:      {self.long_trades_won}
        # Short Trades Winrate: {percentage_format(self.short_trades_winrate)}
        # Long Trades Winrate:  {percentage_format(self.long_trades_winrate)}

    def full_backtest(self) -> None:
        self.quick_backtest()
        self.calculate_drawdown_metrics()
        self.calculate_winlose_metrics()
        self.calculate_performance_metrics()

    def quick_backtest(self) -> None:
        if len(self.signal) != len(self.close_prices):
            raise ValueError(
                f"Signal length must match the length of close prices. {len(self.signal)} != {len(self.close_prices)}")

        # Handle NaN values introduced by shift
        position = np.roll(self.signal, 1)
        position[0] = 0.0
        # price_diff = np.insert(np.diff(self.close_prices), 0, 0.0)
        price_diff = self.close_prices - self.ohlc[:, 0]  # open to close

        raw_pnl: np.ndarray = position * price_diff * self.contract_multiplier
        commision_cost: np.ndarray = self.trade_indicies * self.commission
        slippage_cost: np.ndarray = self.trade_indicies * self.slippage_ticks * \
            self.tick_value
        self.transaction_costs: np.ndarray = commision_cost + slippage_cost
        self.trade_pnl: np.ndarray = raw_pnl - self.transaction_costs
        self.cum_trade_pnl: np.ndarray = self.trade_pnl.cumsum()
        self.equity_curve: np.ndarray = self.initial_equity + self.cum_trade_pnl

        self.final_equity = self.equity_curve[-1]
        self.total_pnl = self.cum_trade_pnl[-1]
        self.total_pct_return = self.total_pnl / self.initial_equity

    def calculate_raw_buy_hold_return(self) -> float:
        return self.close_prices[-1] / self.close_prices[0] - 1

    def calculate_drawdown_metrics(self):
        equity_max: np.ndarray = np.maximum.accumulate(self.equity_curve)

        self.drawdown_curve = self.equity_curve - equity_max
        self.max_drawdown = self.drawdown_curve.min()  # most negative value
        self.max_drawdown_pct = self.max_drawdown / self.equity_curve.max()

        end_index_location: int = np.argmin(self.drawdown_curve)

        # Find the array index (position) of the preceding peak
        # The peak is the maximum equity *before* the drawdown ended.
        start_index_location: int = np.argmax(
            self.equity_curve[:end_index_location])

        self.max_drawdown_start = self.indicies[start_index_location]
        self.max_drawdown_end = self.indicies[end_index_location]

        self.max_drawdown_duration = self.indicies[end_index_location] - \
            self.indicies[start_index_location]

    def calculate_winlose_metrics(self):
        trade_pnl_series: pd.Series = pd.Series(self.trade_pnl)
        trade_groups: np.ndarray = self.trade_indicies.cumsum()
        trade_returns: pd.Series = trade_pnl_series.groupby(trade_groups).sum()
        trade_returns: np.ndarray = trade_returns.values

        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns < 0]

        self.winning_trades = len(winning_trades)
        self.losing_trades = len(losing_trades)
        self.total_trades = len(trade_returns)
        self.win_rate = self.winning_trades / self.total_trades

        self.average_win = winning_trades.sum() / self.winning_trades
        self.average_loss = losing_trades.sum() / self.losing_trades
        self.ratio_avg_win_loss = self.average_win / -self.average_loss  # flip positive

        # !todo missing calculation for the number of winning long trades vs winning short trades
        # !todo hard todo because you have to pair the signal to the returns

        # overwrite trade_returns to comput max consecutive
        winning_consec = np.where(trade_returns > 0, 1, 0)
        losing_consec = np.where(trade_returns < 0, 1, 0)
        self.max_consec_wins = get_max_consecutive_streak(winning_consec)
        self.max_consec_losses = get_max_consecutive_streak(losing_consec)

        self.largest_win = winning_trades.max()
        self.largest_loss = losing_trades.min()

        # average loss stored as negative
        self.expected_value = (self.average_win * self.win_rate) + \
            (self.average_loss * (1 - self.win_rate))

    def calculate_performance_metrics(self):
        bars_annually = bars_per_year(self.timeframe)

        # CAGR
        years = self.duration.days / 365.25
        self.CAGR = ((self.final_equity / self.initial_equity)
                     ** (1 / years) - 1) if years > 0 else 0.0

        # annualized metrics
        equity = self.equity_curve
        bar_returns = np.diff(equity) / equity[:-1]
        mean_ret = bar_returns.mean()
        std_ret = bar_returns.std()

        # sharpe ratio (annualized)
        if std_ret > 0:
            self.sharpe_ratio = (mean_ret / std_ret) * np.sqrt(bars_annually)
        else:
            self.sharpe_ratio = 0.0

        # sortino (annualized)
        downside = bar_returns[bar_returns < 0]
        downside_std = downside.std()
        if downside_std > 0:
            self.sortino_ratio = (mean_ret / downside_std) * \
                np.sqrt(bars_annually)
        else:
            self.sortino_ratio = 0.0

        # profit factor
        total_gains = self.trade_pnl[self.trade_pnl > 0].sum()
        total_losses = -self.trade_pnl[self.trade_pnl < 0].sum()
        if total_losses > 0:
            self.profit_factor = total_gains / total_losses
        else:
            self.profit_factor = float('inf')

        # self.annualized_return = (1 + mean_ret) ** bars_annually - 1
        # self.annualized_vol = std_ret * np.sqrt(bars_annually)
        # print(self.annualized_return, self.annualized_vol)

    def generate_equity_curve(self) -> EquityCurve:
        return EquityCurve(self.equity_curve, self.drawdown_curve, self.trade_indicies, self.indicies)

    def generate_buy_hold_equity_curve(self) -> EquityCurve:
        shares: float = self.initial_equity / self.close_prices[0]
        equity_curve: np.ndarray = shares * self.close_prices
        drawdown_curve: np.ndarray = equity_curve - \
            np.maximum.accumulate(equity_curve)

        return EquityCurve(equity_curve, drawdown_curve, self.trade_indicies, self.indicies)

    def generate_skew_curve(self) -> SkewCurve:
        return SkewCurve(self.trade_pnl, self.trade_indicies, self.initial_equity)

    def generate_trade_time_distribution(self) -> None:
        trade_groups = self.trade_indicies.cumsum()
        trade_returns: pd.Series = pd.Series(
            self.trade_pnl).groupby(trade_groups).sum()
        trade_entry_times: pd.DatetimeIndex = self.indicies[self.trade_indicies.astype(
            bool)]

        trade_returns = trade_returns.iloc[1:]  # remove first trade (0.0)

        assert len(trade_returns) == len(trade_entry_times)

        entry_minutes = trade_entry_times.hour * \
            60 + trade_entry_times.minute

        minute_pnl_series = pd.Series(
            trade_returns.values, index=entry_minutes)

        # Sum the PNL for all trades that happened in the same minute of the day
        total_returns_per_minute = minute_pnl_series.groupby(level=0).sum()

        # --- Prepare for Bar Plotting ---
        # The x-values are the minutes of the day (0 to 1439)
        x_values = total_returns_per_minute.index.values
        # The y-values are the total PNL for that minute
        y_values = total_returns_per_minute.values

        # Determine bar colors: green for profit, red for loss
        colors = ['green' if pnl >= 0 else 'red' for pnl in y_values]

        # --- Plotting ---
        plt.figure(figsize=(12, 6))

        # Use plt.bar to plot the total PNL for each minute bin
        # 'width=1' ensures the bar covers a single minute
        width = (self.indicies[1] - self.indicies[0]).seconds // 60
        plt.bar(x_values, y_values, width=width,
                color=colors, label='Total PNL')

        # Add a horizontal line at PNL = 0 for reference
        plt.axhline(0, color='black', linestyle='--', linewidth=1)

        # Set x-axis ticks to display hours (0, 3, 6, ..., 24)
        hour_ticks = range(0, 25, 3)
        plt.xticks([h * 60 for h in hour_ticks],
                   [f'{h:02d}:00' for h in hour_ticks])

        plt.xlabel(f'Time of Day ({self.indicies.tzinfo})')
        plt.ylabel('Total PNL / Trade Returns per Minute')
        plt.title('Total Trade Returns (PNL) by Entry Time')
        plt.show()

    def monte_carlo_pnl_resampling_test(self, iterations: int = 1000, pct_of_trades: float = 1):
        trade_groups = self.trade_indicies.cumsum()
        trade_returns: np.ndarray = pd.Series(self.trade_pnl).groupby(
            trade_groups).sum().values

        n_trades = len(trade_returns)
        step = int(1 / pct_of_trades)
        x = np.arange(0, n_trades, step)

        # Original strategy
        original_equity = trade_returns.cumsum()
        original_drawdown = (
            original_equity - np.maximum.accumulate(original_equity)).min()

        # Initialize best and worst trackers
        best_equity = None
        best_drawdown = -np.inf
        worst_equity = None
        worst_drawdown = 0  # negative drawdown
        drawdowns = []
        ruin_simulations = 0

        # plotting
        curves = []

        for _ in range(iterations):
            shuffled = np.random.permutation(trade_returns)
            equity_curve = shuffled.cumsum()
            drawdown = (equity_curve -
                        np.maximum.accumulate(equity_curve)).min()
            drawdowns.append(drawdown)

            # risk of ruin calculation
            drawdown_pct = abs(drawdown / equity_curve.max())
            if drawdown_pct > 0.5:
                ruin_simulations += 1

            # Update worst
            if drawdown < worst_drawdown:
                worst_drawdown = drawdown
                worst_equity = equity_curve.copy()

            # Update best
            if drawdown > best_drawdown:
                best_drawdown = drawdown
                best_equity = equity_curve.copy()

            curves.append(equity_curve[x])

        drawdowns = np.array(drawdowns)

        p_worst_drawdown = p_value_one_tailed(
            drawdowns.mean(), drawdowns.std(), worst_drawdown, tail='left')

        p_original_drawdown = p_value_one_tailed(
            drawdowns.mean(), drawdowns.std(), original_drawdown, tail='left')

        fig, ax = plt.subplots(figsize=(12, 6))

        line_segments = [np.column_stack((x, c)) for c in curves]
        lc = LineCollection(line_segments, color='gray',
                            linewidth=0.8, alpha=0.5)
        ax.add_collection(lc)
        all_data = np.concatenate(curves)
        min_y = all_data.min()
        max_y = all_data.max()
        padding = (max_y - min_y) * 0.02
        ax.set_ylim(min_y - padding, max_y + padding)

        # Plot original strategy in black
        ax.plot(x, original_equity[x], color='black',
                linewidth=1.5, label=f'Original Drawdown: {dollar_format(original_drawdown)} ({p_original_drawdown})')

        # Plot worst drawdown in red
        ax.plot(x, worst_equity[x], color='red',
                linewidth=1.5, label=f'Worst Drawdown: {dollar_format(worst_drawdown)} ({p_worst_drawdown})')

        # Plot best drawdown in blue
        ax.plot(x, best_equity[x], color='blue',
                linewidth=1.5, label=f'Best Drawdown: {dollar_format(best_drawdown)}')

        # 3. Create the Title String
        title_text = (
            f"Monte Carlo PnL Resampling Test\n"
            f"Avg. Drawdown: {dollar_format(drawdowns.mean())}\n"
            f"Risk of Ruin (50%): {percentage_format(ruin_simulations / iterations)}"
        )
        plt.title(title_text)
        plt.xlabel("Trades")
        plt.ylabel("Returns")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.legend()
        plt.show()

    def monte_carlo_permutation_test(self, iterations: int = 1000, pct_of_trades: float = 1, title: str = ''):
        null_hypothesis = self.total_pnl
        permutation_returns = []
        pct_of_trades = int(1 / pct_of_trades)

        # create plotting mask
        mask = np.zeros(len(self.equity_curve))
        mask[::pct_of_trades] = 1
        mask = mask.astype(bool)

        # curves
        x = self.indicies[mask]
        x = mdates.date2num(x)
        curves = []

        # 1x2
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        ax_left, ax_right = axes

        for i in range(iterations):
            print(i)
            permuted_ohlc: np.ndarray = DataTimeseries.generate_permutation(
                self.ohlc, self.indicies)
            bt = Backtester(
                permuted_ohlc,
                self.indicies,
                self.signal,
                self.timeframe,
                self.tick_value,
                self.contract_multiplier,
                self.initial_equity,
                self.slippage_ticks,
                self.commission
            )
            bt.quick_backtest()

            permutation_returns.append(bt.total_pnl)
            curves.append(bt.equity_curve[mask])

        # plot all equity curves
        lines = [np.column_stack((x, c)) for c in curves]
        lc = LineCollection(lines, color='gray', linewidth=0.8, alpha=0.5)
        ax_left.add_collection(lc)

        # plot original equity curve
        ax_left.plot(self.indicies[mask], self.equity_curve[mask],
                     linewidth=1, color='blue')
        all_data = np.concatenate(curves)
        min_y = all_data.min()
        max_y = all_data.max()
        padding = (max_y - min_y) * 0.02
        ax_left.set_ylim(min_y - padding, max_y + padding)

        ax_left.set_title(
            f"Monte Carlo {title} Permutation Test (showing every {pct_of_trades} bars)")
        ax_left.set_xlabel("Time")
        ax_left.set_ylabel("Returns")
        ax_left.grid(True, alpha=0.25)
        ax_left.legend()

        # perform P-test
        returns_np = np.array(permutation_returns)
        mu = returns_np.mean()
        std = returns_np.std()

        p = p_value_one_tailed(mu, std, null_hypothesis, tail='right')

        ax_right.hist(returns_np, bins=50, alpha=0.7, edgecolor='blue')
        ax_right.axvline(null_hypothesis, color='blue', linewidth=2,
                         label=f'Actual = {null_hypothesis:.2f}')
        ax_right.axvline(mu, color='red', linewidth=1.5,
                         linestyle='--', label=f'Mean = {mu:.2f}')

        ax_right.set_title(f"Permutation Distribution\np = {p}")
        ax_right.set_xlabel("PnL")
        ax_right.set_ylabel("Frequency")
        ax_right.grid(True, alpha=0.25)
        ax_right.legend()

        plt.tight_layout()
        plt.show()
