import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        self.end_equity: float = None

        # Signal metrics
        self.signal: np.ndarray = signal
        self.trade_indicies: np.ndarray = np.insert(
            np.abs(np.diff(self.signal)), 0, 0.0)

        # Return metrics
        self.trade_pnl: np.ndarray = None  # PnL of each candle
        self.cum_trade_pnl: np.ndarray = None  # cumulative PnL
        self.equity_curve: np.ndarray = None  # dollar equity curve
        self.drawdown_curve: np.ndarray = None  # dollar amounts
        self.transaction_costs: np.ndarray = None  # transaction costs per bar

        self.total_pct_return: float = None  # (PnL ÷ initial equity)
        self.total_pnl: float = None  # Total PnL in dollars
        self.buyhold_pct_return: float = self.calculate_raw_buy_hold_return()
        self.buyhold_equity_pnl: float = self.buyhold_pct_return * self.initial_equity

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
        Equity:     {dollar_format(self.initial_equity)} -> {dollar_format(self.end_equity)}
        Buy & Hold: {dollar_format(self.initial_equity)} -> {dollar_format(self.buyhold_equity_pnl)}
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

        Performance Metrics:
        CAGR:              {percentage_format(self.CAGR)}
        Sharpe Ratio:      {number_format(self.sharpe_ratio)}
        Sortino Ratio:     {number_format(self.sortino_ratio)}
        Profit Factor:     {number_format(self.profit_factor)}
        '''
        # Short Trades Won:     {self.short_trades_won}
        # Long Trades Won:      {self.long_trades_won}
        # Short Trades Winrate: {percentage_format(self.short_trades_winrate)}
        # Long Trades Winrate:  {percentage_format(self.long_trades_winrate)}

    def vectorized_backtest(self, signal_matrix: np.ndarray) -> np.ndarray:
        """
        Runs a fully vectorized backtest on a matrix of signals.

        Parameters
        ----------
        signal_matrix : np.ndarray
            A 2D array of shape (K, N), where K is the number of
            parameter combinations (strategies) and N is the number of bars.

        Returns
        -------
        pd.DataFrame
            A DataFrame with K rows, where each row contains the
            performance metrics for a single strategy.
        """

        # K = number of strategies (param combinations)
        # N = number of bars
        K, N = signal_matrix.shape

        if N != len(self.close_prices):
            raise ValueError(f"Signal matrix bar dimension ({N}) does not match "
                             f"Backtester's close prices dimension ({len(self.close_prices)}).")

        # --- 1. Vectorized PnL Calculation ---

        # Price diff is a 1D array (N,)
        price_diff = np.insert(np.diff(self.close_prices), 0, 0.0)

        # Position matrix (K, N)
        # Roll along the time axis (axis=1)
        position_matrix = np.roll(signal_matrix, 1, axis=1)
        position_matrix[:, 0] = 0.0

        # PnL Matrix (K, N)
        # Broadcasting: (K, N) * (N,) -> (K, N)
        raw_pnl_matrix = position_matrix * price_diff * self.contract_multiplier

        # --- 2. Vectorized Costs Calculation ---

        # Trade Indicies Matrix (K, N)
        trade_indicies_matrix = np.insert(
            np.abs(np.diff(signal_matrix, axis=1)), 0, 0.0, axis=1)

        commission_cost = trade_indicies_matrix * self.commission
        slippage_cost = (trade_indicies_matrix * self.slippage_ticks *
                         self.tick_value * self.contract_multiplier)

        transaction_costs = commission_cost + slippage_cost

        trade_pnl_matrix = raw_pnl_matrix - transaction_costs

        # --- 3. Vectorized Equity & Drawdown ---

        # Equity Curve Matrix (K, N)
        # Cumsum along the time axis (axis=1)
        cum_trade_pnl_matrix = np.cumsum(trade_pnl_matrix, axis=1)
        equity_curve_matrix = self.initial_equity + cum_trade_pnl_matrix

        # Total PnL (K,)
        total_pnl = equity_curve_matrix[:, -1] - self.initial_equity

        # Max Drawdown (K,)
        # Use np.maximum.accumulate along the time axis (axis=1)
        # equity_max = np.maximum.accumulate(equity_curve_matrix, axis=1)
        # drawdown_curve = equity_curve_matrix - equity_max
        # max_drawdown = np.min(drawdown_curve, axis=1)  # (K,)

        # # --- 4. Vectorized Performance Metrics ---

        # # Sharpe Ratio (K,)
        # bar_returns = trade_pnl_matrix / self.initial_equity
        # mean_ret = np.mean(bar_returns, axis=1)
        # std_ret = np.std(bar_returns, axis=1)

        # # Handle division by zero
        # sharpe_ratio = np.divide(mean_ret, std_ret,
        #                          out=np.zeros(K, dtype=float), where=std_ret != 0)

        # bars_year = bars_per_year(self.timeframe)
        # sharpe_ratio *= np.sqrt(bars_year)

        # # Profit Factor (K,)
        # total_gains = np.sum(
        #     np.where(trade_pnl_matrix > 0, trade_pnl_matrix, 0), axis=1)
        # total_losses = -np.sum(np.where(trade_pnl_matrix <
        #                        0, trade_pnl_matrix, 0), axis=1)

        # profit_factor = np.divide(total_gains, total_losses,
        #                           out=np.full(K, np.inf), where=total_losses != 0)

        # --- 5. Assemble Results ---
        # results = {
        #     'TotalPnL': total_pnl,
        #     'Sharpe': sharpe_ratio,
        #     'MaxDrawdown': max_drawdown,
        #     'ProfitFactor': profit_factor
        # }
        return total_pnl

    def full_backtest(self) -> None:
        self.quick_backtest()
        self.calculate_drawdown_metrics()
        self.calculate_winlose_metrics()
        self.calculate_performance_metrics()

    # convert all major calculations into numpy then back to pandas

    def quick_backtest(self) -> None:
        if len(self.signal) != len(self.close_prices):
            raise ValueError(
                f"Signal length must match the length of close prices. {len(self.signal)} != {len(self.close_prices)}")

        # Handle NaN values introduced by shift
        position = np.roll(self.signal, 1)
        position[0] = 0.0
        price_diff = np.insert(np.diff(self.close_prices), 0, 0.0)

        raw_pnl: np.ndarray = position * price_diff * self.contract_multiplier
        commision_cost: np.ndarray = self.trade_indicies * self.commission
        slippage_cost: np.ndarray = self.trade_indicies * self.slippage_ticks * \
            self.tick_value * self.contract_multiplier

        self.transaction_costs: np.ndarray = commision_cost + slippage_cost
        self.trade_pnl: np.ndarray = raw_pnl - self.transaction_costs
        self.cum_trade_pnl: np.ndarray = self.trade_pnl.cumsum()
        self.equity_curve: np.ndarray = self.initial_equity + self.cum_trade_pnl

        self.end_equity = self.equity_curve[-1]
        self.total_pnl = self.initial_equity + self.cum_trade_pnl[-1]
        self.total_pct_return = self.total_pnl / self.initial_equity - 1

    def calculate_raw_buy_hold_return(self) -> float:
        return self.close_prices[-1] / self.close_prices[0] - 1

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
        self.max_consec_wins = get_max_consecutive_streak(winning_trades)
        self.max_consec_losses = get_max_consecutive_streak(losing_trades)

        self.largest_win = winning_trades.max()
        self.largest_loss = losing_trades.min()

        # average loss stored as negative
        self.expected_value = (self.average_win * self.win_rate) + \
            (self.average_loss * (1 - self.win_rate))

    def calculate_performance_metrics(self):
        """Calculate performance metrics using correct timeframe-based annualization."""

        # ----------------------------------------------------
        # 1. Annualization factor from timeframe (bars/year)
        # ----------------------------------------------------
        # USE YOUR DATASERIES TIMEFRAME
        bars_year = bars_per_year(self.timeframe)
        # ----------------------------------------------------
        # 2. CAGR (your old version is correct)
        # ----------------------------------------------------
        years = self.duration.days / 365.25
        self.CAGR = ((self.end_equity / self.initial_equity)
                     ** (1 / years) - 1) if years > 0 else 0.0
        # ----------------------------------------------------
        # 3. Sharpe Ratio (correct annualization)
        # ----------------------------------------------------
        # per-trade or per-bar? fix below
        bar_returns = self.trade_pnl / self.initial_equity

        mean_ret = bar_returns.mean()
        std_ret = bar_returns.std()

        self.sharpe_ratio = (mean_ret / std_ret) * \
            np.sqrt(bars_year) if std_ret > 0 else 0.0

        # ----------------------------------------------------
        # 4. Sortino Ratio (correct annualization)
        # ----------------------------------------------------
        downside = bar_returns[bar_returns < 0]
        downside_std = downside.std()

        self.sortino_ratio = (mean_ret / downside_std) * \
            np.sqrt(bars_year) if downside_std > 0 else 0.0

        # ----------------------------------------------------
        # 5. Profit Factor (your version was correct)
        # ----------------------------------------------------
        total_gains = self.trade_pnl[self.trade_pnl > 0].sum()
        total_losses = -self.trade_pnl[self.trade_pnl < 0].sum()

        self.profit_factor = total_gains / \
            total_losses if total_losses > 0 else float('inf')

    def monte_carlo_pnl_resampling_test(self, iterations: int = 1000, pct_of_trades: float = 1):
        trade_groups = self.trade_indicies.cumsum()
        trade_returns: np.ndarray = pd.Series(self.trade_pnl).groupby(
            trade_groups).sum().values

        n_trades = len(trade_returns)
        step = int(1 / pct_of_trades)
        x = np.arange(0, n_trades, step)

        plt.figure(figsize=(12, 6))

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

        for _ in range(iterations):
            shuffled = np.random.permutation(trade_returns)
            equity_curve = shuffled.cumsum()
            drawdown = (equity_curve -
                        np.maximum.accumulate(equity_curve)).min()
            drawdowns.append(drawdown)

            # Update worst
            if drawdown < worst_drawdown:
                worst_drawdown = drawdown
                worst_equity = equity_curve.copy()

            # Update best
            if drawdown > best_drawdown:
                best_drawdown = drawdown
                best_equity = equity_curve.copy()

            plt.plot(x, equity_curve[x], linewidth=0.8,
                     alpha=0.5, color='gray')

        drawdowns = np.array(drawdowns)

        p_worst_drawdown = p_value_one_tailed(
            drawdowns.mean(), drawdowns.std(), worst_drawdown, tail='left')

        p_original_drawdown = p_value_one_tailed(
            drawdowns.mean(), drawdowns.std(), original_drawdown, tail='left')

        # Calculate the 99% Confidence Interval for the MEAN maximum drawdown
        # alpha = 0.01 for 99% CI
        _, lb, ub = confidence_interval_mean(
            drawdowns.mean(), drawdowns.std(), iterations, 0.01)

        # 2. Format Statistical Results
        ci_range = f"({dollar_format(lb)} to {dollar_format(ub)})"
        confidence_level = (1 - 0.01) * 100

        # Plot original strategy in black
        plt.plot(x, original_equity[x], color='black',
                 linewidth=1.5, label=f'Original Drawdown: {dollar_format(original_drawdown)} ({p_original_drawdown})')

        # Plot worst drawdown in red
        plt.plot(x, worst_equity[x], color='red',
                 linewidth=1.5, label=f'Worst Drawdown: {dollar_format(worst_drawdown)} ({p_worst_drawdown})')

        # Plot best drawdown in blue
        plt.plot(x, best_equity[x], color='blue',
                 linewidth=1.5, label=f'Best Drawdown: {dollar_format(best_drawdown)}')

        # 3. Create the Title String
        title_text = (
            f"Monte Carlo PnL Resampling Test\n"
            f"Avg. Drawdown: {dollar_format(drawdowns.mean())}\n"
            f"{confidence_level:.0f}% CI: {ci_range}"
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

        # 1x2
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        ax_left, ax_right = axes

        for i in range(iterations):
            print(i)
            permuted_ohlc: np.ndarray = DataTimeseries.permute_cme_ohlc(
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

            ax_left.plot(
                bt.indicies[mask],
                bt.equity_curve[mask],
                linewidth=0.8,
                color='gray'
            )

        # plot original equity curve
        ax_left.plot(self.indicies[mask], self.equity_curve[mask],
                     linewidth=1, color='blue')

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
