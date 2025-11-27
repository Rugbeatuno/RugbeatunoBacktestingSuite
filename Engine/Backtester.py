import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Engine.DataTimeseries import DataTimeseries
from Engine.EquityCurve import EquityCurve
from Engine.PnLDistribution import SkewCurve
from Engine.Utils import dollar_format, percentage_format, number_format, confidence_interval, bars_per_year, p_value_one_tailed
from Engine.MonteCarloSimulation import pnl_resampling_test

# equity curves
# trade win distribution, skew analysis
# optimal trade times etc
# monte carlo simulations
# monte carlo permulation test
# hyperparameter stationary testing


class Backtester:

    def __init__(self, price_timeseries: pd.DataFrame, timeframe: str, signal: pd.Series, tick_value: float, contract_multiplier: float, initial_equity: float = 100_000, slippage_ticks: float = 0, commission: float = 0):
        # DataTimeseries initialization, closing prices, start, end
        self.price_timeseries: pd.DataFrame = price_timeseries
        self.timeframe: str = timeframe
        self.close_prices: pd.Series = self.price_timeseries['Close']
        self.start_date: pd.Timestamp = self.price_timeseries.index[0]
        self.end_date: pd.Timestamp = self.price_timeseries.index[-1]
        self.duration: pd.Timedelta = self.end_date - self.start_date

        # Backtesting hyperparameter metrics
        self.tick_value: float = tick_value  # dollar equivalent for 1 tick move
        self.contract_multiplier: float = contract_multiplier  # $ equivalent for 1 point
        self.slippage_ticks: float = slippage_ticks
        self.commission: float = commission

        # equity metrics
        self.initial_equity: float = initial_equity
        self.end_equity: float = None

        # Signal metrics
        self.signal: pd.Series = signal
        self.trade_indicies: pd.Series = self.signal.diff().abs().fillna(0)

        # Return metrics
        self.trade_pnl: pd.Series = None  # PnL of each candle
        self.cum_trade_pnl: pd.Series = None  # cumulative PnL
        self.equity_curve: pd.Series = None  # dollar equity curve
        self.drawdown_curve: pd.Series = None  # dollar amounts
        self.transaction_costs: pd.Series = None  # transaction costs per bar

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

        self.alpha: float = None
        self.beta: float = None

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
        index = self.close_prices.index
        position = self.signal.shift(1).fillna(0)
        price_diff = self.close_prices.diff().fillna(0)

        raw_pnl: pd.Series = position * price_diff * self.contract_multiplier
        commision_cost: pd.Series = self.trade_indicies * self.commission
        slippage_cost: pd.Series = self.trade_indicies * self.slippage_ticks * \
            self.tick_value * self.contract_multiplier

        self.transaction_costs: pd.Series = commision_cost + slippage_cost
        self.trade_pnl: pd.Series = raw_pnl - self.transaction_costs
        self.cum_trade_pnl: pd.Series = self.trade_pnl.cumsum()
        self.equity_curve: pd.Series = self.initial_equity + self.cum_trade_pnl

        self.end_equity = self.equity_curve.iloc[-1]
        self.total_pnl = self.initial_equity + self.cum_trade_pnl.iloc[-1]
        self.total_pct_return = self.total_pnl / self.initial_equity - 1

    def calculate_raw_buy_hold_return(self) -> float:
        return self.close_prices.iloc[-1] / self.close_prices.iloc[0] - 1

    def generate_equity_curve(self) -> EquityCurve:
        return EquityCurve(self.equity_curve, self.drawdown_curve, self.trade_indicies)

    def generate_buy_hold_equity_curve(self) -> EquityCurve:
        shares: float = self.initial_equity / self.close_prices.iloc[0]
        equity_curve: pd.Series = shares * self.close_prices
        drawdown_curve: pd.Series = equity_curve - equity_curve.cummax()

        return EquityCurve(equity_curve, drawdown_curve, self.trade_indicies)

    def generate_skew_curve(self) -> SkewCurve:
        return SkewCurve(self.trade_pnl, self.trade_indicies, self.initial_equity)

    def calculate_drawdown_metrics(self):
        equity_max: pd.Series = self.equity_curve.cummax()

        self.drawdown_curve = self.equity_curve - equity_max
        self.max_drawdown = self.drawdown_curve.min()  # most negative value
        self.max_drawdown_pct = self.max_drawdown / self.equity_curve.max()

        self.max_drawdown_end = self.drawdown_curve.idxmin()
        self.max_drawdown_start = self.equity_curve[:self.max_drawdown_end].idxmax(
        )
        self.max_drawdown_duration = self.max_drawdown_end - self.max_drawdown_start

    def calculate_winlose_metrics(self):
        trade_groups = self.trade_indicies.cumsum()
        trade_returns = self.trade_pnl.groupby(trade_groups).sum()
        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns < 0]

        self.winning_trades = len(winning_trades)
        self.losing_trades = len(losing_trades)
        self.total_trades = len(trade_returns)
        self.win_rate = self.winning_trades / self.total_trades

        self.average_win = winning_trades.sum() / self.winning_trades
        self.average_loss = losing_trades.sum() / self.losing_trades
        self.ratio_avg_win_loss = self.average_win / \
            self.average_loss * -1  # flip positive

        # trade_groups = self.trade_indicies.cumsum()

        # first index of each trade group
        # trade_start_index = trade_groups[trade_groups.diff().fillna(1) == 1]

        # # the direction taken at the start of each trade
        # trade_directions = self.signal.loc[trade_start_index.index]

        # # trade_returns has index = trade group numbers (0,1,2,3,...)
        # # so map group_number → direction
        # trade_directions.index = trade_start_index.values   # assign group numbers as index
        # trade_directions = trade_directions.reindex(
        #     trade_returns.index)  # align to trade_returns

        # long_returns = trade_returns[trade_directions > 0]
        # short_returns = trade_returns[trade_directions < 0]

        # self.long_trades_won = (long_returns > 0).sum()
        # self.short_trades_won = (short_returns > 0).sum()

        # self.long_trades_winrate = self.long_trades_won / \
        #     len(long_returns) if len(long_returns) > 0 else 0
        # self.short_trades_winrate = self.short_trades_won / \
        #     len(short_returns) if len(short_returns) > 0 else 0

        # overwrite trade_returns to comput max consecutive
        trade_returns[trade_returns > 0] = 1
        trade_returns[trade_returns < 0] = -1
        self.max_consec_wins = trade_returns.groupby(
            (trade_returns != 1).cumsum()).transform('size')[trade_returns == 1].max()
        self.max_consec_losses = trade_returns.groupby(
            (trade_returns != -1).cumsum()).transform('size')[trade_returns == -1].max()

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
        per_trade_pnl: np.ndarray = self.trade_pnl.groupby(
            trade_groups).sum().values

        pnl_resampling_test(per_trade_pnl, iterations,
                            pct_of_trades=pct_of_trades)

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
            permuted_series: pd.Series = DataTimeseries.generate_permuted_price_series(
                self.price_timeseries)
            permuted_dataframe = permuted_series.to_frame(name='Close')
            bt = Backtester(
                permuted_dataframe,
                self.timeframe,
                self.signal,
                self.tick_value,
                self.contract_multiplier,
                self.initial_equity,
                self.slippage_ticks,
                self.commission
            )
            bt.quick_backtest()

            permutation_returns.append(bt.total_pnl)

            equity_curve = bt.equity_curve[mask]
            ax_left.plot(
                equity_curve.index,
                equity_curve.values,
                linewidth=0.8,
                color='gray'
            )

        # plot original equity curve
        ax_left.plot(self.equity_curve[mask].index, self.equity_curve[mask].values,
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
