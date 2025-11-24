import pandas as pd
import numpy as np

from Engine.DataTimeseries import DataTimeseries
from Engine.EquityCurve import EquityCurve
from Engine.SkewCurve import SkewCurve
from Engine.Utils import dollar_format, percentage_format

# equity curves
# trade win distribution, skew analysis
# optimal trade times etc
# monte carlo simulations
# monte carlo permulation test
# hyperparameter stationary testing


class Backtester:

    def __init__(self, data_timeseries: DataTimeseries, signal: pd.Series, tick_value: float, contract_multiplier: float, initial_equity: float = 100_000, slippage_ticks: float = 0, commission: float = 0):
        # DataTimeseries initialization, closing prices, start, end
        self.data_timeseries: DataTimeseries = data_timeseries
        self.close_prices: pd.Series = self.data_timeseries.get_close_prices()
        self.start_date: pd.Timestamp = self.data_timeseries.get_start_date()
        self.end_date: pd.Timestamp = self.data_timeseries.get_end_date()
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

        self.consecutive_losses: int = None

        self.quick_backtest()
        self.calculate_drawdown_metrics()

        # self.equity_curve: EquityCurve = self.generate_equity_curve()
        # self.bh_equity_curve: EquityCurve = self.generate_buy_hold_equity_curve()

        self.sharpe_ratio: float = None
        self.sortino_ratio: float = None
        self.profit_factor: float = None

        self.alpha: float = None
        self.beta: float = None

        self.trades: list = []
        self.total_trades: int = len(self.trades)
        self.win_rate: float = None

    def __str__(self):
        return f'''
        Backtest Results from {self.start_date} to {self.end_date} ({self.duration})
        Equity:     {dollar_format(self.initial_equity)} -> {dollar_format(self.end_equity)}
        Buy & Hold: {dollar_format(self.initial_equity)} -> {dollar_format(self.buyhold_equity_pnl)}
        Buy & Hold Return (%): {percentage_format(self.buyhold_pct_return)}
        Strategy Return (%):   {percentage_format(self.total_pct_return)}
        
        Max Drawdown from {self.max_drawdown_start} to {self.max_drawdown_end} ({self.max_drawdown_duration})
        Max Drawdown:     {dollar_format(self.max_drawdown)}
        Max Drawdown (%): {percentage_format(self.max_drawdown_pct)}
    
        Slippage:   {self.slippage_ticks} ticks
        Commission: {dollar_format(self.commission)} 
        Contract Multiplier: {dollar_format(self.contract_multiplier)}
        '''

    def full_backtest(self, signal: pd.Series) -> None:
        pass

    def quick_backtest(self) -> None:
        if len(self.signal) != len(self.close_prices):
            raise ValueError(
                f"Signal length must match the length of close prices. {len(self.signal)} != {len(self.close_prices)}")

        # Handle NaN values introduced by shift
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
        self.total_pct_return = self.total_pnl / self.initial_equity

    def calculate_raw_buy_hold_return(self) -> float:
        return self.close_prices.iloc[-1] / self.close_prices.iloc[0] - 1

    # def calculate_buy_hold_return(self) -> float:
    #     shares: float = self.initial_equity / self.close_prices.iloc[0]
    #     final_equity: float = shares * self.close_prices.iloc[-1]
    #     return final_equity / self.initial_equity - 1

    def generate_equity_curve(self) -> EquityCurve:
        return EquityCurve(self.equity_curve, self.trade_indicies)

    def generate_buy_hold_equity_curve(self) -> EquityCurve:
        shares: float = self.initial_equity / self.close_prices.iloc[0]
        equity_curve: pd.Series = shares * self.close_prices

        return EquityCurve(equity_curve, self.trade_indicies)

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
