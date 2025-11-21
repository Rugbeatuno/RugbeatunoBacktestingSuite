import pandas as pd


class BacktestResults:
    # equity curves
    # trade win distribution, skew analysis
    # optimal trade times etc
    # monte carlo simulations
    # monte carlo permulation test
    # hyperparameter stationary testing

    def __init__(self):
        self.start_date: pd.Timestamp = None
        self.end_date: pd.Timestamp = None
        self.duration: pd.Timedelta = pd.Timedelta(
            self.start_date, self.end_date)

        self.start_equity: float = None
        self.end_equity: float = None
        self.percent_return: float = None
        self.log_return: float = None

        self.sharpe_ratio: float = None
        self.sortino_ratio: float = None
        self.profit_factor: float = None

        self.alpha: float = None
        self.beta: float = None

        self.trades: list = []
        self.total_trades: int = len(self.trades)
        self.trade_win_percentage: float = None
        self.trade_lose_percentage: float = None

    def generate_equity_curve(self):
        pass
