import pandas as pd


class Trade:
    def __init__(self):
        self.entry_date: pd.Timestamp = None
        self.exit_date: pd.Timestamp = None
        self.entry_price: float = None
        self.exit_price: float = None
        self.quantity: float = None
        self.direction: str = None  # "long" or "short"
        self.pnl: float = None
        self.return_percentage: float = None
        self.max_drawdown: float = None
        self.max_runup: float = None
        self.holding_period: pd.Timedelta = None
        self.fees: float = None
        self.slippage: float = None

    def calculate_pnl(self):
        pass

    def calculate_metrics(self):
        pass
