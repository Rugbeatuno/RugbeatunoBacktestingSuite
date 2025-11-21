
class EquityCurve:
    def __init__(self):
        self.dates: list = []
        self.equity_values: list = []
        self.returns: list = []
        self.cumulative_returns: list = []
        self.drawdowns: list = []
        self.max_drawdown: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.sortino_ratio: float = 0.0
