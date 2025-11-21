
class Optimizer:
    def __init__(self):
        self.best_parameters: dict = {}
        self.best_performance: float = None
        self.evaluation_metrics: dict = {}
