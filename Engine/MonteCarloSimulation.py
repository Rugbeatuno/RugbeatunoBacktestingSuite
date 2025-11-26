import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Engine.Utils import dollar_format


def pnl_resampling_test(trade_dollar_returns: np.ndarray, iterations: int = 1000, pct_of_trades: float = 1):
    n_trades = len(trade_dollar_returns)
    step = int(1 / pct_of_trades)
    x = np.arange(0, n_trades, step)

    plt.figure(figsize=(12, 6))

    # Original strategy
    original_equity = np.cumsum(trade_dollar_returns)
    original_drawdown = np.min(
        original_equity - np.maximum.accumulate(original_equity))

    # Initialize best and worst trackers
    best_equity = None
    best_drawdown = -np.inf
    worst_equity = None
    worst_drawdown = 0  # negative drawdown

    for _ in range(iterations):
        shuffled = np.random.permutation(trade_dollar_returns)
        equity_curve = np.cumsum(shuffled)
        drawdown = np.min(equity_curve - np.maximum.accumulate(equity_curve))

        # Update worst
        if drawdown < worst_drawdown:
            worst_drawdown = drawdown
            worst_equity = equity_curve.copy()

        # Update best
        if drawdown > best_drawdown:
            best_drawdown = drawdown
            best_equity = equity_curve.copy()

        plt.plot(x, equity_curve[x], linewidth=0.8, alpha=0.5, color='gray')

    # Plot original strategy in black
    plt.plot(x, original_equity[x], color='black',
             linewidth=1.5, label=f'Original Drawdown: {dollar_format(original_drawdown)}')

    # Plot worst drawdown in red
    plt.plot(x, worst_equity[x], color='red',
             linewidth=1.5, label=f'Worst Drawdown: {dollar_format(worst_drawdown)}')

    # Plot best drawdown in blue
    plt.plot(x, best_equity[x], color='blue',
             linewidth=1.5, label=f'Best Drawdown: {dollar_format(best_drawdown)}')

    plt.title(f"Price Series (showing every {step} bars)")
    plt.xlabel("Trades")
    plt.ylabel("Returns")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.legend()
    plt.show()
