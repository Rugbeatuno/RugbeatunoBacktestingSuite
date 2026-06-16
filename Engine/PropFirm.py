import numpy as np

class PropFirm:

    def __init__(
        self,
        name,
        # Challenge params
        ch_initial_equity,
        ch_profit_target,
        ch_max_loss,
        ch_daily_loss_limit,
        ch_drawdown_type,        # 'eod' or 'trailing_intraday'
        ch_consistency_pct,
        ch_max_contracts_mini,
        ch_max_contracts_micro,
        ch_time_limit_days,
        ch_min_trading_days,
        ch_cost,
        # Funded params
        fu_activation_fee,
        fu_max_loss,
        fu_daily_loss,
        fu_consistency_pct,
        fu_max_contracts_mini,
        fu_max_contracts_micro,
        fu_min_profit_days,
        fu_min_profit_per_day,
        fu_days_to_payout,
        fu_payout_split,
        fu_payout_min_balance,
    ):
        self.name = name

        # Challenge
        self.ch_initial_equity = ch_initial_equity
        self.ch_profit_target = ch_profit_target
        self.ch_max_loss = ch_max_loss
        self.ch_daily_loss_limit = ch_daily_loss_limit
        self.ch_drawdown_type = ch_drawdown_type
        self.ch_consistency_pct = ch_consistency_pct
        self.ch_max_contracts_mini = ch_max_contracts_mini
        self.ch_max_contracts_micro = ch_max_contracts_micro
        self.ch_time_limit_days = ch_time_limit_days
        self.ch_min_trading_days = ch_min_trading_days
        self.ch_cost = ch_cost

        # Funded
        self.fu_activation_fee = fu_activation_fee
        self.fu_max_loss = fu_max_loss
        self.fu_daily_loss = fu_daily_loss
        self.fu_consistency_pct = fu_consistency_pct
        self.fu_max_contracts_mini = fu_max_contracts_mini
        self.fu_max_contracts_micro = fu_max_contracts_micro
        self.fu_min_profit_days = fu_min_profit_days
        self.fu_min_profit_per_day = fu_min_profit_per_day
        self.fu_days_to_payout = fu_days_to_payout
        self.fu_payout_split = fu_payout_split
        self.fu_payout_min_balance = fu_payout_min_balance


class PropFirmSimulator:

    def __init__(self, backtest, propfirm, num_simulations=10000):
        self.backtest = backtest
        self.propfirm = propfirm
        self.num_simulations = num_simulations

        # Trade pool: fully loaded with commission + slippage
        boundaries = np.flatnonzero(backtest.trade_indicies)
        self.trade_pool = np.add.reduceat(backtest.candle_pnl, boundaries)

        self.rng = np.random.default_rng()

        # Challenge results
        self.ch_pass_rate = None
        self.ch_avg_days_to_pass = None
        self.ch_median_days_to_pass = None
        self.ch_avg_attempts_to_pass = None
        self.ch_expected_cost_to_pass = None

        # Optimal sizing results
        self.optimal_sizing = None
        self.optimal_sizing_micros = None

        # Funded results
        self.fu_payout_rate = None
        self.fu_min_payout = None
        self.fu_avg_payout = None
        self.fu_avg_days_to_payout = None

        # Combined EV
        self.total_ev = None

    def run(self):
        self._compute_avg_trades_per_day()
        self._simulate_challenge()
        self._compute_optimal_sizing()

    def _compute_avg_trades_per_day(self):
        bt = self.backtest
        trade_mask = bt.trade_indicies.astype(bool)
        trade_dates = bt.indicies[trade_mask].normalize()
        unique_trading_days = len(trade_dates.unique())
        if unique_trading_days == 0:
            self.avg_trades_per_day = 0
        else:
            self.avg_trades_per_day = bt.total_trades / unique_trading_days

    def _sample_day_trades(self):
        n = max(self.rng.poisson(self.avg_trades_per_day), 1)
        return self.rng.choice(self.trade_pool, size=n, replace=True)

    def _check_consistency(self, daily_pnls):
        pf = self.propfirm
        if pf.ch_consistency_pct <= 0:
            return True
            
        # Convert to numpy array for C-speed operations
        total_profit = 0
        max_day_profit = float('-inf')

        for pnl in daily_pnls:
            if pnl > 0:
                total_profit += pnl
                if pnl > max_day_profit:
                    max_day_profit = pnl
        
        return (max_day_profit / total_profit) <= pf.ch_consistency_pct
    
    def _simulate_challenge(self):
        pass_count = 0
        days_to_pass = []

        for _ in range(self.num_simulations):
            result = self._run_single_challenge()
            if result is not None:
                pass_count += 1
                days_to_pass.append(result)

        self.ch_pass_rate = pass_count / self.num_simulations
        if days_to_pass:
            self.ch_avg_days_to_pass = np.mean(days_to_pass)
            self.ch_median_days_to_pass = np.median(days_to_pass)
            self.ch_avg_attempts_to_pass = 1 / self.ch_pass_rate
            self.ch_expected_cost_to_pass = self.ch_avg_attempts_to_pass * self.propfirm.ch_cost
        else:
            self.ch_avg_days_to_pass = None
            self.ch_median_days_to_pass = None
            self.ch_avg_attempts_to_pass = None
            self.ch_expected_cost_to_pass = None

    def _run_single_challenge(self):
        pf = self.propfirm
        equity = pf.ch_initial_equity
        high_water_mark = equity
        daily_pnls = []
        trading_days = 0

        upper_barrier = pf.ch_initial_equity + pf.ch_profit_target
        lower_barrier = pf.ch_initial_equity - pf.ch_max_loss
        daily_profit_cap = pf.ch_profit_target * pf.ch_consistency_pct if pf.ch_consistency_pct > 0 else pf.ch_profit_target

        n_trades_per_day = self.rng.poisson(self.avg_trades_per_day, size=pf.ch_time_limit_days)
        n_trades_per_day = np.maximum(n_trades_per_day, 1)
        total_trades = np.sum(n_trades_per_day)
        random_indices = self.rng.integers(0, len(self.trade_pool), size=total_trades)
        all_trades = self.trade_pool[random_indices]
        index = 0

        for day in range(pf.ch_time_limit_days):
            trades = all_trades[index: index + n_trades_per_day[day]]
            index += n_trades_per_day[day]

            pnl: np.ndarray = np.cumsum(trades)

            if daily_profit_cap is not None:
                if np.max(pnl) >= daily_profit_cap:
                    pnl = pnl[:np.argmax(pnl >= daily_profit_cap) + 1]
                    pnl[-1] = daily_profit_cap
            intraday_equity: np.ndarray = equity + pnl

            if pf.ch_daily_loss_limit > 0:
                if np.max(-pnl) >= pf.ch_daily_loss_limit:
                    equity -= pf.ch_daily_loss_limit
                    daily_pnls.append(-pf.ch_daily_loss_limit)
                    trading_days += 1
                    continue

            if pf.ch_drawdown_type == "trailing_intraday":
                intraday_peaks: np.ndarray = np.maximum.accumulate(intraday_equity)
                intraday_hwm: np.ndarray = np.maximum(high_water_mark, intraday_peaks)
                trailing_floor: np.ndarray = intraday_hwm - pf.ch_max_loss
                if np.min(intraday_equity) <= trailing_floor:
                    return None
                high_water_mark = intraday_hwm[-1]
            elif pf.ch_drawdown_type == "eod":
                if np.min(intraday_equity) <= lower_barrier:
                    return None
                lower_barrier = max(lower_barrier, intraday_equity[-1] - pf.ch_max_loss)
            else:
                raise ValueError(f"Invalid drawdown type: {pf.ch_drawdown_type}")

            if (trading_days + 1) >= pf.ch_min_trading_days and np.max(intraday_equity) >= upper_barrier:
                if pf.ch_consistency_pct == 0:
                    return trading_days + 1

                pass_pnls = pnl[intraday_equity >= upper_barrier]
                daily_pnls_array = np.array(daily_pnls)
                positive_days = daily_pnls_array[daily_pnls_array > 0]
                total_profit = np.sum(positive_days) + pass_pnls
                max_pos = np.max(positive_days) if positive_days.size > 0 else 0
                max_profit = np.maximum(max_pos, pass_pnls)

                if np.min(max_profit / total_profit) <= pf.ch_consistency_pct:
                    return trading_days + 1

            equity = intraday_equity[-1]
            daily_pnls.append(pnl[-1])
            trading_days += 1

        return None
    
    def _compute_optimal_sizing(self):
        winrate = self.backtest.win_rate
        RR = self.backtest.ratio_avg_win_loss
        ev_rr = (winrate * RR) - (1 - winrate)
        # print(winrate, RR, ev_rr)

    def _simulate_funded(self):
        pass

    def _run_single_funded(self):
        pass

    def _compute_ev(self):
        pass

    def summary(self):
        pf = self.propfirm
        print(f"=== {pf.name} Challenge Simulation ({self.num_simulations:,} runs) ===")
        print(f"Pass rate:          {self.ch_pass_rate * 100:.2f}%")
        if self.ch_avg_days_to_pass is not None:
            print(f"Avg days to pass:   {self.ch_avg_days_to_pass:.1f}")
            print(f"Median days to pass: {self.ch_median_days_to_pass:.1f}")
        print(f"Avg trades/day:     {self.avg_trades_per_day:.2f}")
        print(f"Trade pool size:    {len(self.trade_pool)}")
        print(f"Avg attempts to pass: {self.ch_avg_attempts_to_pass:.2f}" if self.ch_avg_attempts_to_pass is not None else "Avg attempts to pass: N/A")
        print(f"Expected cost to pass: ${self.ch_expected_cost_to_pass:.2f}" if self.ch_expected_cost_to_pass is not None else "Expected cost to pass: N/A")
        print(f"Optimal sizing:     {self.optimal_sizing} contracts" if self.optimal_sizing is not None else "Optimal sizing: N/A")
        print(f"Optimal sizing (micros): {self.optimal_sizing_micros} contracts" if self.optimal_sizing_micros is not None else "Optimal sizing (micros): N/A")



# To update these terms in the future, visit the respective Prop Firm Match links below.
# Parameter order:
# 1.  name                   # 12. fu_activation_fee
# 2.  ch_initial_equity      # 13. fu_max_loss
# 3.  ch_profit_target       # 14. fu_daily_loss
# 4.  ch_max_loss            # 15. fu_consistency_pct
# 5.  ch_daily_loss_limit    # 16. fu_max_contracts_mini
# 6.  ch_drawdown_type       # 17. fu_max_contracts_micro
# 7.  ch_consistency_pct     # 18. fu_min_profit_days
# 8.  ch_max_contracts_mini  # 19. fu_min_profit_per_day
# 9.  ch_max_contracts_micro # 20. fu_days_to_payout
# 10. ch_time_limit_days     # 21. fu_payout_split
# 11. ch_min_trading_days    # 22. fu_payout_min_balance
# 12. ch_cost
prop_firms = [
    # =========================================================================================================================
    # LUCID TRADING (Flex & Pro Tracks Only - Direct Removed)
    # Rules nuance: 
    # - Flex uses a 50% consistency cap during challenge; 0% (none) when funded.
    # - Pro uses a 0% challenge consistency rule (1-day pass eligible) but enforces a strict 40% payout consistency cap.
    # =========================================================================================================================
    #            1. Name,         2. Eq,   3.Tgt, 4.MLss, 5.DLL, 6.Type,  7.Cons, 8.Mn, 9.Mc, 10.Lim, 11.Min, 12.Cost, | 13.Act, 14.fMLss, 15.fDLL, 16.fCons, 17.fMn, 18.fMc, 19.fWnD, 20.fWnP, 21.fPOut, 22.fSplt, 23.fMinB
    # PropFirm("Lucid 25K Flex",      25000,  1250,  1000,     0, "eod",   0.50,    2,   20,     30,       2,    50.00,     149,    1000,      0,    0.00,     2,    20,      5,    150,       14,    0.90,   26100),
    # PropFirm("Lucid 25K Pro",       25000,  1250,  1000,   500, "eod",   0.00,    2,   20,     30,       1,    67.50,     149,    1000,    500,    0.40,     2,    20,      3,    150,       14,    0.90,   26100),
    
    PropFirm("Lucid 50K Flex",      50000,  3000,  2000,     0, "eod",   0.50,    4,   40,     30,       2,    70.00,     149,    2000,      0,    0.00,     4,   40,       5,    150,       14,    0.90,   52100),
    # PropFirm("Lucid 50K Pro",       50000,  3000,  2000,  1100, "eod",   0.00,    4,   40,     30,       1,    92.50,     149,    2000,   1100,    0.40,     4,   40,       3,    150,       14,    0.90,   52100),
    
    # PropFirm("Lucid 100K Flex",    100000,  6000,  3000,     0, "eod",   0.50,    6,   60,     30,       2,   112.50,     149,    3000,      0,    0.00,     6,   60,       5,    150,       14,    0.90,  103100),
    # PropFirm("Lucid 100K Pro",     100000,  6000,  3000,  2200, "eod",   0.00,    6,   60,     30,       1,   142.50,     149,    3000,   2200,    0.40,     6,   60,       3,    150,       14,    0.90,  103100),
    
    # PropFirm("Lucid 150K Flex",    150000,  9000,  4500,     0, "eod",   0.50,   10,  100,     30,       2,   210.00,     169,    4500,      0,    0.00,    10,  100,       5,    150,       14,    0.90,  154600),
    # PropFirm("Lucid 150K Pro",     150000,  9000,  4500,  3300, "eod",   0.00,   10,  100,     30,       1,   185.00,     169,    4500,   3300,    0.40,    10,  100,       3,    150,       14,    0.90,  154600),

    # =========================================================================================================================
    # ALPHA FUTURES (Standard, Pro, Advanced Tracks)
    # Rules nuance: 
    # - Standard & Advanced evaluations have a 50% consistency rule; Pro has a 0% evaluation rule.
    # - Standard/Zero qualified tiers force a 40% payout consistency cap, while Pro/Advanced have 0% funded consistency.
    # =========================================================================================================================
    #            1. Name,         2. Eq,   3.Tgt, 4.MLss, 5.DLL, 6.Type,  7.Cons, 8.Mn, 9.Mc, 10.Lim, 11.Min, 12.Cost, | 13.Act, 14.fMLss, 15.fDLL, 16.fCons, 17.fMn, 18.fMc, 19.fWnD, 20.fWnP, 21.fPOut, 22.fSplt, 23.fMinB
    # PropFirm("Alpha 25K Standard",  25000,  1500,  1000,   500, "eod",   0.50,    1,   10,     30,       2,    59.25,     130,    1000,    500,    0.40,     1,   10,       5,    200,       14,    0.80,   26100),
    # PropFirm("Alpha 25K Pro",       25000,  1500,  1000,   500, "eod",   0.00,    2,   20,     30,       1,    79.25,       0,    1000,    500,    0.00,     2,   20,       5,    200,       14,    0.80,   26100),
    # PropFirm("Alpha 25K Advanced",  25000,  2000,   875,   500, "eod",   0.50,    3,   30,     30,       2,    89.25,     140,     875,    500,    0.00,     3,   30,       5,    200,       14,    0.80,   25975),
    
    # PropFirm("Alpha 50K Standard",  50000,  3000,  2000,  1100, "eod",   0.50,    3,   30,     30,       2,    89.25,     140,    2000,   1100,    0.40,     3,   30,       5,    200,       14,    0.80,   52100),
    # PropFirm("Alpha 50K Pro",       50000,  3000,  2000,  1100, "eod",   0.00,    4,   40,     30,       1,   119.25,       0,    2000,   1100,    0.00,     4,   40,       5,    200,       14,    0.80,   52100),
    # PropFirm("Alpha 50K Advanced",  50000,  4000,  1750,  1100, "eod",   0.50,    5,   50,     30,       2,   104.25,     150,    1750,   1100,    0.00,     5,   50,       5,    200,       14,    0.80,   51850),
    
    # PropFirm("Alpha 100K Standard",100000,  6000,  3000,  2200, "eod",   0.50,    6,   60,     30,       2,   179.25,     150,    3000,   2200,    0.40,     6,   60,       5,    200,       14,    0.80,  103100),
    # PropFirm("Alpha 100K Pro",     100000,  6000,  3000,  2200, "eod",   0.00,    8,   80,     30,       1,   119.25,       0,    3000,   2200,    0.00,     8,   80,       5,    200,       14,    0.80,  103100),
    # PropFirm("Alpha 100K Advanced",100000,  8000,  3500,  2200, "eod",   0.50,   10,  100,     30,       2,   209.25,     160,    3500,   2200,    0.00,    10,  100,       5,    200,       14,    0.80,  103600),
    
    # PropFirm("Alpha 150K Standard",150000,  9000,  4500,  3300, "eod",   0.50,   12,  120,     30,       2,   284.25,     160,    4500,   3300,    0.40,    12,  120,       5,    200,       14,    0.80,  154600),
    # PropFirm("Alpha 150K Pro",     150000,  9000,  4500,  3300, "eod",   0.00,   12,  120,     30,       1,   179.25,       0,    4500,   3300,    0.00,    12,  120,       5,    200,       14,    0.80,  154600),
    # PropFirm("Alpha 150K Advanced",150000, 12000,  5250,  3300, "eod",   0.50,   15,  150,     30,       2,   314.25,     190,    5250,   3300,    0.00,    15,  150,       5,    200,       14,    0.80,  155350),

    # =========================================================================================================================
    # MY FUNDED FUTURES / MFFU (Starter vs Expert Tracks)
    # Rules nuance: 
    # - Starter has 0% challenge consistency but enforces a 50% payout consistency target when funded.
    # - Expert scales with a 50% consistency evaluation rule but eliminates payout consistency targets later.
    # =========================================================================================================================
    #            1. Name,         2. Eq,   3.Tgt, 4.MLss, 5.DLL, 6.Type,  7.Cons, 8.Mn, 9.Mc, 10.Lim, 11.Min, 12.Cost, | 13.Act, 14.fMLss, 15.fDLL, 16.fCons, 17.fMn, 18.fMc, 19.fWnD, 20.fWnP, 21.fPOut, 22.fSplt, 23.fMinB
    # PropFirm("MFF 25K Starter",     25000,  1500,  1000,     0, "eod",   0.00,    2,   20,     30,       1,    76.00,     149,    1000,      0,    0.50,     2,    20,      1,      0,       14,    0.90,   26100),
    # PropFirm("MFF 25K Expert",      25000,  1500,  1000,     0, "eod",   0.50,    3,   30,     30,       2,    87.20,     149,    1000,      0,    0.00,     3,    30,      1,      0,       14,    0.90,   26100),
    
    # PropFirm("MFF 50K Starter",     50000,  3000,  2000,     0, "eod",   0.00,    3,   30,     30,       1,   122.40,     149,    2000,      0,    0.50,     3,    30,      1,      0,       14,    0.90,   52100),
    # PropFirm("MFF 50K Expert",      50000,  3000,  2000,     0, "eod",   0.50,    5,   50,     30,       2,   117.75,     149,    2000,      0,    0.00,     5,    50,      1,      0,       14,    0.90,   52100),
    
    # PropFirm("MFF 100K Starter",   100000,  6000,  3000,     0, "eod",   0.00,    6,   60,     30,       1,   172.00,     149,    3000,      0,    0.50,     6,    60,      1,      0,       14,    0.90,  103100),
    # PropFirm("MFF 100K Expert",    100000,  6000,  3000,     0, "eod",   0.50,   10,  100,     30,       2,   213.60,     149,    3000,      0,    0.00,    10,  100,      1,      0,       14,    0.90,  103100),
    
    # PropFirm("MFF 150K Starter",   150000,  9000,  4500,     0, "eod",   0.00,    9,   90,     30,       1,   238.50,     149,    4500,      0,    0.50,     9,    90,      1,      0,       14,    0.90,  154600),
    # PropFirm("MFF 150K Expert",    150000,  9000,  4500,     0, "eod",   0.50,   15,  150,     30,       2,   277.60,     149,    4500,      0,    0.00,    15,  150,      1,      0,       14,    0.90,  154600),

    # =========================================================================================================================
    # APEX TRADER FUNDING (Standard Evaluation Tracks - Full EOD Track Model)
    # Rules nuance:
    # - Evaluation phase requires a minimum of 7 trading days.
    # - Performance Accounts (PA) feature a 30% consistency ceiling strictly applied when requesting payouts.
    # =========================================================================================================================
    #            1. Name,         2. Eq,   3.Tgt, 4.MLss, 5.DLL, 6.Type,  7.Cons, 8.Mn, 9.Mc, 10.Lim, 11.Min, 12.Cost, | 13.Act, 14.fMLss, 15.fDLL, 16.fCons, 17.fMn, 18.fMc, 19.fWnD, 20.fWnP, 21.fPOut, 22.fSplt, 23.fMinB
    # PropFirm("Apex 25K Full",       25000,  1500,  1500,     0, "eod",   0.00,    4,   40,     30,       7,   147.00,      85,    1500,      0,    0.30,     4,    40,      5,      0,       10,    0.90,   26600),
    # PropFirm("Apex 50K Full",       50000,  3000,  2500,     0, "eod",   0.00,   10,  100,     30,       7,   160.00,      85,    2500,      0,    0.30,    10,   100,      5,      0,       10,    0.90,   52600),
    # PropFirm("Apex 75K Full",       75000,  4250,  2750,     0, "eod",   0.00,   11,  110,     30,       7,   180.00,      85,    2750,      0,    0.30,    11,   110,      5,      0,       10,    0.90,   77850),
    # PropFirm("Apex 100K Full",     100000,  6000,  3000,     0, "eod",   0.00,   14,  140,     30,       7,   220.00,      85,    3000,      0,    0.30,    14,   140,      5,      0,       10,    0.90,  103100),
    # PropFirm("Apex 150K Full",     150000,  9000,  5000,     0, "eod",   0.00,   17,  170,     30,       7,   340.00,      85,    5000,      0,    0.30,    17,   170,      5,      0,       10,    0.90,  155100),
    # PropFirm("Apex 250K Full",     250000, 15000,  6500,     0, "eod",   0.00,   27,  270,     30,       7,   550.00,      85,    6500,      0,    0.30,    27,   270,      5,      0,       10,    0.90,  256600),
    # PropFirm("Apex 300K Full",     300000, 20000,  7500,     0, "eod",   0.00,   35,  350,     30,       7,   700.00,      85,    7500,      0,    0.30,    35,   350,      5,      0,       10,    0.90,  307600)
]


