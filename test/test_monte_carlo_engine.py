"""
Tests for MonteCarloEngine.
All tests use synthetic trade-log fixtures with known P&Ls — no real data.
n_simulations=1000 + seed=42 for speed and determinism.
"""
import math
import pytest
from monte_carlo_engine import MonteCarloEngine

# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_trade(pnl: float) -> dict:
    return {"pnl": pnl, "exit_reason": "stop_loss" if pnl < 0 else "trailing_stop"}

# 10 winning trades → equity 100k→110k, never ruins
ALL_WIN = [make_trade(+1_000)] * 10

# 10 losing trades → equity 100k→50k, always below ruin floor (100k×0.60=60k)
ALL_LOSS = [make_trade(-5_000)] * 10

# Alternating wins/losses — uncertain outcome
MIXED = [make_trade(+2_000 if i % 2 == 0 else -1_000) for i in range(20)]

INITIAL = 100_000.0

REQUIRED_KEYS = {
    "p5_final", "p50_final", "p95_final",
    "p_ruin",
    "p95_max_drawdown",
    "median_cagr",
    "equity_band",
    "p5_sharpe", "p50_sharpe", "p95_sharpe",
    "p5_win_rate", "p50_win_rate", "p95_win_rate",
    "p95_max_consec_losses",
    "kelly_fraction",
    "median_time_to_ruin",
    "ruin_severity",
}


# ── Cycle 1: output contract ──────────────────────────────────────────────────

class TestOutputContract:
    def test_result_has_all_required_keys(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(MIXED, INITIAL)
        assert REQUIRED_KEYS == set(r.keys())

    def test_p_ruin_is_float(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(MIXED, INITIAL)
        assert isinstance(r["p_ruin"], float)

    def test_equity_band_is_list(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(MIXED, INITIAL)
        assert isinstance(r["equity_band"], list)


# ── Cycle 2: P(ruin) ──────────────────────────────────────────────────────────

class TestPRuin:
    def test_all_winning_trades_p_ruin_is_zero(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_WIN, INITIAL)
        assert r["p_ruin"] == 0.0

    def test_all_losing_trades_p_ruin_is_one(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_LOSS, INITIAL)
        assert r["p_ruin"] == 1.0

    def test_p_ruin_between_zero_and_one(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(MIXED, INITIAL)
        assert 0.0 <= r["p_ruin"] <= 1.0


# ── Cycle 3: equity percentile ordering ──────────────────────────────────────

class TestEquityPercentiles:
    def test_p5_leq_p50_leq_p95(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(MIXED, INITIAL)
        assert r["p5_final"] <= r["p50_final"] <= r["p95_final"]

    def test_all_win_p50_above_initial(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_WIN, INITIAL)
        assert r["p50_final"] > INITIAL

    def test_all_loss_p50_below_initial(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_LOSS, INITIAL)
        assert r["p50_final"] < INITIAL


# ── Cycle 4: equity band ──────────────────────────────────────────────────────

class TestEquityBand:
    def test_equity_band_has_20_steps(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(MIXED, INITIAL)
        assert len(r["equity_band"]) == 20

    def test_equity_band_entry_has_required_keys(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(MIXED, INITIAL)
        for entry in r["equity_band"]:
            assert {"step", "p5", "p50", "p95"} == set(entry.keys())

    def test_equity_band_ordering_at_each_step(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(MIXED, INITIAL)
        for entry in r["equity_band"]:
            assert entry["p5"] <= entry["p50"] <= entry["p95"]

    def test_equity_band_step_zero_near_initial(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(MIXED, INITIAL)
        # First step: no trades applied yet, all sims start at initial_portfolio
        assert r["equity_band"][0]["p50"] == pytest.approx(INITIAL, rel=0.01)


# ── Cycle 5: CAGR ─────────────────────────────────────────────────────────────

class TestCAGR:
    def test_median_cagr_positive_for_all_wins(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_WIN, INITIAL)
        assert r["median_cagr"] > 0.0

    def test_median_cagr_negative_for_all_losses(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_LOSS, INITIAL)
        assert r["median_cagr"] < 0.0


# ── Cycle 6: Sharpe distribution ─────────────────────────────────────────────

class TestSharpeDistribution:
    def test_sharpe_ordering(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(MIXED, INITIAL)
        assert r["p5_sharpe"] <= r["p50_sharpe"] <= r["p95_sharpe"]

    def test_all_win_p50_sharpe_positive(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_WIN, INITIAL)
        assert r["p50_sharpe"] > 0.0

    def test_all_loss_p50_sharpe_negative(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_LOSS, INITIAL)
        assert r["p50_sharpe"] < 0.0


# ── Cycle 7: win rate distribution ───────────────────────────────────────────

class TestWinRateDistribution:
    def test_win_rate_ordering(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(MIXED, INITIAL)
        assert r["p5_win_rate"] <= r["p50_win_rate"] <= r["p95_win_rate"]

    def test_all_win_p50_win_rate_is_one(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_WIN, INITIAL)
        assert r["p50_win_rate"] == pytest.approx(1.0)

    def test_all_loss_p50_win_rate_is_zero(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_LOSS, INITIAL)
        assert r["p50_win_rate"] == pytest.approx(0.0)

    def test_win_rate_between_zero_and_one(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(MIXED, INITIAL)
        assert 0.0 <= r["p5_win_rate"] <= 1.0
        assert 0.0 <= r["p95_win_rate"] <= 1.0


# ── Cycle 8: p95 max consecutive losses ──────────────────────────────────────

class TestConsecutiveLosses:
    def test_p95_max_consec_losses_is_int(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(MIXED, INITIAL)
        assert isinstance(r["p95_max_consec_losses"], int)

    def test_all_win_consec_losses_is_zero(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_WIN, INITIAL)
        assert r["p95_max_consec_losses"] == 0

    def test_all_loss_consec_losses_equals_trade_count(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_LOSS, INITIAL)
        assert r["p95_max_consec_losses"] == len(ALL_LOSS)


# ── Cycle 9: Kelly fraction ───────────────────────────────────────────────────

class TestKellyFraction:
    def test_kelly_positive_for_profitable_trades(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_WIN, INITIAL)
        assert r["kelly_fraction"] > 0.0

    def test_kelly_negative_for_losing_trades(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_LOSS, INITIAL)
        assert r["kelly_fraction"] < 0.0

    def test_kelly_is_float(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(MIXED, INITIAL)
        assert isinstance(r["kelly_fraction"], float)


# ── Cycle 10: time-to-ruin and ruin severity ──────────────────────────────────

class TestRuinDetail:
    def test_time_to_ruin_none_when_no_ruin(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_WIN, INITIAL)
        assert r["median_time_to_ruin"] is None

    def test_ruin_severity_none_when_no_ruin(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_WIN, INITIAL)
        assert r["ruin_severity"] is None

    def test_time_to_ruin_is_int_when_ruin_occurs(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_LOSS, INITIAL)
        assert isinstance(r["median_time_to_ruin"], int)

    def test_time_to_ruin_leq_trade_count(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run(ALL_LOSS, INITIAL)
        assert r["median_time_to_ruin"] <= len(ALL_LOSS)

    def test_ruin_severity_below_ruin_floor_when_ruin_occurs(self):
        """Mean final equity in ruined sims must be below the ruin floor."""
        mc   = MonteCarloEngine(n_simulations=1000, ruin_threshold=0.40, seed=42)
        r    = mc.run(ALL_LOSS, INITIAL)
        ruin_floor = INITIAL * (1 - 0.40)
        assert r["ruin_severity"] < ruin_floor


# ── Cycle 11: configurable ruin threshold ────────────────────────────────────

class TestConfiguration:
    def test_tight_ruin_threshold_increases_p_ruin(self):
        """A tighter threshold (0.05 = ruin if equity drops 5%) catches more sims."""
        mc_tight = MonteCarloEngine(n_simulations=1000, ruin_threshold=0.05, seed=42)
        mc_loose = MonteCarloEngine(n_simulations=1000, ruin_threshold=0.40, seed=42)
        r_tight  = mc_tight.run(MIXED, INITIAL)
        r_loose  = mc_loose.run(MIXED, INITIAL)
        assert r_tight["p_ruin"] >= r_loose["p_ruin"]

    def test_n_simulations_respected(self):
        """With a fixed seed, small n produces same ordering as large n."""
        mc = MonteCarloEngine(n_simulations=200, seed=42)
        r  = mc.run(MIXED, INITIAL)
        assert r["p5_final"] <= r["p50_final"] <= r["p95_final"]


# ── Cycle 12: empty trade log ────────────────────────────────────────────────

class TestEmptyTradeLog:
    def test_empty_trade_log_does_not_crash(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run([], INITIAL)
        assert set(r.keys()) == REQUIRED_KEYS

    def test_empty_trade_log_p_ruin_is_zero(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run([], INITIAL)
        assert r["p_ruin"] == 0.0

    def test_empty_trade_log_p50_equals_initial(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run([], INITIAL)
        assert r["p50_final"] == pytest.approx(INITIAL)

    def test_empty_trade_log_time_to_ruin_is_none(self):
        mc = MonteCarloEngine(n_simulations=1000, seed=42)
        r  = mc.run([], INITIAL)
        assert r["median_time_to_ruin"] is None
