"""
Adversarial Verification Probe: Benchmark Index Order Routing & Price Handling
Written by challenger_final (critic/specialist).
"""

import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_trader import (
    MultiBotTournamentEngine,
    is_benchmark_index,
    INDEX_TRADEABLE_MAP,
    get_current_price
)


def run_routing_adversarial_probes():
    print("=" * 80)
    print("ADVERSARIAL VERIFICATION: BENCHMARK ORDER ROUTING PRICE INTEGRITY")
    print("=" * 80)

    temp_dir = Path(tempfile.mkdtemp(prefix="adv_routing_probe_"))
    temp_file = temp_dir / "routing_probe_state.json"
    try:
        with patch("src.paper_trader.LOCAL_STATE_FILE", temp_file):
            engine = MultiBotTournamentEngine(client=None)

            # -------------------------------------------------------------
            # Test Case 1: Caller supplies index level price 23,146.0 for ^NSEI
            # -------------------------------------------------------------
            print("\n[Probe 1] Testing execute_buy('^NSEI') with caller current_price=23146.0...")
            with patch("src.paper_trader.get_current_price") as mock_get_price:
                # When ETF price is requested, return realistic ₹264.15
                def price_side_effect(sym):
                    if sym == "NIFTYBEES.NS":
                        return 264.15
                    elif sym == "^NSEI":
                        return 23146.0
                    return 100.0
                mock_get_price.side_effect = price_side_effect

                success = engine.execute_buy(
                    bot_id="conservative_swing",
                    symbol="^NSEI",
                    amount_dollars=500.0,
                    rationale="Adversarial index routing test with caller price 23146.0",
                    current_price=23146.0
                )

                assert success, "execute_buy should succeed for ^NSEI routing"
                positions = engine.storage.get_positions("conservative_swing")
                assert len(positions) == 1, f"Expected 1 position, got {len(positions)}"
                pos = positions[0]

                print(f"  Result Position Symbol: {pos['symbol']} (Expected: 'NIFTYBEES.NS')")
                print(f"  Result Average Buy Price: {pos['average_buy_price']} (Expected: 264.15, NOT 23146.0)")
                print(f"  Result Shares Purchased: {pos['shares']} (Expected: {round(500.0 / 264.15, 4)})")

                # Assertions
                assert pos["symbol"] == "NIFTYBEES.NS", f"Symbol mismatch: {pos['symbol']}"
                assert pos["average_buy_price"] == 264.15, f"Price mismatch: {pos['average_buy_price']} != 264.15"
                assert pos["average_buy_price"] != 23146.0, "FATAL: Buy price is poisoned with index level 23146.0!"
                expected_shares = round(500.0 / 264.15, 4)
                assert abs(pos["shares"] - expected_shares) < 1e-4, f"Shares mismatch: {pos['shares']} vs {expected_shares}"
                print("  --> [PASS] Probe 1: Successfully routed ^NSEI to NIFTYBEES.NS at 264.15, overriding 23146.0.")

            # -------------------------------------------------------------
            # Test Case 2: Valuation of portfolio - verify no artificial drawdown
            # -------------------------------------------------------------
            print("\n[Probe 2] Testing portfolio valuation and P&L consistency...")
            with patch("src.paper_trader.get_current_price") as mock_get_price:
                mock_get_price.return_value = 265.0  # Slight gain to 265.0

                perf = engine.get_bot_status("conservative_swing")
                print(f"  Total Equity: ${perf['total_equity']:.2f}")
                print(f"  Net P&L: ${perf['total_pnl']:.2f} ({perf['total_pnl_pct']:+.2f}%)")

                pos_eval = perf["positions"][0]
                print(f"  Holding Market Value: ${pos_eval['market_value']:.2f}")
                print(f"  Holding P&L: ${pos_eval['pnl']:.2f} ({pos_eval['pnl_pct']:+.2f}%)")

                # If it had been bought at 23146, valuation at 265 would register a ~ -98.86% loss
                assert perf["total_pnl_pct"] > -1.0, f"Unrealized drawdown anomaly detected: {perf['total_pnl_pct']}%"
                assert pos_eval["pnl_pct"] > -1.0, f"Position drawdown anomaly: {pos_eval['pnl_pct']}%"
                print("  --> [PASS] Probe 2: Portfolio valuation is accurate and free of price distortion drawdown.")

            # -------------------------------------------------------------
            # Test Case 3: Caller supplies current_price=None for ^NSEI
            # -------------------------------------------------------------
            print("\n[Probe 3] Testing execute_buy('^NSEI') with current_price=None...")
            engine.storage.remove_position("conservative_swing", "NIFTYBEES.NS")
            engine.storage.update_cash_balance("conservative_swing", 5000.0)
            with patch("src.paper_trader.get_current_price") as mock_get_price:
                mock_get_price.return_value = 263.80
                success = engine.execute_buy(
                    bot_id="conservative_swing",
                    symbol="^NSEI",
                    amount_dollars=300.0,
                    rationale="Test buy with None price",
                    current_price=None
                )
                assert success, "execute_buy should succeed"
                pos = engine.storage.get_positions("conservative_swing")[0]
                assert pos["symbol"] == "NIFTYBEES.NS"
                assert pos["average_buy_price"] == 263.80
                print(f"  Result Position: {pos['symbol']} @ {pos['average_buy_price']}")
                print("  --> [PASS] Probe 3: Executed at fetched market price 263.80.")

            # -------------------------------------------------------------
            # Test Case 4: Non-tradeable benchmark index (e.g. ^GSPC) rejected safely
            # -------------------------------------------------------------
            print("\n[Probe 4] Testing non-tradeable index without tradeable mapping (e.g. ^GSPC)...")
            success = engine.execute_buy(
                bot_id="conservative_swing",
                symbol="^GSPC",
                amount_dollars=500.0,
                rationale="Test non-tradeable index",
                current_price=5800.0
            )
            assert not success, "Non-tradeable index without mapping must be rejected"
            print("  --> [PASS] Probe 4: Non-tradeable benchmark index without mapping safely rejected.")

            # -------------------------------------------------------------
            # Test Case 5: Direct buy of NIFTYBEES.NS (non-routed)
            # -------------------------------------------------------------
            print("\n[Probe 5] Testing direct buy of NIFTYBEES.NS without routing...")
            with patch("src.paper_trader.get_current_price") as mock_get_price:
                mock_get_price.return_value = 264.50
                success = engine.execute_buy(
                    bot_id="conservative_swing",
                    symbol="NIFTYBEES.NS",
                    amount_dollars=400.0,
                    rationale="Direct ETF purchase",
                    current_price=264.50
                )
                assert success
                positions = engine.storage.get_positions("conservative_swing")
                assert len(positions) == 1, f"Expected 1 merged position for NIFTYBEES.NS, got {len(positions)}"
                pos = positions[0]
                assert pos["symbol"] == "NIFTYBEES.NS"
                expected_total_shares = round(1.1372 + 1.5123, 4)
                assert abs(pos["shares"] - expected_total_shares) < 1e-3, f"Expected merged shares ~{expected_total_shares}, got {pos['shares']}"
                print(f"  Result Merged Position: {pos['shares']} shares of {pos['symbol']} @ avg {pos['average_buy_price']}")
                print("  --> [PASS] Probe 5: Direct buy of NIFTYBEES.NS operates normally and merges position cleanly.")

            print("\n" + "=" * 80)
            print("ALL 5 BENCHMARK ROUTING ADVERSARIAL PROBES PASSED!")
            print("=" * 80)
            return True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_routing_adversarial_probes()
