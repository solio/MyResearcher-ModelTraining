"""Deterministic checks for the v4 report pipeline.

Run from the ModelTraining worktree root:

    python3 -m unittest discover -s v4/tests -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v4.build_v4 import check_data  # noqa: E402
from v4.lib.analysis import events_for  # noqa: E402
from v4.lib.timebase import (  # noqa: E402
    forward_return,
    next_trading_day,
    post_info_day,
    prior_drawdown,
)


CAL = [
    "2026-10-09",
    "2026-10-12",
    "2026-10-13",
    "2026-10-14",
    "2026-10-15",
    "2026-10-16",
    "2026-10-19",
    "2026-10-20",
]


class TimebaseTests(unittest.TestCase):
    def test_post_at_cutoff_is_known_at_that_close(self):
        self.assertEqual(post_info_day("2026-10-16T14:59:59+08:00", CAL), "2026-10-16")
        self.assertEqual(post_info_day("2026-10-16T15:00:00+08:00", CAL), "2026-10-16")

    def test_post_after_cutoff_waits_for_next_day(self):
        self.assertEqual(post_info_day("2026-10-16T15:00:01+08:00", CAL), "2026-10-19")
        self.assertEqual(post_info_day("2026-10-16T22:44:49+08:00", CAL), "2026-10-19")

    def test_weekend_maps_to_monday(self):
        self.assertEqual(post_info_day("2026-10-17T10:00:00+08:00", CAL), "2026-10-19")

    def test_next_trading_day(self):
        self.assertEqual(next_trading_day(CAL, "2026-10-16"), "2026-10-19")
        self.assertIsNone(next_trading_day(CAL, "2026-10-20"))

    def test_forward_return_is_next_open_to_h_close(self):
        bars = {
            "2026-10-09": {"open": 10.0, "close": 10.0},
            "2026-10-12": {"open": 10.5, "close": 10.8},
            "2026-10-13": {"open": 10.9, "close": 11.0},
            "2026-10-14": {"open": 11.0, "close": 11.2},
            "2026-10-15": {"open": 11.2, "close": 11.4},
            "2026-10-16": {"open": 11.4, "close": 11.6},
            "2026-10-19": {"open": 11.6, "close": 11.8},
        }
        entry, exit_, ret, status = forward_return(bars, CAL, "2026-10-09", 3)
        self.assertEqual(entry, "2026-10-12")
        self.assertEqual(exit_, "2026-10-15")
        self.assertEqual(status, "COMPLETE")
        self.assertAlmostEqual(ret, 11.4 / 10.5 - 1)

    def test_forward_return_pending(self):
        bars = {"2026-10-12": {"open": 10.0, "close": 10.1}}
        entry, exit_, ret, status = forward_return(bars, CAL, "2026-10-09", 10)
        self.assertEqual(entry, "2026-10-12")
        self.assertIsNone(exit_)
        self.assertIsNone(ret)
        self.assertEqual(status, "PENDING_EXIT")

    def test_prior_drawdown(self):
        closes = {d: 100.0 for d in CAL[:6]}
        closes["2026-10-16"] = 80.0
        self.assertAlmostEqual(prior_drawdown(closes, CAL, "2026-10-16", 20), -0.2)


def _series_rows(z_values, dd=(-0.25,) * 8):
    rows = []
    for i, z in enumerate(z_values):
        rows.append(
            {
                "d": f"2026-01-{i + 1:02d}",
                "pos": i,
                "n": 10,
                "eligible": True,
                "z": z,
                "dd": dd[i] if i < len(dd) else -0.25,
                "density": 0.4,
                "p5": -0.1,
                "returns": {5: 0.01},
                "statuses": {5: "COMPLETE"},
                "entry": {5: f"2026-01-{i + 2:02d}"},
                "exit": {5: f"2026-01-{i + 7:02d}"},
                "deep": True,
            }
        )
    return {"code": "TEST", "days": rows, "baselines": {5: {"mean": 0.0, "n": len(rows)}}}


class EventRuleTests(unittest.TestCase):
    def test_third_consecutive_extreme_is_skipped(self):
        series = _series_rows([2.5] * 8)
        events = events_for(series, 2.0, -0.20, 5, 5)
        # positions 0,1 entered; position 2 is the third in the cluster and skipped;
        # position 3 is a fourth and skipped; 4..7 continue to be skipped because
        # the gap to the previous extreme stays <= 5. The rule must not enter a
        # long panic cluster after its second day.
        self.assertEqual([e["_index"] for e in events], [0, 1])

    def test_cluster_resets_after_gap(self):
        series = _series_rows([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
        # Put the last extreme at a position more than five sessions after any
        # previous extreme: use a sparse series instead.
        sparse = _series_rows([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
        sparse["days"][7]["pos"] = 20
        events = events_for(sparse, 2.0, -0.20, 5, 5)
        self.assertEqual([e["_index"] for e in events], [0, 1, 20])

    def test_deep_gate_excludes_non_deep(self):
        series = _series_rows([2.5] + [2.5] * 7, dd=(-0.19,) * 8)
        events = events_for(series, 2.0, -0.20, 5, 5)
        self.assertEqual(events, [])

    def test_z_gate(self):
        series = _series_rows([2.5] + [2.5] * 7)
        self.assertEqual(events_for(series, 3.0, -0.20, 5, 5), [])


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "v4" / "out" / "v4-data.json"
        if not path.is_file():
            raise unittest.SkipTest("v4/out/v4-data.json is not built yet")
        cls.data = json.loads(path.read_text(encoding="utf-8"))

    def test_schema_invariants(self):
        errors = check_data(self.data)
        self.assertEqual(errors, [])

    def test_versions_and_candidate_count(self):
        self.assertEqual(self.data["report"]["observation"], "4.0")
        self.assertEqual(self.data["report"]["training"], "4.1")
        self.assertEqual(self.data["report"]["prediction"], "4.2")
        self.assertEqual(
            self.data["prediction"]["headline"]["candidate_count"],
            len(self.data["prediction"]["candidates"]),
        )

    def test_specification_curve_has_all_48_points(self):
        spec = self.data["training"]["specification_curve"]
        self.assertEqual(spec["n_tests"], 48)
        self.assertEqual(len(spec["points"]), 48)
        self.assertTrue(spec["primary"]["is_primary"])

    def test_no_raw_post_fields(self):
        forbidden = {"title", "content", "author_id", "author_name"}
        found = set()

        def walk(node):
            if isinstance(node, dict):
                found.update(forbidden.intersection(node))
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(self.data)
        self.assertEqual(found, set())

    def test_asof_separates_info_and_market_days(self):
        self.assertEqual(self.data["as_of"]["info_day"], "2026-10-16")
        self.assertEqual(self.data["as_of"]["market_as_of"], "2026-10-19")
        self.assertEqual(self.data["as_of"]["next_trading_day"], "2026-10-19")

    def test_h4_is_forward_and_failed(self):
        h4 = self.data["prediction"]["headline"]["h4"]
        self.assertEqual(h4["status"], "matured")
        self.assertLess(h4["gross_pp"], -15)
        self.assertEqual(h4["signal_date"], "2026-10-09")


if __name__ == "__main__":
    unittest.main()
