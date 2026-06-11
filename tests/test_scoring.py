from __future__ import annotations

import unittest

from toolmem.scoring import score_episode


class ScoringTests(unittest.TestCase):
    def test_wasteful_memory_lowers_score(self) -> None:
        trace = []
        efficient = score_episode(
            True,
            [],
            trace,
            {"active_saved_tools": 1, "saved_tools_never_used": 0, "exact_duplicate_rate": 0},
        )
        wasteful = score_episode(
            True,
            [],
            trace,
            {
                "active_saved_tools": 20,
                "saved_tools_never_used": 15,
                "exact_duplicate_rate": 0.8,
            },
        )
        self.assertGreater(efficient["memory"], wasteful["memory"])
        self.assertGreater(efficient["composite"], wasteful["composite"])


if __name__ == "__main__":
    unittest.main()
