"""Invariants of the relative model: the forbidden quadrant never happens, and the two HOLD kinds.

The headline guarantee is no longer "no oscillation via a closeness margin" (one-and-done state
handles that) but the dominance invariant: a grabbed pick is never BOTH lower-score AND bigger than
the current file, because a worse-on-both candidate has a lower closeness than the current file and
the decision only ACTs when a candidate beats it.
"""

import random

from optimizarr.features.optimizer.config import default_topsis
from optimizarr.features.optimizer.decision import decide
from optimizarr.features.optimizer.topsis import GB, Topsis

CFG = default_topsis()
CFG.min_candidates = 2  # swap tests use 2-candidate pools; pool-size is tested in test_topsis
T = Topsis(CFG)
PRESETS = ("Remux", "Quality", "Balanced", "Efficient", "Compact")


def _release(score, size_gb, res=2160, guid="g"):
    return {
        "guid": guid,
        "title": f"{res}p {size_gb}GB {guid}",
        "customFormatScore": score,
        "quality": {"quality": {"resolution": res}},
        "size": int(size_gb * GB),
        "rejections": [],
    }


def _file(score, size_gb, res=2160):
    return {
        "id": 1,
        "customFormatScore": score,
        "size": int(size_gb * GB),
        "quality": {"quality": {"resolution": res}},
    }


def test_pick_is_never_lower_score_and_bigger():
    """The forbidden quadrant: across random pools and presets, a grabbed pick is never both
    lower-score AND bigger than the current file."""
    rnd = random.Random(11)
    for profile in PRESETS:
        for _ in range(600):
            pool = [
                _release(rnd.randint(0, 1_000_000), round(rnd.uniform(3.0, 30.0), 1), guid=f"g{i}")
                for i in range(rnd.randint(1, 8))
            ]
            cur = _file(rnd.randint(0, 1_000_000), round(rnd.uniform(3.0, 30.0), 1))
            d = decide(T, pool, 2.0, f"2160p {profile}", 2160, current_file=cur)
            if d.action != "ACT" or d.pick is None:
                continue
            cur_gb = cur["size"] / GB
            lower = (d.pick["score"] or 0) < (cur["customFormatScore"] or 0)
            bigger = d.pick["size_gb"] > cur_gb + 1e-6
            assert not (lower and bigger), (profile, cur["customFormatScore"], d.pick)


def test_too_few_candidates_holds_without_satisfying():
    # One candidate (< min_candidates) -> HOLD, not satisfied (retry later).
    d = decide(
        T, [_release(900_000, 8.0)], 2.0, "2160p Efficient", 2160, current_file=_file(800_000, 20.0)
    )
    assert d.action == "HOLD" and d.satisfy is False
    assert "too few" in d.reason


def test_current_optimal_holds_and_satisfies():
    # Current file is the best possible (highest score AND smallest) -> HOLD and satisfy.
    cur = _file(1_000_000, 6.0)
    pool = [_release(1_000_000, 6.0, guid="same"), _release(900_000, 20.0, guid="worse")]
    d = decide(T, pool, 2.0, "2160p Quality", 2160, current_file=cur)
    assert d.action == "HOLD" and d.satisfy is True


def test_unknown_current_score_acts_on_any_candidate():
    cur = {
        "id": 1,
        "size": int(20.0 * GB),
        "quality": {"quality": {"resolution": 2160}},
    }  # no score
    pool = [_release(900_000, 8.0), _release(800_000, 10.0)]
    d = decide(T, pool, 2.0, "2160p Quality", 2160, current_file=cur)
    assert d.action == "ACT"
