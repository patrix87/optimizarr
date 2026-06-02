"""Tests for the swap rule (closeness-gain + resolution guard) and its no-oscillation guarantee."""

import random

from optimizarr.features.optimizer.config import default_topsis
from optimizarr.features.optimizer.decision import decide, resolution_ok, swap_allowed
from optimizarr.features.optimizer.topsis import GB, Topsis

CFG = default_topsis()
T = Topsis(CFG)
PRESETS = ("Remux", "Quality", "Balanced", "Efficient", "Compact")


def _release(score, res, size_gb, guid="g"):
    return {
        "guid": guid,
        "title": f"{res}p {size_gb}GB",
        "customFormatScore": score,
        "quality": {"quality": {"resolution": res}},
        "size": int(size_gb * GB),
        "rejections": [],
    }


def _file(score, res, size_gb):
    return {
        "id": 1,
        "customFormatScore": score,
        "size": int(size_gb * GB),
        "quality": {"quality": {"resolution": res}},
    }


# ----- resolution guard -----


def test_resolution_ok_blocks_drop_below_target():
    assert not resolution_ok(2160, 1080, 2160)  # drop below target
    assert resolution_ok(1080, 2160, 2160)  # upgrade toward target
    assert resolution_ok(2160, 2160, 2160)


def test_resolution_ok_allows_drop_to_a_leaner_target():
    assert resolution_ok(2160, 1080, 1080)  # leaner profile: dropping to its target is fine
    assert not resolution_ok(2160, 720, 1080)  # but never below the target


def test_resolution_ok_no_target_means_no_downgrade():
    assert resolution_ok(1080, 2160, None)
    assert not resolution_ok(2160, 1080, None)


# ----- closeness-gain rule -----


def test_swap_requires_closeness_margin():
    assert swap_allowed(0.50, 0.55, 2160, 2160, 2160, 0.02)
    assert not swap_allowed(0.50, 0.51, 2160, 2160, 2160, 0.02)  # below the margin
    assert not swap_allowed(0.50, 0.50, 2160, 2160, 2160, 0.02)  # equal


def test_swap_unknown_current_is_an_upgrade_but_still_guards_resolution():
    assert swap_allowed(None, 0.1, 2160, 2160, 2160, 0.02)
    assert not swap_allowed(None, 0.9, 2160, 1080, 2160, 0.02)  # res guard still applies


def test_swap_resolution_guard_overrides_a_big_gain():
    assert not swap_allowed(0.1, 0.99, 2160, 1080, 2160, 0.02)


# ----- the "no senseless swap" property falls out of closeness -----


def test_lower_score_bigger_or_equal_is_never_grabbed():
    """Same resolution, score <= current and size >= current cannot raise closeness, so it is
    never grabbed — for any preset. (No explicit rule needed; it follows from the gate.)"""
    rnd = random.Random(7)
    for profile in PRESETS:
        for _ in range(400):
            cur_score = rnd.randint(0, 1_000_000)
            cur_gbh = round(rnd.uniform(4.0, 12.0), 2)
            cand_score = rnd.randint(0, cur_score)  # <= current
            cand_gbh = round(rnd.uniform(cur_gbh, 18.0), 2)  # >= current
            cur = _file(cur_score, 2160, cur_gbh * 2)  # runtime 2h: gbh = size_gb / 2
            cand = _release(cand_score, 2160, cand_gbh * 2)
            d = decide(T, [cand], 2.0, f"2160p {profile}", 2160, current_file=cur)
            assert d.action == "HOLD", (profile, cur_score, cur_gbh, cand_score, cand_gbh)


# ----- the headline guarantee: convergence, no oscillation -----


def test_optimizer_converges_no_oscillation_on_random_pools():
    """Iterate decide -> grab -> re-evaluate on random static pools. Closeness must strictly
    increase on every grab (so a file is never revisited) and the walk must reach HOLD."""
    rnd = random.Random(20260601)
    for profile in PRESETS:
        rp = T.resolve_profile(f"2160p {profile}")
        gain = rp.min_closeness_gain
        for _ in range(300):
            pool = [
                _release(
                    rnd.randint(0, 1_000_000),
                    rnd.choice([720, 1080, 2160]),
                    round(rnd.uniform(0.5, 60.0), 1),
                    guid=f"g{i}",
                )
                for i in range(rnd.randint(1, 8))
            ]
            start = rnd.choice(pool)
            cur = _file(
                start["customFormatScore"],
                start["quality"]["quality"]["resolution"],
                start["size"] / GB,
            )
            prev_clo = None
            for _step in range(80):  # closeness rises >= gain each grab, so <= 1/gain grabs
                d = decide(T, pool, 2.0, f"2160p {profile}", 2160, current_file=cur)
                if d.action == "HOLD":
                    break
                assert d.pick is not None and d.current is not None and d.release is not None
                pick_clo = d.pick["closeness"]
                cur_clo = d.current["closeness"]
                if cur_clo is not None:
                    assert pick_clo >= cur_clo + gain - 1e-9, (profile, cur_clo, pick_clo)
                if prev_clo is not None:
                    assert pick_clo >= prev_clo - 1e-9  # monotonic across the whole walk
                prev_clo = pick_clo
                rel = d.release
                cur = _file(
                    rel["customFormatScore"],
                    rel["quality"]["quality"]["resolution"],
                    rel["size"] / GB,
                )
            else:
                raise AssertionError(f"{profile}: did not converge (possible oscillation)")
