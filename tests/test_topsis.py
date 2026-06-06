from optimizarr.features.optimizer.config import default_topsis
from optimizarr.features.optimizer.topsis import GB, Topsis, _norm, _score_floor_tier, eligible


def _topsis() -> Topsis:
    return Topsis(default_topsis())


def _release(score=900_000, size_gb=20.0, resolution=2160, rejections=None, temp=False):
    return {
        "customFormatScore": score,
        "quality": {"quality": {"resolution": resolution}},
        "size": int(size_gb * GB),
        "rejections": rejections or [],
        "temporarilyRejected": temp,
    }


def test_eligible_drops_blocklisted_and_temp():
    keep = eligible(
        [
            _release(),
            _release(rejections=["Release was blocklisted"]),
            _release(temp=True),
            _release(rejections=["Unable to parse release"]),
        ]
    )
    assert len(keep) == 1


def test_norm_minmax_and_degenerate():
    assert _norm(5, 0, 10) == 0.5
    assert _norm(5, 0, 10, invert=True) == 0.5
    assert _norm(20, 0, 10) == 1.0  # clamped
    assert _norm(-5, 0, 10) == 0.0  # clamped
    assert _norm(7, 7, 7) == 1.0  # degenerate range -> non-discriminating


def test_score_floor_tier_three_tiers():
    # Tier 1: enough candidates at or above max-score_window.
    scores = list(range(950_000, 950_000 + 6 * 10_000, 10_000))  # 6 in [950k, 1M]
    floor, tier = _score_floor_tier(scores, current_score=800_000, score_window=100_000, min_pool=6)
    assert tier == 1
    assert floor == max(scores) - 100_000

    # Tier 2: too few at top, but enough when expanded down to current_score.
    # max=990k, Tier-1 floor=890k; only top 3 are >= 890k (3 < 6).
    # current=860k; Tier-2 floor=860k; all 8 (3+5) are >= 860k (8 >= 6) -> Tier 2 fires.
    top = [990_000, 970_000, 950_000]  # 3 releases above Tier-1 floor 890k
    mid = [885_000, 880_000, 875_000, 870_000, 865_000]  # 5 releases below 890k, above 860k
    floor2, tier2 = _score_floor_tier(
        top + mid, current_score=860_000, score_window=100_000, min_pool=6
    )
    assert tier2 == 2
    assert floor2 == 860_000

    # Tier 3: sparse everywhere, falls all the way through.
    few = [1_000_000, 800_000, 700_000]
    floor3, tier3 = _score_floor_tier(few, current_score=900_000, score_window=100_000, min_pool=6)
    assert tier3 == 3
    assert floor3 == max(0, 900_000 - 100_000)


def test_score_window_tier1_tight_pool():
    # With enough top candidates, Tier 1 fires and the window is anchored at the top.
    t = _topsis()  # score_window=100000, min_candidates=6
    # 8 releases all within score_window of 1M (floor 900k), current far below at 500k.
    rels = [_release(score=1_000_000 - i * 5_000) for i in range(8)]  # 1M down to 965k
    kept = t.filter_by_score_window(rels, current_score=500_000)
    # Tier 1 floor = 1M - 100k = 900k; all 8 are >= 900k.
    assert len(kept) == 8
    assert all(r["customFormatScore"] >= 900_000 for r in kept)


def test_score_window_tier2_expands_to_current():
    t = _topsis()
    t.cfg.min_candidates = 6  # tier-expansion tests need a pool threshold above 3
    # Only 3 releases above the Tier-1 floor (900k); 6 more clustered just below current (880k).
    top = [_release(score=s) for s in [1_000_000, 990_000, 980_000]]
    mid = [_release(score=880_000 + i * 1_000) for i in range(6)]  # 880k-885k
    kept = t.filter_by_score_window(top + mid, current_score=880_000)
    # Tier 1: floor=900k, 3 survive < 6; Tier 2: floor=880k, 9 survive >= 6.
    assert len(kept) == 9
    assert all(r["customFormatScore"] >= 880_000 for r in kept)


def test_score_window_tier3_fallback_and_negatives():
    # With too few candidates at every tier, Tier 3 fires (full budget).
    t = _topsis()
    t.cfg.min_candidates = 6  # tier-expansion tests need a pool threshold above 3
    rels = [
        _release(score=1_000_000),
        _release(score=905_000),
        _release(score=850_000),  # below Tier-1 floor 900k and below Tier-2 floor 1M
        _release(score=-5),  # negative: always dropped
    ]
    # Only 3 non-negative candidates total -> all tiers fail min_candidates=6 -> Tier 3
    kept = t.filter_by_score_window(rels, current_score=1_000_000)
    assert sorted(r["customFormatScore"] for r in kept) == [905_000, 1_000_000]
    # Tier 3 floor = max(0, 1M-100k) = 900k -> 850k is dropped, 905k and 1M kept.

    # No current_score: Tier 2 is skipped, Tier 3 floor = 0.
    kept_no_cur = t.filter_by_score_window(rels, current_score=None)
    assert sorted(r["customFormatScore"] for r in kept_no_cur) == [850_000, 905_000, 1_000_000]


def test_filter_by_resolution_keeps_only_target():
    t = _topsis()
    rels = [_release(resolution=2160), _release(resolution=1080), _release(resolution=720)]
    assert [_["quality"]["quality"]["resolution"] for _ in t.filter_by_resolution(rels, 2160)] == [
        2160
    ]
    assert len(t.filter_by_resolution(rels, None)) == 3  # no target -> keep all


def test_filter_by_size_band_drops_below_floor_and_above_ceiling():
    t = _topsis()  # 2160 bounds (3.0, 30.0)
    fake = _release(size_gb=2.0)  # 1.0 GiB/h < floor 3.0
    real = _release(size_gb=18.0)  # 9.0 GiB/h, in band
    bloat = _release(size_gb=80.0)  # 40 GiB/h > ceiling 30.0
    assert t.filter_by_size_band([fake, real, bloat], 2.0) == [real]


def test_bounds_for_nearest_below():
    t = _topsis()
    floor_2160, ceil_2160 = t.bounds_for(2160)
    assert 0 < floor_2160 < ceil_2160
    # 1440 has no entry; should fall back to the nearest defined resolution below it (1080).
    floor_1440, ceil_1440 = t.bounds_for(1440)
    assert 0 < floor_1440 < ceil_1440
    assert (floor_1440, ceil_1440) != (floor_2160, ceil_2160)  # not the 2160 entry


def test_resolve_profile_matches_by_name():
    t = _topsis()
    rp = t.resolve_profile("1080p Efficient")
    assert rp.weights == t.cfg.presets["Efficient"].weights


def test_score_pool_relative_smallest_and_highest_get_ideal():
    t = _topsis()
    rp = t.resolve_profile("Balanced")
    rels = [_release(score=900_000, size_gb=10.0), _release(score=800_000, size_gb=30.0)]
    scored, _cur = t.score_pool(rels, None, 2.0, rp)
    by_score = {s[1]["raw"]["score"]: s[1] for s in scored}
    # Highest score -> n_score 1, smallest file -> n_size 1.
    assert by_score[900_000]["n_score"] == 1.0 and by_score[900_000]["n_size"] == 1.0
    assert by_score[800_000]["n_score"] == 0.0 and by_score[800_000]["n_size"] == 0.0


def test_score_pool_efficient_prefers_smaller_quality_prefers_score():
    t = _topsis()
    small_low = _release(score=900_000, size_gb=10.0)  # smaller, a touch lower score
    big_high = _release(score=950_000, size_gb=30.0)  # bigger, higher score
    eff, _ = t.score_pool([small_low, big_high], None, 2.0, t.resolve_profile("Efficient"))
    qual, _ = t.score_pool([small_low, big_high], None, 2.0, t.resolve_profile("Quality"))
    assert eff[0][1]["raw"]["score"] == 900_000  # Efficient takes the smaller
    assert qual[0][1]["raw"]["score"] == 950_000  # Quality takes the higher score


def test_score_pool_sorted_best_first():
    # The pick is always the highest-closeness candidate (scored is sorted best-first).
    t = _topsis()
    rp = t.resolve_profile("Compact")  # size-leaning
    rels = [_release(score=900_000, size_gb=10.0), _release(score=950_000, size_gb=30.0)]
    scored, _ = t.score_pool(rels, None, 2.0, rp)
    assert scored[0][1]["raw"]["size_gb"] == 10.0  # Compact ranks the smaller first
