from dataclasses import replace

import pytest

from optimizarr.features.optimizer.config import default_topsis
from optimizarr.features.optimizer.topsis import GB, Topsis, eligible


def _topsis() -> Topsis:
    return Topsis(default_topsis())


def _release(score=900_000, resolution=2160, size_gb=10.0, rejections=None, temp=False):
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


def test_size_band_drops_below_floor_and_above_ceiling():
    t = _topsis()
    ref = t.resolve_profile("2160p Balanced").reference  # 2160 -> (5.0, 8.1, 9.3, 11.0, 20.0)
    fake = _release(resolution=2160, size_gb=4.0)  # 2.0 GiB/h < floor 5.0
    real = _release(resolution=2160, size_gb=20.0)  # 10 GiB/h, in band
    bloated = _release(resolution=2160, size_gb=44.0)  # 22 GiB/h > ceiling 20
    kept = t.filter_by_size_band([fake, real, bloated], 2.0, ref)
    assert kept == [real]


def test_normalize_size_trapezoid():
    t = _topsis()  # size_shoulder = 0.85
    f, lo, tg, hi, c = 4.0, 8.0, 10.0, 12.0, 20.0
    assert t.normalize_size(4.0, f, lo, tg, hi, c) == 0.0  # at floor
    assert t.normalize_size(3.0, f, lo, tg, hi, c) == 0.0  # below floor -> too small, penalized
    assert t.normalize_size(8.0, f, lo, tg, hi, c) == pytest.approx(0.85)  # at lo = shoulder
    assert t.normalize_size(10.0, f, lo, tg, hi, c) == 1.0  # at target = peak
    assert t.normalize_size(12.0, f, lo, tg, hi, c) == pytest.approx(0.85)  # at hi = shoulder
    assert t.normalize_size(20.0, f, lo, tg, hi, c) == 0.0  # at ceiling
    assert t.normalize_size(25.0, f, lo, tg, hi, c) == 0.0  # above ceiling
    assert t.normalize_size(6.0, f, lo, tg, hi, c) == pytest.approx(0.425)  # midway floor->lo
    assert t.normalize_size(9.0, f, lo, tg, hi, c) == pytest.approx(0.925)  # midway lo->target


def test_normalize_size_degenerate_zero_width_peak():
    t = _topsis()
    # lo == target == hi: a sharp peak, shoulders still ramp to floor/ceiling
    f, lo, tg, hi, c = 4.0, 9.0, 9.0, 9.0, 16.0
    assert t.normalize_size(9.0, f, lo, tg, hi, c) == 1.0
    assert t.normalize_size(6.5, f, lo, tg, hi, c) == pytest.approx(0.85 * (6.5 - 4) / (9 - 4))
    assert t.normalize_size(12.5, f, lo, tg, hi, c) == pytest.approx(0.85 * (16 - 12.5) / (16 - 9))


def test_score_gap_keeps_cluster_drops_tail_and_negatives():
    t = _topsis()  # default score_gap = 0.20
    rels = [
        _release(score=1_000_000),
        _release(score=950_000),
        _release(score=930_000),
        _release(score=200_000),  # 930k -> 200k is a >20% drop: the tail
        _release(score=-50),  # negatives always dropped
    ]
    kept = t.filter_by_score_gap(rels)
    scores = sorted((r["customFormatScore"] for r in kept), reverse=True)
    assert scores == [1_000_000, 950_000, 930_000]


def test_resolve_profile_matches_preset_by_name():
    t = _topsis()
    rp = t.resolve_profile("1080p Efficient")
    assert rp.weights == t.cfg.presets["Efficient"].weights
    assert rp.reference == t.cfg.presets["Efficient"].reference
    assert rp.pick == "topsis"


def test_resolve_profile_falls_back_to_default_preset():
    t = _topsis()
    rp = t.resolve_profile("Something Unmatched")
    assert rp.weights == t.cfg.presets[t.cfg.default_preset].weights


def test_resolve_profile_override_reference_pick_and_gain():
    cfg = default_topsis()
    from optimizarr.features.optimizer.config import ProfileOverride

    cfg.profiles["Special"] = ProfileOverride(
        preset="Efficient",
        pick="min_size",
        reference={2160: (3.0, 5.0, 6.0, 8.0, 11.0)},
        min_closeness_gain=0.05,
    )
    rp = Topsis(cfg).resolve_profile("Special")
    assert rp.pick == "min_size"
    assert rp.reference == {2160: (3.0, 5.0, 6.0, 8.0, 11.0)}
    assert rp.min_closeness_gain == 0.05
    assert rp.weights == cfg.presets["Efficient"].weights  # inherited


def test_score_candidates_orders_by_closeness():
    t = _topsis()
    rp = t.resolve_profile("2160p Quality")
    good = _release(score=1_000_000, resolution=2160, size_gb=23.0)  # 11.5 GiB/h = Quality target
    weak = _release(score=950_000, resolution=1080, size_gb=14.0)  # lower res, 7 GiB/h
    scored, diag = t.score_candidates([weak, good], 2.0, rp, target_resolution=2160)
    assert scored[0][0] is good
    assert scored[0][2] >= scored[1][2]
    assert diag["input"] == 2


def test_compact_aims_at_its_band_not_the_smallest():
    # Sweet-spot model: Compact targets its band (~P10), not the absolute smallest. A near-peak
    # release with a strong score beats a smaller one that has fallen below the band.
    t = _topsis()
    rp = t.resolve_profile("Compact")  # 2160 band floor 3.5, lo 5.9, target 6.8, hi 7.6, ceiling 12
    below = _release(score=850_000, resolution=2160, size_gb=9.0)  # 4.5 GiB/h, below lo
    inband = _release(score=1_000_000, resolution=2160, size_gb=13.0)  # 6.5 GiB/h, near peak
    scored, _ = t.score_candidates([below, inband], 2.0, rp, 2160)
    selected = t.select(scored, rp)
    assert selected is not None and selected[0] is inband


def test_select_pick_methods():
    # select() still supports all three pick methods for profile overrides, applied to already-legal
    # candidates: max_score = highest score, min_size = smallest GiB/h, topsis = highest closeness.
    t = _topsis()
    rp = t.resolve_profile("Balanced")
    a = _release(score=900_000, resolution=2160, size_gb=18.0)  # 9 GiB/h
    b = _release(score=950_000, resolution=2160, size_gb=22.0)  # 11 GiB/h

    def attrs(r):
        return {"raw": {"score": r["customFormatScore"], "gbh": r["size"] / GB / 2}}

    cands = [(a, attrs(a), 0.50), (b, attrs(b), 0.60)]
    assert t.select(cands, replace(rp, pick="max_score"))[0] is b  # higher score
    assert t.select(cands, replace(rp, pick="min_size"))[0] is a  # smaller GiB/h
    assert t.select(cands, replace(rp, pick="topsis"))[0] is b  # higher closeness


def test_select_empty_returns_none():
    t = _topsis()
    assert t.select([], t.resolve_profile(None)) is None


def test_current_resolution_reads_quality_block():
    t = _topsis()
    # the file's own quality bucket is the source, reliable even for scope content whose pixel
    # height (mediaInfo) is misleadingly short
    f = {"quality": {"quality": {"resolution": 2160}}, "mediaInfo": {"resolution": "3840x1608"}}
    assert t._current_resolution(f) == 2160
    assert t._current_resolution({"quality": {"quality": {"resolution": 1080}}}) == 1080
    assert t._current_resolution({}) == 0  # unknown quality
    assert t._current_resolution({"quality": {"quality": {}}}) == 0
