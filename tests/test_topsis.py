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


def test_size_band_drops_below_floor_and_above_bloat():
    t = _topsis()
    ref = t.resolve_profile("2160p Balanced").reference  # 2160 -> (4.5, 9.0, 16.0)
    fake = _release(resolution=2160, size_gb=4.0)  # 2.0 GiB/h < floor 4.5
    real = _release(resolution=2160, size_gb=20.0)  # 10 GiB/h, in band
    bloated = _release(resolution=2160, size_gb=40.0)  # 20 GiB/h > bloat 16
    kept = t.filter_by_size_band([fake, real, bloated], 2.0, ref)
    assert kept == [real]


def test_normalize_size_one_sided_plateau_then_ramp():
    t = _topsis()
    # target=9, bloat=16 (2160 Balanced-ish)
    assert t.normalize_size(3.0, 9.0, 16.0) == 1.0  # below target: never penalized
    assert t.normalize_size(9.0, 9.0, 16.0) == 1.0  # at target: still 1.0
    assert t.normalize_size(16.0, 9.0, 16.0) == 0.0  # at bloat
    mid = t.normalize_size(12.5, 9.0, 16.0)  # halfway target->bloat
    assert abs(mid - 0.5) < 1e-9
    assert t.normalize_size(25.0, 9.0, 16.0) == 0.0  # past bloat


def test_normalize_size_degenerate_bloat_equals_target():
    t = _topsis()
    assert t.normalize_size(9.0, 9.0, 9.0) == 1.0  # at/below target
    assert t.normalize_size(9.5, 9.0, 9.0) == 0.0  # above a zero-width ramp


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
        reference={2160: (3.0, 6.0, 11.0)},
        min_closeness_gain=0.05,
    )
    rp = Topsis(cfg).resolve_profile("Special")
    assert rp.pick == "min_size"
    assert rp.reference == {2160: (3.0, 6.0, 11.0)}
    assert rp.min_closeness_gain == 0.05
    assert rp.weights == cfg.presets["Efficient"].weights  # inherited


def test_score_candidates_orders_by_closeness():
    t = _topsis()
    rp = t.resolve_profile("2160p Quality")
    good = _release(score=1_000_000, resolution=2160, size_gb=13.0)  # 6.5 GiB/h, in band
    weak = _release(score=950_000, resolution=1080, size_gb=14.0)  # lower res, 7 GiB/h
    scored, diag = t.score_candidates([weak, good], 2.0, rp, target_resolution=2160)
    assert scored[0][0] is good
    assert scored[0][2] >= scored[1][2]
    assert diag["input"] == 2


def test_select_max_score_for_remux():
    t = _topsis()
    rp = t.resolve_profile("2160p Remux")  # 2160 band 15..80
    big = _release(score=1_000_000, resolution=2160, size_gb=60.0)  # 30 GiB/h
    lean = _release(score=900_000, resolution=2160, size_gb=36.0)  # 18 GiB/h, in band
    scored, _ = t.score_candidates([lean, big], 2.0, rp, 2160)
    selected = t.select(scored, rp)
    assert selected is not None
    rel, _attrs, _clo = selected
    assert rel is big  # highest score wins regardless of size


def test_select_min_size_for_compact():
    t = _topsis()
    rp = t.resolve_profile("Compact")  # 2160 band 4..12
    small = _release(score=850_000, resolution=2160, size_gb=9.0)  # 4.5 GiB/h, in band
    bigger = _release(score=1_000_000, resolution=2160, size_gb=13.0)  # 6.5 GiB/h
    scored, _ = t.score_candidates([bigger, small], 2.0, rp, 2160)
    selected = t.select(scored, rp)
    assert selected is not None
    rel, _attrs, _clo = selected
    assert rel is small  # smallest wins


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
