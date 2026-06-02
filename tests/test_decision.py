from optimizarr.features.optimizer.config import default_topsis
from optimizarr.features.optimizer.decision import decide, format_decision
from optimizarr.features.optimizer.topsis import GB, Topsis


def _topsis() -> Topsis:
    return Topsis(default_topsis())


def _release(guid="g1", score=1_000_000, resolution=2160, size_gb=13.0):
    return {
        "guid": guid,
        "indexerId": 1,
        "title": f"Movie.{resolution}p",
        "customFormatScore": score,
        "quality": {"quality": {"resolution": resolution}},
        "size": int(size_gb * GB),
        "rejections": [],
    }


def _file(score=200_000, resolution=1080, size_gb=30.0):
    return {
        "id": 555,
        "customFormatScore": score,
        "size": int(size_gb * GB),
        "quality": {"quality": {"resolution": resolution}},
    }


def test_format_decision_act_shows_current_and_pick():
    releases = [_release(score=1_000_000, resolution=2160, size_gb=13.0)]
    d = decide(
        _topsis(),
        releases,
        2.0,
        "2160p Quality",
        2160,
        current_file=_file(score=200_000, resolution=1080),
    )
    msg = format_decision("radarr", "Movie (2024)", d, dry_run=True)
    assert "would GRAB" in msg
    assert "current:" in msg and "pick:" in msg
    assert "profile=2160p Quality" in msg
    assert "Δsize" in msg and "Δcloseness" in msg


def test_format_decision_hold_when_nothing_better():
    # Current already at the candidate's exact spec -> no legal transition -> HOLD.
    releases = [_release(score=1_000_000, resolution=2160, size_gb=13.0)]
    current = _file(score=1_000_000, resolution=2160, size_gb=13.0)
    d = decide(_topsis(), releases, 2.0, "2160p Quality", 2160, current_file=current)
    assert d.action == "HOLD"
    msg = format_decision("radarr", "Movie (2024)", d, dry_run=False)
    assert "HOLD" in msg
    assert "nothing better" in msg


def test_decide_hold_when_no_candidates():
    d = decide(_topsis(), [], 2.0, None, None, current_file=_file())
    assert d.action == "HOLD"
    assert "no viable candidate" in d.reason


def test_decide_act_on_clear_upgrade():
    # Current is a bloated 1080p low-score file; candidate is a clean 2160p high-score (res up).
    releases = [_release(score=1_000_000, resolution=2160, size_gb=13.0)]
    d = decide(
        _topsis(),
        releases,
        2.0,
        "2160p Quality",
        2160,
        current_file=_file(score=200_000, resolution=1080),
    )
    assert d.action == "ACT"
    assert d.release is not None and d.release["guid"] == "g1"


def test_decide_act_smaller_at_equal_score():
    # Same res + score, meaningfully smaller -> a free size win for any profile.
    current = _file(score=900_000, resolution=2160, size_gb=24.0)  # 12 GiB/h
    smaller = _release(guid="lean", score=900_000, resolution=2160, size_gb=13.0)  # 6.5 GiB/h
    d = decide(_topsis(), [smaller], 2.0, "2160p Efficient", 2160, current_file=current)
    assert d.action == "ACT"
    assert d.release is not None and d.release["guid"] == "lean"


def test_decide_hold_on_bigger_file_without_score_gain():
    # Bigger at same res + same score must never be grabbed.
    current = _file(score=900_000, resolution=2160, size_gb=13.0)
    bigger = _release(guid="big", score=900_000, resolution=2160, size_gb=24.0)
    d = decide(_topsis(), [bigger], 2.0, "2160p Efficient", 2160, current_file=current)
    assert d.action == "HOLD"
    assert d.reason == "nothing better"


def test_decide_remux_refuses_lower_score_efficient_takes_it():
    # Lower score, smaller file: Efficient (size-leaning) gains closeness and takes it; Remux
    # (score-dominated) does not, and the lean encode is also below Remux's size floor.
    current = _file(score=900_000, resolution=2160, size_gb=20.0)  # 10 GiB/h
    leaner = _release(guid="lean", score=850_000, resolution=2160, size_gb=9.0)  # 4.5 GiB/h
    d_eff = decide(_topsis(), [leaner], 2.0, "2160p Efficient", 2160, current_file=current)
    assert d_eff.action == "ACT"
    d_remux = decide(_topsis(), [leaner], 2.0, "2160p Remux", 2160, current_file=current)
    assert d_remux.action == "HOLD"


def test_decide_compact_picks_smallest_legal():
    current = _file(score=900_000, resolution=2160, size_gb=24.0)  # 12 GiB/h
    a = _release(guid="a", score=900_000, resolution=2160, size_gb=13.0)  # 6.5 GiB/h
    b = _release(guid="b", score=880_000, resolution=2160, size_gb=8.0)  # 4 GiB/h, slightly lower
    d = decide(_topsis(), [a, b], 2.0, "Compact", 2160, current_file=current)
    assert d.action == "ACT"
    assert d.release is not None and d.release["guid"] == "b"  # min_size among legal survivors


def test_decide_drops_bigger_releases_when_size_increase_disallowed():
    current = _file(score=400_000, resolution=2160, size_gb=20.0)
    bigger = _release(guid="big", score=1_000_000, resolution=2160, size_gb=30.0)
    smaller = _release(guid="small", score=900_000, resolution=2160, size_gb=13.0)
    d = decide(
        _topsis(),
        [bigger, smaller],
        2.0,
        "2160p Quality",
        2160,
        current_file=current,
        allow_size_increase=False,
    )
    assert d.action == "ACT"
    assert d.release is not None and d.release["guid"] == "small"


def test_decide_drops_lower_score_releases_when_downgrade_disallowed():
    current = _file(score=800_000, resolution=2160, size_gb=28.0)
    higher = _release(guid="hi", score=1_000_000, resolution=2160, size_gb=22.0)
    lower = _release(guid="lo", score=700_000, resolution=2160, size_gb=10.0)
    d = decide(
        _topsis(),
        [higher, lower],
        2.0,
        "2160p Quality",
        2160,
        current_file=current,
        allow_quality_downgrade=False,
    )
    assert d.action == "ACT"
    assert d.release is not None and d.release["guid"] == "hi"


def test_decide_hold_when_current_already_good():
    releases = [_release(score=1_000_000, resolution=2160, size_gb=13.0)]
    current = _file(score=1_000_000, resolution=2160, size_gb=13.0)
    d = decide(_topsis(), releases, 2.0, "2160p Quality", 2160, current_file=current)
    assert d.action == "HOLD"


def test_decide_scope_4k_not_phantom_upgraded_to_same_class():
    # Current is a 2.39:1 scope 4K file: its quality block says 2160p, even though the mediaInfo
    # pixel height is 1608. A same-score 2160p candidate that is slightly BIGGER must NOT be
    # grabbed: the current file's own quality (2160) makes resolution equal, so there is no
    # closeness gain. (Regression: reading the 1608 pixel height looked like a 1608->2160 upgrade
    # and justified the swap.)
    current = {
        "id": 7,
        "customFormatScore": 924_600,
        "size": int(12.4 * GB),  # 6.2 GiB/h at 2h
        "quality": {"quality": {"resolution": 2160}},
        "mediaInfo": {"resolution": "3840x1608"},
    }
    bigger_same_score = _release(guid="scope", score=924_600, resolution=2160, size_gb=12.8)
    d = decide(_topsis(), [bigger_same_score], 2.0, "2160p Quality", 2160, current_file=current)
    assert d.action == "HOLD"


def test_decide_unknown_current_score_treated_as_upgrade():
    # current file with no customFormatScore -> any viable candidate is an improvement.
    current = {"id": 9, "size": int(30 * GB), "mediaInfo": {"resolution": "3840x2160"}}
    releases = [_release(score=900_000, resolution=2160, size_gb=13.0)]
    d = decide(_topsis(), releases, 2.0, "2160p Quality", 2160, current_file=current)
    assert d.action == "ACT"
