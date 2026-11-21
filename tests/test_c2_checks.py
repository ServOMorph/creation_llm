import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rd.experiences import c2_checks  # noqa: E402


def test_masks_all_pass():
    res = c2_checks.check_masks()
    assert all(res.values()), res


def test_resume_is_identical():
    assert c2_checks.check_resume(optimizer="adamw")["reprise_identique"]


def test_schedule_all_pass():
    res = c2_checks.check_schedule()
    assert all(res.values()), res


def test_sampling_with_replacement_covers_about_63_percent():
    res = c2_checks.simulate_sampling(n_windows=5000)
    assert 0.60 < res["couverture_avec_remise_1x"] < 0.66
