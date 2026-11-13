import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import vram_guard  # noqa: E402
from test_train import MCFG, make_cfg  # noqa: E402
from train import train  # noqa: E402
from vram_guard import PAUSE_CODE, PauseMonitor, TrainingPaused, run_guarded  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_monitor_probes_at_interval_only():
    clock, calls = Clock(), []
    m = PauseMonitor("creation_llm", 60, 0, clock=clock, probe=lambda s: calls.append(s) or False)
    clock.t = 30
    assert m.reason() is None and calls == []
    clock.t = 61
    assert m.reason() is None and calls == ["creation_llm"]


def test_monitor_reports_pause_and_token_renewal():
    clock = Clock()
    m = PauseMonitor("creation_llm", 10, 1, clock=clock, probe=lambda s: True)
    clock.t = 11
    assert m.reason() == "pause_demandee"
    clock.t = 3600
    assert m.reason() == "renouvellement_jeton"


def test_run_guarded_cycles_until_training_ends():
    codes, events = [PAUSE_CODE, 0], []

    def acq(*a):
        events.append("acquire")
        return 0

    def rel(s):
        events.append("release")
        return 0

    def run(cmd):
        events.append("run")
        return SimpleNamespace(returncode=codes.pop(0))

    rc = run_guarded("creation_llm", 6.5, 1440, ["x"], run=run, acq=acq, rel=rel, log=lambda m: None)
    assert rc == 0
    assert events == ["acquire", "run", "release"] * 2


def test_run_guarded_stops_on_failed_acquire_or_release():
    run = lambda cmd: SimpleNamespace(returncode=0)  # noqa: E731
    assert run_guarded("s", 1, 1, ["x"], run=run, acq=lambda *a: 3, rel=lambda s: 0, log=lambda m: None) == 3
    assert run_guarded("s", 1, 1, ["x"], run=run, acq=lambda *a: 0, rel=lambda s: 5, log=lambda m: None) == 5


def test_training_pauses_saves_and_resumes(tmp_path, monkeypatch):
    answers = iter([False, True])
    monkeypatch.setattr(vram_guard, "should_pause", lambda s: next(answers, False))
    cfg = make_cfg(tmp_path, max_steps=40, save_interval=100, vram_session="creation_llm", pause_check_s=0)
    with pytest.raises(TrainingPaused):
        train(cfg, MCFG)
    ckpt_dir = Path(cfg.ckpt_dir)
    etat = json.loads((ckpt_dir / "etat.json").read_text(encoding="utf-8"))
    assert etat["raison"] == "pause_demandee"
    assert torch.load(ckpt_dir / "ckpt.pt", map_location="cpu")["step"] == etat["step"] == 2

    resumed = make_cfg(tmp_path, max_steps=40, save_interval=100, resume=True)
    train(resumed, MCFG)
    assert torch.load(ckpt_dir / "ckpt.pt", map_location="cpu")["step"] == 40
    assert json.loads((ckpt_dir / "etat.json").read_text(encoding="utf-8"))["raison"] == "termine"


def test_timed_save(tmp_path):
    cfg = make_cfg(tmp_path, max_steps=5, save_interval=1000, save_minutes=1e-9)
    train(cfg, MCFG)
    assert json.loads((Path(cfg.ckpt_dir) / "etat.json").read_text(encoding="utf-8"))["step"] == 5


def test_run_guarded_stops_after_max_cycles():
    events = []
    run = lambda cmd: events.append("run") or SimpleNamespace(returncode=PAUSE_CODE)  # noqa: E731
    rc = run_guarded("s", 1, 1, ["x"], max_cycles=2, run=run, acq=lambda *a: 0,
                     rel=lambda s: events.append("release") or 0, log=lambda m: None)
    assert rc == PAUSE_CODE
    assert events == ["run", "release"] * 2
