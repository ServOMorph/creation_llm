import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gpu_task  # noqa: E402
from vram_guard import PAUSE_CODE, SESSION  # noqa: E402


def setup(monkeypatch, codes):
    events = []
    monkeypatch.setattr(gpu_task, "lock", lambda *a: events.append(a[0]) or 0)
    monkeypatch.setattr(gpu_task, "token_holder", lambda: SESSION)
    monkeypatch.setattr(gpu_task, "pids_369m", lambda: [])
    monkeypatch.setattr(gpu_task, "run_watched", lambda cmd: events.append("run") or codes.pop(0))
    return events


def test_relaunches_after_pause(monkeypatch):
    events = setup(monkeypatch, [PAUSE_CODE, PAUSE_CODE, 0])
    assert gpu_task.main(["--gb", "6.5", "--minutes", "1440", "--", "x"]) == 0
    assert events == ["acquire", "run", "release"] * 3


def test_single_run_without_pause(monkeypatch):
    events = setup(monkeypatch, [3])
    assert gpu_task.main(["--gb", "3", "--minutes", "30", "--", "x"]) == 3
    assert events == ["acquire", "run", "release"]
