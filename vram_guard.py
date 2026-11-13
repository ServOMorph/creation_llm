"""Jeton GPU VertIA (protocole version 1) pour les entraînements longs interruptibles.

Usage : python vram_guard.py <session> --gb <Go> -- <commande d'entraînement avec --resume>
Cycle : acquire --long --wait, lancement ; code PAUSE_CODE (pause demandée ou jeton à renouveler)
-> release, nouvel acquire --wait, relance ; autre code -> release et fin.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

LOCK = Path(r"D:\ServOMorph\VertIA_v3\_orchestration\vram_lock.py")
PAUSE_CODE = 75
ACQUIRE_TIMEOUT_MIN = 240


def lock_cmd(*args):
    return subprocess.run([sys.executable, str(LOCK), *args], capture_output=True, text=True).returncode


def should_pause(session):
    return lock_cmd("should-pause", session) == 0


def release(session):
    return lock_cmd("release", session)


def acquire(session, gb, minutes, priority=5):
    return lock_cmd(
        "acquire", session, "--gb", str(gb), "--minutes", str(minutes), "--long", "--wait",
        "--priority", str(priority), "--timeout", str(ACQUIRE_TIMEOUT_MIN),
    )


class TrainingPaused(Exception):
    pass


class PauseMonitor:
    def __init__(self, session, check_s, token_hours, clock=time.time, probe=None):
        self.session = session
        self.check_s = check_s
        self.token_s = token_hours * 3600
        self.clock = clock
        self.probe = probe or (lambda s: should_pause(s))
        self.start = self.last = clock()

    def reason(self):
        now = self.clock()
        if self.token_s > 0 and now - self.start >= self.token_s:
            return "renouvellement_jeton"
        if now - self.last < self.check_s:
            return None
        self.last = now
        return "pause_demandee" if self.probe(self.session) else None


def write_state(path, step, reason):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"step": step, "raison": reason, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "reprise": "relancer la même commande avec --resume"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def run_guarded(session, gb, minutes, cmd, priority=5, max_cycles=0, run=subprocess.run, acq=acquire,
                rel=release, log=print):
    cycles = 0
    while True:
        code = acq(session, gb, minutes, priority)
        if code != 0:
            log(f"vram_guard : acquire a renvoyé {code}, ESCALADE requise")
            return code
        rc = run(cmd).returncode
        cycles += 1
        rcode = rel(session)
        if rcode != 0:
            log(f"vram_guard : release a renvoyé {rcode} (VRAM non libérée ou jeton perdu)")
            return rcode
        if rc != PAUSE_CODE:
            log(f"vram_guard : entraînement terminé, code {rc}")
            return rc
        if max_cycles and cycles >= max_cycles:
            log(f"vram_guard : {cycles} cycle(s) effectué(s), arrêt avec GPU libéré")
            return rc
        log("vram_guard : pause, attente d'un nouveau jeton")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--gb", type=float, required=True)
    p.add_argument("--minutes", type=int, default=1440)
    p.add_argument("--priority", type=int, default=5)
    p.add_argument("--cycles", type=int, default=0)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        p.error("commande d'entraînement manquante")
    return run_guarded(a.session, a.gb, a.minutes, cmd, a.priority, a.cycles)


if __name__ == "__main__":
    sys.exit(main())
