"""Seul lanceur autorisé pour une tâche GPU ponctuelle (évaluation, expérience) sous jeton VertIA.

python gpu_task.py --gb 3 --minutes 30 -- <commande>
python gpu_task.py --gb 3 --minutes 30 --pause-369m -- <commande>

Cycle : (option) arrêt du 369M juste après une écriture d'état, release de son jeton ;
acquire creation_llm --wait ; contrôle du détenteur et de l'absence du 369M ; commande ;
release ; si la commande sort en PAUSE_CODE, nouvel acquire puis relance ;
(option) relance du 369M, qui redemande lui-même le jeton.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from vram_guard import LOCK, PAUSE_CODE, SESSION, token_holder

ROOT = Path(__file__).parent
STATE_369M = ROOT / "checkpoints_369m" / "etat.json"
MATCH_369M = ("run_369m.py", "checkpoints_369m")
MAX_SHORT_MIN = 45
WAIT_SAVE_S = 1800


def lock(*args):
    r = subprocess.run([sys.executable, str(LOCK), *args], capture_output=True, text=True)
    print(f"vram_lock {args[0]} : code {r.returncode} {r.stdout.strip()}", flush=True)
    return r.returncode


def pids_369m():
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*creation_llm*' -and ("
        + " -or ".join(f"$_.CommandLine -like '*{m}*'" for m in MATCH_369M)
        + ") } | ForEach-Object { $_.ProcessId }"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def kill(pids):
    for pid in pids:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)


def wait_state_write(path=STATE_369M, timeout=WAIT_SAVE_S, poll=5):
    start = path.stat().st_mtime if path.exists() else 0
    end = time.time() + timeout
    while time.time() < end:
        if path.exists() and path.stat().st_mtime != start:
            time.sleep(3)
            return json.loads(path.read_text(encoding="utf-8"))
        time.sleep(poll)
    return None


def pause_369m():
    if not pids_369m():
        return False
    print("369M en cours : attente de sa prochaine écriture d'état", flush=True)
    state = wait_state_write()
    if state is None:
        raise SystemExit("aucune écriture d'état du 369M dans le délai : abandon, GPU non touché")
    print(f"état 369M : {state}", flush=True)
    kill(pids_369m())
    time.sleep(5)
    if pids_369m():
        raise SystemExit("processus du 369M toujours présents : abandon")
    if token_holder() == SESSION:
        code = lock("release", SESSION)
        if code != 0:
            raise SystemExit(f"release du jeton du 369M : code {code}, abandon")
    return True


def run_watched(cmd, poll=10):
    proc = subprocess.Popen(cmd, cwd=ROOT)
    while proc.poll() is None:
        time.sleep(poll)
        intruders = pids_369m()
        if intruders:
            print(f"369M relancé pendant la tâche (PID {intruders}) : arrêté, il sera relancé à la fin", flush=True)
            kill(intruders)
    return proc.returncode


def relaunch_369m():
    subprocess.Popen(
        [sys.executable, "run_369m.py"], cwd=ROOT,
        stdout=open(ROOT / "run_369m_reprise.out", "a"), stderr=open(ROOT / "run_369m_reprise.err", "a"),
        creationflags=0x00000008 | 0x00000200,
    )
    print("369M relancé", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gb", type=float, required=True)
    p.add_argument("--minutes", type=int, required=True)
    p.add_argument("--note", default="")
    p.add_argument("--pause-369m", action="store_true")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        p.error("commande manquante")
    if pids_369m() and not a.pause_369m:
        raise SystemExit("le 369M tourne : relancer avec --pause-369m ou attendre")
    paused = pause_369m() if a.pause_369m else False
    rc = 1
    try:
        args = ["acquire", SESSION, "--gb", str(a.gb), "--minutes", str(a.minutes), "--wait", "--note", a.note[:140]]
        if a.minutes > MAX_SHORT_MIN:
            args.append("--long")
        while True:
            code = lock(*args)
            if code != 0:
                print(f"ESCALADE : acquire a renvoyé {code}", flush=True)
                return code
            if token_holder() != SESSION or pids_369m():
                lock("release", SESSION)
                raise SystemExit("contrôle avant lancement échoué (jeton ou 369M) : commande non lancée")
            rc = run_watched(cmd)
            print(f"commande terminée : code {rc}", flush=True)
            code = lock("release", SESSION)
            if code == 5:
                print("ESCALADE : VRAM non libérée après la commande", flush=True)
                return code
            if rc != PAUSE_CODE:
                break
            print("pause demandée : jeton rendu, attente d'un nouveau jeton", flush=True)
    finally:
        if paused:
            relaunch_369m()
    return rc


if __name__ == "__main__":
    sys.exit(main())
