#!/usr/bin/env python3
"""AEGIS cluster scheduler — runs on the FAU CIP hub, orchestrates the C-phase
re-run across the pool, maximizing parallel GPU usage. Jobs are sharded per
(experiment, seed): one seed per host (AEGIS_SEEDS env), so the whole 10-seed
suite runs in parallel.

- Poll every POLL=120 s; each cycle launch the next queued job on every FREE
  host (reachable, GPU >=6GB free, no local console login, not in cooldown).
- Local login or a running job dying without its done.flag -> cooldown host 1 h
  and requeue the job; retry after the hour.
- Each job writes done.flag "DONE <exit>" so genuine completion is distinct
  from an external kill. queue.json re-read each cycle (live append).
"""
import json, os, subprocess, time
from pathlib import Path

BASE = "/proj/ciptmp/up89uvox/aegis"
VENV = "/proj/ciptmp/up89uvox/my_project_venv"
CL = f"{BASE}/results/cluster"
SSHO = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "ProxyCommand=none"]
POLL = 120
COOLDOWN = 3600
GPU_FREE_MIN = 6000
QUEUE_F = f"{CL}/queue.json"
STATE_F = f"{CL}/state.json"
POOL_F = f"{CL}/host_inventory.txt"
LOG = open(f"{CL}/scheduler.log", "a", buffering=1)


def log(m):
    LOG.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}\n")


def ssh(host, cmd, timeout=15):
    try:
        r = subprocess.run(["ssh", *SSHO, host, cmd], capture_output=True,
                           text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception:
        return 255, "", "exc"


def load_pool():
    hosts = []
    try:
        for ln in open(POOL_F):
            ln = ln.strip()
            if not ln or "DOWN" in ln:
                continue
            h = ln.split("|")[0].strip().split()[0]
            if h.startswith("cip") and h != "cip7f0":
                hosts.append(h)
    except FileNotFoundError:
        pass
    return hosts


def load_queue():
    # entries: [label, script, args, seed]
    return json.load(open(QUEUE_F))


def probe(host):
    cmd = ("mf=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|head -1);"
           "loc=$(who 2>/dev/null|grep -cE '\\(:[0-9]|tty[0-9]|seat|login');"
           "echo ${mf:-X} ${loc:-0}")
    rc, out, _ = ssh(host, cmd, timeout=12)
    if rc != 0 or not out:
        return "DOWN"
    p = out.split()
    try:
        mf = int(float(p[0])); loc = int(p[1])
    except Exception:
        return "DOWN"
    if loc > 0:
        return "LOCAL"
    return "FREE" if mf >= GPU_FREE_MIN else "BUSY"


def launch(label, script, args, seed, host):
    cmd = (f"nohup {BASE}/scripts/run_job.sh {json.dumps(label)} {json.dumps(script)} "
           f"{json.dumps(args)} {json.dumps(seed)} >/dev/null 2>&1 </dev/null & echo $!")
    rc, out, _ = ssh(host, cmd)
    return out.strip() or "?"


def cleanup_pool(pool):
    log(f"startup cleanup: killing stale exp jobs on {len(pool)} hosts")
    procs = [subprocess.Popen(
        ["ssh", *SSHO, h, "pkill -u up89uvox -f 'aegis/scripts/exp_' 2>/dev/null; true"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for h in pool]
    for p in procs:
        try:
            p.wait(timeout=12)
        except Exception:
            p.kill()


def main():
    pool = load_pool()
    log(f"=== scheduler RESUME | pool={len(pool)} hosts ===")
    # RESUME-safe init: preserve existing state.json, then override each job from
    # its done.flag (ground truth on the shared NFS). Jobs still 'running' are
    # kept so the main loop re-verifies their liveness. Do NOT cleanup_pool here
    # -- that would kill the jobs that are still legitimately running.
    try:
        base = json.load(open(STATE_F))
    except Exception:
        base = {}
    state = {}
    for lab, sc, ar, sd in load_queue():
        j = dict(base.get(lab, {}))
        j.update(script=sc, args=ar, seed=sd)
        j.setdefault("host", None)
        j.setdefault("pid", None)
        code = None
        try:
            parts = open(f"{CL}/{lab}/done.flag").read().split()
            code = parts[1] if len(parts) > 1 else None
        except Exception:
            code = None
        if code is not None:
            j["status"] = "done" if code == "0" else "failed"
        elif j.get("status") != "running":
            j["status"] = "queued"
        state[lab] = j
    rc = {s: sum(1 for v in state.values() if v["status"] == s)
          for s in ("queued", "running", "done", "failed")}
    log(f"resume state: {rc}")
    cooldown = {}

    while True:
        now = time.time()
        try:
            for lab, sc, ar, sd in load_queue():
                if lab not in state:
                    state[lab] = {"status": "queued", "host": None, "pid": None,
                                  "script": sc, "args": ar, "seed": sd}
                    log(f"queue: new job {lab}")
        except Exception:
            pass

        for lab, j in state.items():
            if j["status"] != "running":
                continue
            h, D = j["host"], f"{CL}/{lab}"
            rc, out, _ = ssh(h, f"if [ -f {D}/done.flag ]; then cat {D}/done.flag; "
                                f"elif pgrep -u up89uvox -f 'run_job.sh {lab} ' >/dev/null 2>&1; then echo ALIVE; "
                                f"else echo GONE; fi", timeout=12)
            if out.startswith("DONE"):
                code = (out.split() + ["?"])[1]
                j["status"] = "done" if code == "0" else "failed"
                log(f"{lab} on {h} -> {j['status']} (exit {code})")
            elif out == "GONE":
                log(f"{lab} on {h} KILLED (no flag) -> requeue; cooldown {h} 1h")
                j.update(status="queued", host=None, pid=None)
                cooldown[h] = now + COOLDOWN
            elif rc != 0:
                log(f"{lab} host {h} unreachable -> requeue; cooldown {h} 1h")
                j.update(status="queued", host=None, pid=None)
                cooldown[h] = now + COOLDOWN

        c = {s: sum(1 for j in state.values() if j["status"] == s)
             for s in ("queued", "running", "done", "failed")}
        log(f"status queued={c['queued']} running={c['running']} done={c['done']} "
            f"failed={c['failed']} cooldown={sum(1 for t in cooldown.values() if t>now)}")
        json.dump(state, open(STATE_F, "w"), indent=1)

        if c["queued"] == 0 and c["running"] == 0:
            log("=== ALL JOBS COMPLETE ===")
            break

        if c["queued"] > 0:
            running_hosts = {j["host"] for j in state.values() if j["status"] == "running"}
            for h in pool:
                if not any(j["status"] == "queued" for j in state.values()):
                    break
                if h in running_hosts or cooldown.get(h, 0) > now:
                    continue
                st = probe(h)
                if st == "FREE":
                    lab = next(l for l, j in state.items() if j["status"] == "queued")
                    j = state[lab]
                    pid = launch(lab, j["script"], j["args"], j["seed"], h)
                    j.update(status="running", host=h, pid=pid)
                    running_hosts.add(h)
                    log(f"launched {lab} (seed {j['seed']}) -> {h}")
                elif st == "LOCAL":
                    cooldown[h] = now + COOLDOWN
                    log(f"{h} LOCAL login -> cooldown 1h")

        time.sleep(POLL)

    json.dump(state, open(STATE_F, "w"), indent=1)
    log("scheduler exit")


if __name__ == "__main__":
    main()
