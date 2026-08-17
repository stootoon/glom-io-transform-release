"""Monitor the progress of fitting jobs.

Two independent signals are combined:

  - the fits themselves: driver.py writes one out.N.p for every in.N.p, so
    counting them gives progress that does not depend on slurm at all;
  - the slurm logs: a run that finished cleanly ends with ALLDONE, so a log
    without it (for a job no longer in the queue) means the run died.

Run:
    python glom_io_transform/model_fitting/monitor_fits.py
    python -m glom_io_transform.model_fitting.monitor_fits --watch 60
    python ... monitor_fits.py --logs '~/slurm-*.out' --failed
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

BOLD, DIM, RED, GRN, YEL, OFF = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"


def plain(s):
    return re.sub(r"\033\[[0-9;]*m", "", s)


# ----------------------------------------------------------------------------
# slurm queue
# ----------------------------------------------------------------------------

def squeue_states(name_filter=None):
    """Job states from squeue, as a Counter. Empty if squeue isn't available."""
    try:
        out = subprocess.run(["squeue", "--me", "--noheader", "--format=%T|%j"],
                             capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    states = Counter()
    for line in out.stdout.splitlines():
        if "|" not in line:
            continue
        state, jobname = line.split("|", 1)
        if name_filter and name_filter not in jobname:
            continue
        states[state.strip()] += 1
    return states


# ----------------------------------------------------------------------------
# fit directories
# ----------------------------------------------------------------------------

def fit_dirs(root):
    """Directories holding in.N.p files, i.e. one per model per split."""
    found = set()
    for path in glob.glob(os.path.join(root, "**", "in.*.p"), recursive=True):
        found.add(os.path.dirname(path))
    return sorted(found)


def dir_progress(d):
    n_in  = len(glob.glob(os.path.join(d, "in.*.p")))
    ins   = {os.path.basename(p)[3:-2] for p in glob.glob(os.path.join(d, "in.*.p"))}
    outs  = {os.path.basename(p)[4:-2] for p in glob.glob(os.path.join(d, "out.*.p"))}
    return n_in, len(ins & outs), sorted(ins - outs, key=lambda s: int(s) if s.isdigit() else s)


# ----------------------------------------------------------------------------
# slurm logs
# ----------------------------------------------------------------------------

ERROR_RE = re.compile(r"^(Traceback|\S*Error:|slurmstepd:|srun:.*error)", re.M)

def scan_logs(patterns):
    """Classify each slurm log as done / failed / running-or-truncated."""
    done, failed, unfinished = [], [], []
    for pattern in patterns:
        for path in glob.glob(os.path.expanduser(pattern)):
            try:
                with open(path, "r", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            if "ALLDONE" in text:
                done.append(path)
            elif ERROR_RE.search(text):
                failed.append((path, first_error(text)))
            else:
                unfinished.append(path)
    return done, failed, unfinished


def first_error(text):
    for line in text.splitlines():
        if ERROR_RE.match(line):
            return line.strip()[:100]
    return "?"


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------

def bar(frac, width=22):
    filled = int(round(frac * width))
    return "█" * filled + "·" * (width - filled)


def report(args):
    root = os.path.expanduser(args.fits_root)
    dirs = fit_dirs(root)

    print(f"{BOLD}fits{OFF}  {root}")
    if not dirs:
        print(f"  {DIM}no directories with in.*.p found{OFF}")
    total_in = total_done = 0
    rows = []
    for d in dirs:
        n_in, n_done, missing = dir_progress(d)
        total_in += n_in
        total_done += n_done
        rows.append((os.path.relpath(d, root), n_in, n_done, missing))

    width = max((len(r[0]) for r in rows), default=10)
    for name, n_in, n_done, missing in rows:
        frac = n_done / n_in if n_in else 0
        colour = GRN if n_done == n_in else (YEL if n_done else DIM)
        print(f"  {name:<{width}}  {colour}{bar(frac)}{OFF} "
              f"{n_done:>4}/{n_in:<4} {DIM}({100*frac:5.1f}%){OFF}")
    if rows:
        frac = total_done / total_in if total_in else 0
        print(f"  {BOLD}{'total':<{width}}{OFF}  {bar(frac)} {total_done:>4}/{total_in:<4} "
              f"{DIM}({100*frac:5.1f}%){OFF}")

    states = squeue_states(args.job_name)
    print(f"\n{BOLD}queue{OFF}")
    if states is None:
        print(f"  {DIM}squeue not available here{OFF}")
    elif not states:
        print(f"  {DIM}no jobs queued or running{OFF}")
    else:
        for state, k in sorted(states.items(), key=lambda kv: -kv[1]):
            colour = GRN if state == "RUNNING" else YEL if state == "PENDING" else RED
            print(f"  {colour}{state.lower():<12}{OFF} {k}")

    done, failed, unfinished = scan_logs(args.logs)
    if done or failed or unfinished:
        n_running = sum(states.get(s, 0) for s in ("RUNNING", "PENDING")) if states else 0
        print(f"\n{BOLD}logs{OFF}  {', '.join(args.logs)}")
        w = 22
        print(f"  {GRN}{'completed (ALLDONE)':<{w}}{OFF} {len(done)}")
        print(f"  {RED}{'failed':<{w}}{OFF} {len(failed)}")
        label = "in progress" if n_running else "no ALLDONE, not queued"
        colour = YEL if n_running else RED
        print(f"  {colour}{label:<{w}}{OFF} {len(unfinished)}")
        if failed and args.failed:
            print(f"\n{BOLD}failures{OFF}")
            for path, err in failed[:args.max_failed]:
                print(f"  {os.path.basename(path)}: {DIM}{err}{OFF}")
            if len(failed) > args.max_failed:
                print(f"  {DIM}... and {len(failed)-args.max_failed} more{OFF}")
        elif failed:
            print(f"  {DIM}re-run with --failed to list them{OFF}")

    return total_done, total_in


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = None
    try:
        import glom_io_transform.paths as paths
        default_root = os.path.join(paths.fits_root, "fits")
    except Exception:
        pass
    p.add_argument("--fits-root", default=default_root or "fits",
                   help="root of the fits tree (default: <fits_root>/fits)")
    p.add_argument("--logs", nargs="*", default=["slurm-*.out", "*.out"],
                   help="glob(s) for slurm log files (default: slurm-*.out *.out)")
    p.add_argument("--job-name", default=None,
                   help="only count squeue jobs whose name contains this")
    p.add_argument("--failed", action="store_true", help="list the failing logs")
    p.add_argument("--max-failed", type=int, default=20)
    p.add_argument("--watch", type=int, metavar="SECONDS",
                   help="refresh every SECONDS until everything is done")
    args = p.parse_args(argv)

    if not args.watch:
        done, total = report(args)
        return 0 if (total and done == total) else 1

    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            print(f"{DIM}{time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"— refreshing every {args.watch}s, Ctrl-C to stop{OFF}\n")
            done, total = report(args)
            if total and done == total:
                print(f"\n{GRN}{BOLD}All {total} fits have output files.{OFF}")
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nstopped")
        return 1


if __name__ == "__main__":
    sys.exit(main())
