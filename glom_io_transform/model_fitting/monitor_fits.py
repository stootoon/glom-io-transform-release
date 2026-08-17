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

def squeue_jobs(name_filter=None):
    """{job_id: state} from squeue. None if squeue isn't available here."""
    try:
        out = subprocess.run(["squeue", "--me", "--noheader", "--format=%i|%T|%j"],
                             capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    jobs = {}
    for line in out.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        job_id, state, jobname = (p.strip() for p in parts[:3])
        if name_filter and name_filter not in jobname:
            continue
        # Array tasks appear as 1234_5; key on both so either form matches.
        jobs[job_id] = state
        jobs[job_id.split("_")[0]] = state
    return jobs


# ----------------------------------------------------------------------------
# fit directories
# ----------------------------------------------------------------------------

def job_states(jobs):
    """Counter of slurm states. jobs is keyed on both '1234_5' and '1234', so
    count each job once by ignoring a base key that has array tasks."""
    if not jobs:
        return Counter()
    return Counter(state for jid, state in jobs.items()
                   if "_" in jid or not any(k.startswith(jid + "_") for k in jobs))


def fit_dirs(root):
    """Directories holding in.N.p files, i.e. one per model per split."""
    found = set()
    for path in glob.glob(os.path.join(root, "**", "in.*.p"), recursive=True):
        found.add(os.path.dirname(path))
    return sorted(found)


def dir_progress(d):
    """(n_configs, n_done, [paths of the in.N.p files with no output yet])"""
    ins  = {os.path.basename(p)[3:-2]: p for p in glob.glob(os.path.join(d, "in.*.p"))}
    outs = {os.path.basename(p)[4:-2]    for p in glob.glob(os.path.join(d, "out.*.p"))}
    todo = sorted(set(ins) - outs, key=lambda s: int(s) if s.isdigit() else s)
    return len(ins), len(set(ins) & outs), [ins[k] for k in todo]


# ----------------------------------------------------------------------------
# slurm logs
# ----------------------------------------------------------------------------

ERROR_RE = re.compile(r"^(Traceback|\S*Error:|slurmstepd:|srun:.*error)", re.M)
JOBID_RE = re.compile(r"slurm-(\d+(?:_\d+)?)\.out$")


def log_job_id(path):
    """Job id from the slurm-JOBID.out filename, or None if it doesn't match."""
    m = JOBID_RE.search(os.path.basename(path))
    return m.group(1) if m else None


def scan_logs(patterns, jobs):
    """Classify each slurm log by content AND whether its job is still queued.

    jobs is {job_id: state} from squeue, or None if squeue isn't available.
    Returns a dict of category -> list of (path, detail).

      completed : contains ALLDONE
      failed    : contains a traceback or slurm error
      running   : no ALLDONE, and its job id is still in the queue
      died      : no ALLDONE, no error, and NOT in the queue -- i.e. the job
                  stopped without finishing (walltime, OOM, node failure), or
                  the queue could not be read to say otherwise
    """
    out = {"completed": [], "failed": [], "running": [], "died": [], "unknown": []}
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.expanduser(pattern))):
            real = os.path.realpath(path)
            if real in seen:      # the default globs overlap
                continue
            seen.add(real)
            try:
                with open(path, "r", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            job_id = log_job_id(path)
            if "ALLDONE" in text:
                out["completed"].append((path, ""))
            elif ERROR_RE.search(text):
                out["failed"].append((path, first_error(text)))
            elif jobs is None or job_id is None:
                # Can't tell whether it is still running.
                out["unknown"].append((path, "job state unknown"))
            elif job_id in jobs:
                out["running"].append((path, jobs[job_id].lower()))
            else:
                out["died"].append((path, "no ALLDONE and not in the queue"))
    return out


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

    jobs = squeue_jobs(args.job_name)
    cats = scan_logs(args.logs, jobs)

    print(f"{BOLD}fits{OFF}  {root}")
    if not dirs:
        print(f"  {DIM}no directories with in.*.p found{OFF}")
    total_in = total_done = 0
    outstanding = []
    rows = []
    for d in dirs:
        n_in, n_done, todo = dir_progress(d)
        total_in += n_in
        total_done += n_done
        outstanding += todo
        rows.append((os.path.relpath(d, root), n_in, n_done, todo))

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

        # Fits and jobs are different units -- one job usually covers many
        # configs -- so report them on separate lines rather than mixing them.
        states = job_states(jobs)
        n_run  = states.get("RUNNING", 0)
        n_pend = sum(k for s, k in states.items() if s != "RUNNING")
        print(f"\n  {BOLD}fits{OFF}  {GRN}{total_done} done{OFF} · "
              f"{DIM}{len(outstanding)} to go{OFF}")
        if jobs is None:
            print(f"  {BOLD}jobs{OFF}  {DIM}squeue not available here{OFF}")
        else:
            print(f"  {BOLD}jobs{OFF}  {YEL}{n_run} running{OFF} · "
                  f"{DIM}{n_pend} queued{OFF}")

    print(f"\n{BOLD}queue{OFF}")
    if jobs is None:
        print(f"  {DIM}squeue not available here{OFF}")
    elif not jobs:
        print(f"  {DIM}no jobs queued or running{OFF}")
    else:
        for state, k in sorted(job_states(jobs).items(), key=lambda kv: -kv[1]):
            colour = GRN if state == "RUNNING" else YEL if state == "PENDING" else RED
            print(f"  {colour}{state.lower():<12}{OFF} {k}")

    if any(v for k, v in cats.items() if k != "in_flight"):
        print(f"\n{BOLD}logs{OFF}  {', '.join(args.logs)}")
        w = 24
        for key, label, colour in [
                ("completed", "completed (ALLDONE)",      GRN),
                ("running",   "still running",            YEL),
                ("failed",    "failed (error in log)",    RED),
                ("died",      "stopped without ALLDONE",  RED),
                ("unknown",   "state unknown",            DIM)]:
            n = len(cats[key])
            if n or key in ("completed", "failed"):
                print(f"  {colour}{label:<{w}}{OFF} {n}")

        problems = cats["failed"] + cats["died"]
        if problems and args.failed:
            print(f"\n{BOLD}needs attention{OFF}")
            for path, detail in problems[:args.max_failed]:
                print(f"  {os.path.basename(path):<22} {DIM}{detail}{OFF}")
            if len(problems) > args.max_failed:
                print(f"  {DIM}... and {len(problems)-args.max_failed} more{OFF}")
        elif problems:
            print(f"  {DIM}re-run with --failed to list the {len(problems)} needing attention{OFF}")

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
