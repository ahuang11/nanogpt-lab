"""The interface to the platform: launch experiments and read their results, from Python or `python src/lab.py <command>`."""
import argparse
import itertools
import json
import math
import re
import statistics
import time
import tomllib
from pathlib import Path
from typing import Any

from metaflow import Flow, Run, Task, namespace

from config import (
    DEFAULT_BRANCH, EVENT, INIT_LOSS, MEASUREMENT_FIELDS, PRESETS, UI_URL, Config, changes, config_hash, family_hash,
    load_text, resolve,
)

PROJECT = tomllib.loads((Path(__file__).parents[1] / "obproject.toml").read_text())["project"]
ERROR_LINE = re.compile(r"^[\w.]+(Error|Exception|Interrupt)\b")


def _tag(run: Run, key: str) -> str | None:
    return next((t.split(":", 1)[1] for t in run.user_tags if t.startswith(f"{key}:")), None)


def find_runs(config_hash: str | None = None, sweep: str | None = None, successful_only: bool = False) -> list[Run]:
    """TrainFlow runs in this project, newest first, across all users.

    Runs without a config tag predate tagging and are left out.
    """
    tags = [f"project:{PROJECT}"]
    if config_hash:
        tags.append(f"config:{config_hash}")
    if sweep:
        tags.append(f"sweep:{sweep}")
    namespace(None)
    runs = [r for r in Flow("TrainFlow").runs(*tags) if _tag(r, "config")]
    return [r for r in runs if r.successful] if successful_only else runs


def run_status(run: Run) -> tuple[str, Task | None]:
    """'succeeded', 'failed' (with the failed task) or 'running'. Divergence is decided in summarize()."""
    if run.successful:
        return "succeeded", None
    for step in run:
        for task in step:
            if task.finished and not task.successful:
                return "failed", task
    return "running", None


def launch(overrides: dict[str, Any] | None = None, label: str = "", sweep: str = "", launched_by: str = "lab",
           branch: str = DEFAULT_BRANCH, frozen: tuple[str, ...] = ()) -> dict[str, Any]:
    """Ask the deployed TrainFlow on `branch` for one run, unless the config already succeeded or is running.

    Returns {config_hash, launched, existing}, where existing is the run that made this launch unnecessary.
    """
    overrides = dict(overrides or {})
    cfg = resolve(overrides, frozen=frozen)
    h = config_hash(cfg)
    for run in find_runs(config_hash=h):
        if run_status(run)[0] in ("succeeded", "running"):
            return {"config_hash": h, "launched": False, "existing": run.pathspec}
    # Deferred: only launching needs obproject, so the compare app and other read-only users don't.
    from obproject import ProjectEvent

    ProjectEvent(EVENT, project=PROJECT, branch=branch).publish(payload={
        "overrides": json.dumps(overrides),
        "label": label or changes(cfg),
        "sweep": sweep,
        "launched_by": launched_by,
    }, ignore_errors=False)  # the default swallows a failed publish, which would report a launch that never happened
    return {"config_hash": h, "launched": True, "existing": None}


def sweep(name: str, presets: list[str] | None = None, grid: dict[str, list[Any]] | None = None,
          branch: str = DEFAULT_BRANCH) -> list[dict[str, Any]]:
    """Launch presets and/or every combination in a grid as one sweep; with neither, all presets.

    Everything is validated before anything is sent.
    """
    requests = []
    if grid:
        keys = sorted(grid)
        requests += [("", dict(zip(keys, values))) for values in itertools.product(*(grid[k] for k in keys))]
    if presets or not grid:
        names = presets or list(PRESETS)
        unknown = sorted(set(names) - set(PRESETS))
        if unknown:
            raise ValueError(f"unknown presets {unknown}; have {sorted(PRESETS)}")
        requests += [(n, PRESETS[n]) for n in names]
    for _, overrides in requests:
        resolve(overrides)
    return [launch(o, label=label, sweep=name, launched_by="lab.sweep", branch=branch) for label, o in requests]


def replicate(config_hash: str, seeds: tuple[int, ...] = (43, 44), branch: str = DEFAULT_BRANCH) -> list[dict[str, Any]]:
    """Rerun a succeeded config with other seeds, into the same sweep.

    It's the only way to change the seed from the CLI, so replicates use a fixed,
    shared set of seeds instead of whichever one happened to score best.
    """
    runs = find_runs(config_hash=config_hash, successful_only=True)
    if not runs:
        raise ValueError(f"no succeeded run with config {config_hash}")
    start = runs[0]["start"].task.data
    overrides = {k: v for k, v in json.loads(start.overrides).items() if k != "seed"}
    return [launch({**overrides, "seed": s}, label=f"{start.exp_name} seed={s}", sweep=_tag(runs[0], "sweep") or "",
                   launched_by="lab.replicate", branch=branch) for s in seeds]


def error_signature(error: str) -> str:
    """Error text with numbers removed, so the same bug under different configs compares equal."""
    return re.sub(r"\d+", "N", error)[:160]


def _error(task: Task) -> str:
    lines = [line.strip() for line in task.stderr.splitlines() if line.strip()]
    found = next((line for line in reversed(lines) if ERROR_LINE.match(line)), None)
    if found:
        return found[:300]
    return lines[-1][:300] if lines else "failed with no error output (killed pod or infrastructure)"


def curve_stats(history: list[dict[str, float]], points: int | None = 8) -> dict[str, Any]:
    """Losses, health flags and a downsampled [iter, train, val] curve from a run's eval history.

    Flags: still_improving (under-trained), overfitting (val rose off its best while
    train kept falling), and non_finite_loss / not_learning / loss_spike, which
    summarize() reports as divergence.
    """
    if not history:
        return {"flags": ["no_history"]}
    vals = [r["val_loss"] for r in history]
    trains = [r["train_loss"] for r in history]
    if not all(math.isfinite(x) for x in vals + trains):
        return {"final_val_loss": vals[-1], "best_val_loss": None, "flags": ["non_finite_loss"]}

    best = min(vals)
    quarter_ago = vals[-min(len(vals), max(2, len(vals) // 4))]
    stats = {
        "final_val_loss": vals[-1],
        "best_val_loss": best,
        "final_train_loss": round(trains[-1], 4),
        "train_val_gap": round(vals[-1] - trains[-1], 4),
        "best_val_iter": history[vals.index(best)]["iter"],
        "val_drop_last_quarter": round(quarter_ago - vals[-1], 4),
    }
    flags = []
    if vals[-1] >= 0.98 * INIT_LOSS:
        flags.append("not_learning")
    if any(v > low + 0.5 for v, low in zip(vals, itertools.accumulate(vals, min))):
        flags.append("loss_spike")
    if stats["val_drop_last_quarter"] > 0.01:
        flags.append("still_improving")
    if vals[-1] - best > 0.02 and stats["train_val_gap"] > 0.05:
        flags.append("overfitting")
    stats["flags"] = flags
    if points:
        n = len(history)
        idx = sorted({round(i * (n - 1) / (points - 1)) for i in range(points)}) if n > points else range(n)
        stats["curve"] = [[history[i]["iter"], round(trains[i], 3), round(vals[i], 3)] for i in idx]
    return stats


def summarize(run: Run, points: int | None = 8) -> dict[str, Any]:
    """Compact, LLM-readable view of one run: status, config, cost, and curve_stats() once it succeeded."""
    status, failed_task = run_status(run)
    out = {
        "run": run.pathspec,
        "url": f"{UI_URL}/{run.pathspec}",
        "status": status,
        "config_hash": _tag(run, "config"),
        "sweep": _tag(run, "sweep"),
        "launched_by": _tag(run, "launched_by"),
        "age_s": round(time.time() - run.created_at.timestamp()),
    }
    if failed_task:
        out["failed_step"] = failed_task.pathspec.split("/")[2]
        out["error"] = _error(failed_task)
        out["error_signature"] = error_signature(out["error"])

    start = run["start"].task  # tags are added in start, so every run find_runs returns has one
    if not start.successful:
        return out
    train_config = start.data.train_config
    out.update({
        "family": family_hash(Config(**train_config)),
        "label": start.data.exp_name,
        "changes": start.data.changes,
        "overrides": json.loads(start.data.overrides),
        "seed": train_config["seed"],
        "max_iters": train_config["max_iters"],
        "gpu_seconds": None,
    })
    if "train" in run and run["train"].task.finished:
        t = run["train"].task
        out["gpu_seconds"] = round((t.finished_at - t.created_at).total_seconds())
    if status != "succeeded":
        return out

    d = run.data
    out.update({"n_params": d.n_params, "train_seconds": d.train_seconds,
                "sec_per_iter": round(d.train_seconds / train_config["max_iters"], 4)})
    out.update(curve_stats(d.history, points))
    if {"non_finite_loss", "not_learning", "loss_spike", "no_history"} & set(out["flags"]):
        out["status"] = "diverged"
    return out


def observe(sweep: str | None = None, limit: int | None = None, points: int | None = 8) -> list[dict[str, Any]]:
    """summarize() for a sweep's runs, newest first: running, failed, diverged and succeeded alike."""
    return [summarize(r, points=points) for r in find_runs(sweep=sweep)[:limit]]


def sweeps() -> list[str]:
    """Sweep names in this project, most recently used first."""
    return list(dict.fromkeys(s for r in find_runs() if (s := _tag(r, "sweep"))))


def curves(sweep: str) -> list[dict[str, Any]]:
    """Every eval point of every finished run in a sweep, one row each: run, label, iter, losses and lr."""
    return [{"run": r.pathspec, "label": r.data.exp_name, **point}
            for r in find_runs(sweep=sweep, successful_only=True) for point in r.data.history]


def _family_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    losses = [r["final_val_loss"] for r in rows]
    first = rows[0]
    return {
        "family": first["family"], "label": first["label"], "changes": first["changes"],
        "overrides": {k: v for k, v in first["overrides"].items() if k != "seed"},
        "n_seeds": len(rows), "seeds": sorted(r["seed"] for r in rows),
        "mean_val_loss": round(statistics.fmean(losses), 4),
        "std_val_loss": round(statistics.stdev(losses), 4) if len(rows) > 1 else None,
        "min_val_loss": round(min(losses), 4),
        "mean_gpu_seconds": round(statistics.fmean(r["gpu_seconds"] or 0 for r in rows)),
        "flags": sorted({f for r in rows for f in r["flags"]}),
        "runs": [r["run"] for r in rows],
    }


def leaderboard(sweep: str | None = None, limit: int | None = None, by_family: bool = False) -> list[dict[str, Any]]:
    """Healthy succeeded runs, best final val loss first: the latest run per config.

    by_family=True groups seed replicates and ranks by their mean, with the spread,
    so a lucky seed doesn't look like a better config.
    """
    latest = {}
    for run in find_runs(sweep=sweep, successful_only=True):
        s = summarize(run, points=None)
        if s["status"] == "succeeded":
            latest.setdefault(s["config_hash"], s)  # newest first, so the first one seen wins
    rows = sorted(latest.values(), key=lambda r: r["final_val_loss"])
    if by_family:
        groups = {}
        for r in rows:
            groups.setdefault(r["family"], []).append(r)
        rows = sorted((_family_row(g) for g in groups.values()), key=lambda r: r["mean_val_loss"])
    return rows[:limit]


def _print_rows(rows: list[dict[str, Any]], cols: list[str]) -> None:
    """Fixed-width table; the last column is printed in full so links stay clickable."""
    def line(cells: list[str]) -> str:
        return "  ".join(f"{c:>14.14}" for c in cells[:-1]) + "  " + cells[-1]

    print(line(cols))
    for r in rows:
        print(line([str(r.get(c, "")) for c in cols]))


def _print_links(results: list[dict[str, Any]], timeout: float = 120) -> None:
    """Print each launch's run link as soon as the platform has turned its event into a run."""
    t0 = time.time()
    pending = []
    for r in results:
        if r["launched"]:
            pending.append(r["config_hash"])
        else:
            print(f"skipped   {r['config_hash']}  {UI_URL}/{r['existing']}")
    if pending:
        print(f"waiting up to {timeout:.0f}s for {len(pending)} run(s) to start…")
    while pending and time.time() - t0 < timeout:
        for h in list(pending):
            # an older run of the same config can exist, so only accept one created after this launch
            new = [r for r in find_runs(config_hash=h) if r.created_at.timestamp() >= t0 - 60]
            if new:
                print(f"started   {h}  {UI_URL}/{new[0].pathspec}")
                pending.remove(h)
        if pending:
            time.sleep(5)
    for h in pending:
        print(f"pending   {h}  not started yet; see `python src/lab.py status`")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="download the training text")
    s = sub.add_parser("sweep", help="launch presets and/or a grid as one sweep")
    s.add_argument("name")
    s.add_argument("--presets", default="", help="comma-separated (default: all, unless --grid)")
    s.add_argument("--grid", default="", help="JSON {field: [values]}")
    s.add_argument("--branch", default=DEFAULT_BRANCH)
    one = sub.add_parser("launch", help="launch one config (seed and measurement fields are fixed)")
    one.add_argument("overrides", nargs="?", default="{}")
    one.add_argument("--sweep", default="")
    one.add_argument("--label", default="")
    one.add_argument("--branch", default=DEFAULT_BRANCH)
    rep = sub.add_parser("replicate", help="rerun a succeeded config with other seeds")
    rep.add_argument("config_hash")
    rep.add_argument("--seeds", default="43,44")
    rep.add_argument("--branch", default=DEFAULT_BRANCH)
    for name in ("status", "leaderboard"):
        q = sub.add_parser(name)
        q.add_argument("--sweep", default=None)
        q.add_argument("--limit", type=int, default=20)
        q.add_argument("--json", action="store_true", help="full rows as JSON, for scripts and agents")
    sub.choices["leaderboard"].add_argument("--by-family", action="store_true", help="group seed replicates")
    args = p.parse_args()

    if args.cmd == "setup":
        print(f"{len(load_text()):,} characters ready")
        return
    if args.cmd == "sweep":
        presets = [n.strip() for n in args.presets.split(",") if n.strip()]
        _print_links(sweep(args.name, presets, json.loads(args.grid) if args.grid else None, branch=args.branch))
        return
    if args.cmd == "launch":
        _print_links([launch(json.loads(args.overrides), label=args.label, sweep=args.sweep,
                             launched_by="lab.cli", branch=args.branch, frozen=MEASUREMENT_FIELDS)])
        return
    if args.cmd == "replicate":
        seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
        _print_links(replicate(args.config_hash, seeds, branch=args.branch))
        return

    if args.cmd == "status":
        rows = observe(sweep=args.sweep, limit=args.limit, points=None)
        cols = ["config_hash", "label", "status", "final_val_loss", "gpu_seconds", "error", "url"]
    elif args.by_family:
        rows = leaderboard(sweep=args.sweep, limit=args.limit, by_family=True)
        cols = ["family", "changes", "n_seeds", "mean_val_loss", "std_val_loss", "mean_gpu_seconds"]
    else:
        rows = leaderboard(sweep=args.sweep, limit=args.limit)
        cols = ["config_hash", "changes", "final_val_loss", "train_val_gap", "gpu_seconds", "flags"]
    if args.json:
        print(json.dumps(rows, indent=1, default=str))
    else:
        _print_rows(rows, cols)


if __name__ == "__main__":
    main()
