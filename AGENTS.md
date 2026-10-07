# Notes for coding agents

Read `README.md` first; it explains the workflow and the files. This file adds only what an agent needs.

## Tools

- Launch and read experiments with `src/lab.py`, not the UI or your own scripts. Use `--json` on `status` and `leaderboard` and parse the output.
- Config-only experiments: `python src/lab.py launch '<json overrides>' --sweep <name>`. The seed, train/val split and eval settings are fixed, so leave them out.
- Seed reruns: `python src/lab.py replicate <config_hash>`. This is the only way to change the seed.
- Code changes: work on a branch named `exp/<idea>`, then run your worktree's code with `PYTHONPATH=src python flows/train/flow.py --environment=fast-bakery run --with kubernetes --sweep <name>`.

## Running a search

1. Agree a sweep name and a GPU budget with the user before launching anything. Track GPU time from the `gpu_seconds` field in `status --json`.
2. Launch at most three runs at a time, since the GPU pool has three GPUs. For each one, write down a one-line hypothesis and the val loss you expect.
3. Poll `status --json` until every run is `succeeded`, `diverged` or `failed`, then compare each outcome with its hypothesis before proposing the next round. Read the `flags`: `still_improving` means under-trained, `overfitting` means regularize or shrink, `diverged` usually means the learning rate is too high.
4. Before calling a config better, `replicate` it and compare with `leaderboard --by-family`. A gap smaller than about twice the seed-to-seed spread is noise.
5. Stop when the budget is spent, or when two rounds in a row don't beat the best by more than that noise. Report the best config confirmed on three seeds, its cost, and what was ruled out.

## Stop and ask the user

- The same error appears in runs with different configs: that's a code bug. Fix it on an `exp/` branch or report it; don't launch more runs.
- Any `ImportError`, `NameError`, `SyntaxError` or `AttributeError`.
- Runs stay pending or never appear after about 15 minutes.
- Every run in a round fails.

## Don't

- Edit `LIMITS`, `MEASUREMENT_FIELDS` or `resolve()` in `src/config.py`. They are the guardrails for every launch, and changes to them need human review.
- Relaunch something that is running or has succeeded; `lab.py` skips those, so a `skipped` line is expected.
- Commit `.cache/`.
