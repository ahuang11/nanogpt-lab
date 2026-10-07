# Notes for coding agents

Read `README.md` first; it explains the workflow and the files. This file adds only what an agent needs.

## Setup, once per worktree

- Run `python src/lab.py setup`. The training text lives in `.cache/`, which isn't committed, so a fresh worktree doesn't have it and `flow.py run` fails without it.
- Use your branch name as your sweep name (e.g. branch `exp/rope` → `--sweep rope`), so parallel sessions' results don't mix in `status`.
- Label every launch `kilo-am`: pass `--launched-by kilo-am` to `lab.py` and `--launched_by kilo-am` to `flow.py run`, as in the commands below. Runs are tagged `launched_by:kilo-am`, which is how the user tells your runs from theirs.

## Tools

- Launch and read experiments with `src/lab.py`, not the UI or your own scripts. Use `--json` on `status` and `leaderboard` and parse the output.
- Config-only experiments: `python src/lab.py launch '<json overrides>' --sweep <name> --launched-by kilo-am`. These run the deployed code on `main`. The seed, train/val split and eval settings are fixed, so leave them out.
- Seed reruns: `python src/lab.py replicate <config_hash> --launched-by kilo-am`. This is the only way to change the seed.
- Code changes: work on a branch named `exp/<idea>`, **commit**, then run your worktree's code with `PYTHONPATH=src python flows/train/flow.py --environment=fast-bakery run --with kubernetes --sweep <name> --launched_by kilo-am`.

Every run is tagged with the commit it ran (`code` in `status` and `leaderboard`), and the same config on different code counts as a different experiment. Uncommitted changes show as `<commit>-dirty`, which can't be reproduced or compared later, so commit before every code-change run.

## Running a search

1. Agree a sweep name and a GPU budget with the user before launching anything. Track GPU time from the `gpu_seconds` field in `status --json`.
2. Launch a few runs at a time, each with a one-line hypothesis and the val loss you expect. The GPU pool has three GPUs shared by every session, so extra runs queue rather than fail; don't treat a queued run as stuck until it has waited about 15 minutes.
3. Poll `status --json` until every run is `succeeded`, `diverged` or `failed`, then compare each outcome with its hypothesis before proposing the next round. Read the `flags`: `still_improving` means under-trained, `overfitting` means regularize or shrink, `diverged` usually means the learning rate is too high.
4. Before calling a config better, `replicate` it and compare with `leaderboard --by-family`, on the same `code`. A gap smaller than about twice the seed-to-seed spread is noise.
5. Stop when the budget is spent, or when two rounds in a row don't beat the best by more than that noise. Report the best config confirmed on three seeds, its code commit, its cost, and what was ruled out.

## Stop and ask the user

- The same error appears in runs with different configs: that's a code bug. Fix it on an `exp/` branch or report it; don't launch more runs.
- Any `ImportError`, `NameError`, `SyntaxError` or `AttributeError`.
- Runs stay pending or never appear after about 15 minutes.
- Every run in a round fails.

## Don't

- Edit `LIMITS`, `MEASUREMENT_FIELDS` or `resolve()` in `src/config.py`. They are the guardrails for every launch, and changes to them need human review.
- Relaunch something that is running or has succeeded on the same code; `lab.py` skips those, so a `skipped` line is expected.
- Commit `.cache/`.
