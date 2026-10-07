"""Experiment settings, presets and the limits every launch is checked against."""
# Lazy annotations: platform pods may run Python < 3.10, where `int | float` fails at import.
from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

# config.py is in src/, not beside flow.py, so Metaflow's code packager won't find it by walking
# the flow's folder. "include" ships it with every run; without it, pods fail with ModuleNotFoundError.
METAFLOW_PACKAGE_POLICY = "include"

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path(__file__).parents[1] / "data" / "input.txt"
# Must match the data: parameter counts and INIT_LOSS are computed from it without loading the text.
VOCAB_SIZE = 65
INIT_LOSS = math.log(VOCAB_SIZE)  # a run that ends at or above this learned nothing

# lab.launch() publishes EVENT and flow.py's @project_trigger listens for it.
# DEFAULT_BRANCH must match [dev-assets] in obproject.toml, or launches reach no deployment.
EVENT = "train_request"
DEFAULT_BRANCH = "main"
# The Client API doesn't expose the UI address, so run links are built from this.
UI_URL = "https://ui.merced.obp.outerbounds.com/dashboard/runs/p/default"

# Sized for one g5.2xlarge (24 GB A10G) and a few minutes per run.
LIMITS = {
    "max_params": 20_000_000,
    "max_iters": 20_000,
    "max_batch_size": 256,
    "max_block_size": 512,
    "max_lr": 1e-2,
}

PRESETS = {
    "exp-a-baseline": {},
    "exp-b-lr": {"lr": 1e-3},
    "exp-c-depth": {"n_layer": 6},
    "exp-d-width": {"n_embd": 256},
    "exp-e-dropout": {"dropout": 0.0},
}

# These decide how a run is scored, not what is trained, so changing them could improve the
# number without improving the model. `lab.py launch` freezes them; seeds change only via `replicate`.
MEASUREMENT_FIELDS = ("train_frac", "eval_iters", "eval_interval", "seed")  # cleanup: ignore[single-use-constant] a named policy the README and AGENTS.md refer to


@dataclass(frozen=True)
class Config:
    """One experiment's settings. Every instance is validated against LIMITS, so an invalid Config can't exist."""

    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    lr: float = 3e-4
    min_lr_frac: float = 0.1
    warmup_iters: int = 100
    max_iters: int = 3000
    batch_size: int = 64
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 20
    train_frac: float = 0.9
    seed: int = 42

    def __post_init__(self) -> None:
        checks = [
            (self.n_embd % self.n_head == 0, f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"),
            (0 <= self.dropout < 1, "dropout must be in [0, 1)"),
            (0 < self.lr <= LIMITS["max_lr"], f"lr must be in (0, {LIMITS['max_lr']}]"),
            (0 < self.min_lr_frac <= 1, "min_lr_frac must be in (0, 1]"),
            (0 < self.max_iters <= LIMITS["max_iters"], f"max_iters must be in (0, {LIMITS['max_iters']}]"),
            (0 <= self.warmup_iters < self.max_iters, "warmup_iters must be in [0, max_iters)"),
            (0 < self.batch_size <= LIMITS["max_batch_size"], f"batch_size must be in (0, {LIMITS['max_batch_size']}]"),
            (0 < self.block_size <= LIMITS["max_block_size"], f"block_size must be in (0, {LIMITS['max_block_size']}]"),
            (min(self.n_layer, self.n_head, self.n_embd, self.eval_interval, self.eval_iters) > 0,
             "n_layer, n_head, n_embd, eval_interval and eval_iters must be positive"),
            (0 < self.train_frac < 1, "train_frac must be in (0, 1)"),
            (self.n_params <= LIMITS["max_params"], f"~{self.n_params:,} params exceeds the {LIMITS['max_params']:,} limit"),
        ]
        problems = [msg for ok, msg in checks if not ok]
        if problems:
            raise ValueError("invalid config: " + "; ".join(problems))

    @property
    def n_params(self) -> int:
        """Parameter count of nanogpt.GPT for this config, without building it."""
        e = self.n_embd
        return self.n_layer * (12 * e * e + 13 * e) + (VOCAB_SIZE + self.block_size) * e + 2 * e + VOCAB_SIZE * e


def load_text(path: Path = DATA_PATH) -> str:
    """Tiny Shakespeare, downloaded once to data/input.txt."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return path.read_text()


def _coerce(key: str, value: Any, kind: type) -> int | float:
    if kind is int:
        if isinstance(value, bool) or float(value) != int(float(value)):
            raise ValueError(f"{key} must be an integer, got {value!r}")
        return int(float(value))
    return float(value)


def resolve(overrides: dict[str, Any] | None = None, base: Config | None = None, frozen: tuple[str, ...] = ()) -> Config:
    """Config from a dict of overrides on `base` (default: the baseline). Raises ValueError on unknown, frozen or invalid fields.

    frozen names fields this caller may not override (MEASUREMENT_FIELDS for the launch CLI).
    """
    overrides = dict(overrides or {})
    types = {f.name: type(f.default) for f in fields(Config)}  # f.type is a string under lazy annotations
    unknown = sorted(set(overrides) - set(types))
    if unknown:
        raise ValueError(f"unknown config fields {unknown}; allowed: {sorted(types)}")
    locked = sorted(set(overrides) & set(frozen))
    if locked:
        raise ValueError(f"fields {locked} are fixed for this caller and cannot be overridden")
    return replace(base or Config(), **{k: _coerce(k, v, types[k]) for k, v in overrides.items()})


def config_hash(cfg: Config) -> str:
    """Stable 10-char id for a full config, used to tag and dedupe runs."""
    blob = json.dumps(asdict(cfg), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def family_hash(cfg: Config) -> str:
    """Like config_hash but ignoring seed, so seed replicates of one config share an id."""
    blob = json.dumps({k: v for k, v in asdict(cfg).items() if k != "seed"}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def changes(cfg: Config, base: Config | None = None) -> str:
    """The knob(s) this config turns relative to `base` (default: the baseline), e.g. 'lr=0.001'."""
    b = asdict(base or Config())
    return ", ".join(f"{k}={v}" for k, v in asdict(cfg).items() if b[k] != v) or "baseline"
