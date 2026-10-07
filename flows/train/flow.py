"""TrainFlow: one experiment as one platform run, on a GPU, with a live card."""
from __future__ import annotations  # see config.py

import json
from dataclasses import asdict
from typing import Any

from metaflow import IncludeFile, Parameter, Run, card, current, kubernetes, pypi, resources, step
from metaflow.cards import Markdown, ProgressBar, VegaChart
from obproject import ProjectFlow, project_trigger

from config import DATA_PATH, EVENT, Config, changes, code_version, config_hash, resolve

# On a shared queue, extra GPU tasks can be placed on pools that never schedule them,
# so the train step is pinned to one GPU-only pool. Other steps use the default pool.
GPU_POOL = "ai-core-flows-gpu"  # cleanup: ignore[single-use-constant] the deployment knob the README points to


@project_trigger(event=EVENT)  # event payload keys map onto the parameters of the same name
class TrainFlow(ProjectFlow):

    overrides = Parameter("overrides", default="{}", help='JSON overrides on the baseline, e.g. {"lr": 1e-3}')
    label = Parameter("label", default="", help="human label; defaults to the changed knobs")
    sweep = Parameter("sweep", default="", help="batch this run belongs to")
    launched_by = Parameter("launched_by", default="manual", help="person, script or agent launching it")
    # Same config on different code is a different experiment, so runs carry the commit they ran.
    code = Parameter("code", default=code_version, help="git commit of the code this run uses")
    # An absolute path, since obproject-deploy runs this file from flows/train/, not the repo root.
    corpus = IncludeFile("corpus", default=str(DATA_PATH), help="Tiny Shakespeare, uploaded at launch")

    @step
    def start(self) -> None:
        cfg = resolve(json.loads(self.overrides or "{}"))
        self.train_config = asdict(cfg)
        self.config_hash = config_hash(cfg)
        self.changes = changes(cfg)
        self.exp_name = self.label or self.changes
        self.estimated_params = cfg.n_params

        tags = [f"label:{self.exp_name}", f"config:{self.config_hash}", f"code:{self.code}",
                f"launched_by:{self.launched_by}"]
        if self.sweep:
            tags.append(f"sweep:{self.sweep}")
        Run(f"{current.flow_name}/{current.run_id}").add_tags(tags)
        print(f"{self.exp_name}: config {self.config_hash}, ~{self.estimated_params:,} params")
        self.next(self.train)

    @card(type="blank", id="live", refresh_interval=5)
    @kubernetes(compute_pool=GPU_POOL)
    @resources(gpu=1, cpu=4, memory=16000)
    @pypi(packages={"torch": "2.6.0"})
    @step
    def train(self) -> None:
        # torch, and nanogpt which imports it, exist only in this step's @pypi environment
        import torch
        from nanogpt import train as run_experiment

        cfg = Config(**self.train_config)
        live = current.card["live"]
        progress = ProgressBar(max=cfg.max_iters, label="iterations")
        loss_chart = VegaChart(line_spec([], "loss", "loss", color="split"))
        lr_chart = VegaChart(line_spec([], "lr", "learning rate"))
        header = (f"# {self.exp_name}\n**changes:** {self.changes} · **config** `{self.config_hash}`"
                  f" · **code** `{self.code}` · **launched by** {self.launched_by}")
        if self.sweep:
            header += f" · **sweep** {self.sweep}"
        live.extend([Markdown(header), progress, loss_chart, lr_chart])
        live.refresh()

        seen = []

        def on_eval(record: dict[str, float]) -> None:
            seen.append(record)
            progress.update(record["iter"])
            loss_chart.update(line_spec(loss_rows(seen), "loss", "loss", color="split"))
            lr_chart.update(line_spec(seen, "lr", "learning rate"))
            live.refresh()

        result = run_experiment(cfg, on_eval=on_eval, text=self.corpus)
        model, chars = result["model"], result["chars"]

        self.history = result["history"]
        self.final_val_loss = self.history[-1]["val_loss"]
        self.best_val_loss = min(r["val_loss"] for r in self.history)
        self.train_seconds = self.history[-1]["elapsed_s"]
        self.n_params = result["n_params"]
        self.device = result["device"]
        # on CPU so the checkpoint loads on a laptop or a CPU-only dashboard
        self.state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        model.eval()
        prompt = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.sample = "".join(chars[i] for i in model.generate(prompt, 300)[0].tolist())

        live.append(Markdown(
            f"**final val loss** {self.final_val_loss:.4f} (best {self.best_val_loss:.4f})"
            f" · {self.n_params:,} params · {self.train_seconds:.0f}s on {self.device}"
            f"\n\n```\n{self.sample}\n```"
        ))
        live.refresh()
        self.next(self.end)

    @step
    def end(self) -> None:
        print(f"{self.exp_name} [{self.config_hash}]: final val loss {self.final_val_loss:.4f}")


def line_spec(rows: list[dict[str, Any]], y: str, title: str, color: str | None = None, width: int = 520) -> dict[str, Any]:
    """Vega-Lite line chart of `y` over iterations, one line per `color` value."""
    encoding = {
        "x": {"field": "iter", "type": "quantitative", "title": "iteration"},
        "y": {"field": y, "type": "quantitative", "scale": {"zero": False}},
    }
    if color:
        encoding["color"] = {"field": color, "type": "nominal", "title": None}
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title, "width": width, "height": 220,
        "data": {"values": rows}, "mark": "line", "encoding": encoding,
    }


def loss_rows(history: list[dict[str, float]]) -> list[dict[str, Any]]:
    return [
        {"iter": r["iter"], "loss": r[key], "split": split}
        for r in history
        for split, key in (("train", "train_loss"), ("val", "val_loss"))
    ]


if __name__ == "__main__":
    TrainFlow()
