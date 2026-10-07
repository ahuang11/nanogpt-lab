"""Compare app: one sweep's runs side by side, with their loss curves overlaid."""
import hvplot.pandas  # noqa: F401
import pandas as pd
import panel as pn
import panel_material_ui as pmui
import param

from lab import curves, observe, sweeps

pn.extension("tabulator", throttled=True, loading_indicator=True, sizing_mode="stretch_width")
pn.widgets.Tabulator.param.theme.default = "materialize"

COLUMNS = ["label", "status", "final_val_loss", "train_val_gap", "flags", "gpu_seconds", "error", "run"]
PLOT_HEIGHT = 320


class CompareApp(pn.viewable.Viewer):

    sweep = param.Selector(default=None, objects=[], label="Sweep")

    def __init__(self, **params) -> None:
        self._status = pmui.Alert(object="Loading sweeps…", severity="info")
        self._table = pn.widgets.Tabulator(
            pd.DataFrame(columns=COLUMNS), selectable="checkbox", disabled=True, show_index=False,
            hidden_columns=["run"], layout="fit_data_stretch", height=260,
        )
        self._val = pn.pane.HoloViews(height=PLOT_HEIGHT)
        self._train = pn.pane.HoloViews(height=PLOT_HEIGHT)
        self._lr = pn.pane.HoloViews(height=PLOT_HEIGHT)
        super().__init__(**params)
        self._curves = pd.DataFrame()

        refresh = pmui.Button(label="Refresh", icon="refresh")
        refresh.on_click(self._on_refresh)
        self._table.param.watch(self._update_plots, "selection")
        self._page = pmui.Page(
            title="ob_train · compare",
            sidebar=[pmui.Select.from_param(self.param.sweep), refresh],
            main=[self._status, self._table, pn.Row(self._val, self._train), self._lr],
        )
        pn.state.onload(self._load_sweeps)

    def _load_sweeps(self) -> None:
        names = sweeps()
        if not names:
            self._status.param.update(object="No sweeps yet. Launch one with `python src/lab.py sweep <name>`.",
                                      severity="warning")
            return
        self.param.sweep.objects = names
        if self.sweep not in names:
            self.sweep = names[0]  # fires _update_sweep
        # Runs finish while the page is open, so pick up new results without a manual refresh.
        pn.state.add_periodic_callback(self._on_refresh, period=30_000)

    def _on_refresh(self, *events) -> None:
        self.param.sweep.objects = sweeps()  # pick up sweeps launched since the page opened
        self._load_data(keep_selection=True)

    @param.depends("sweep", watch=True)
    def _update_sweep(self) -> None:
        self._load_data(keep_selection=False)

    def _load_data(self, keep_selection: bool) -> None:
        if not self.sweep:
            return
        runs = pd.DataFrame(observe(sweep=self.sweep, points=None)).reindex(columns=COLUMNS)
        runs["flags"] = runs["flags"].map(lambda f: ", ".join(f) if isinstance(f, list) else "")
        runs = runs.sort_values("final_val_loss", na_position="last").reset_index(drop=True)
        old = self._table.value
        kept = set(old["run"].iloc[self._table.selection]) if keep_selection and len(old) else set()
        seen = set(old["run"]) if keep_selection else set()
        # Keep what the user picked, and add runs that finished since the last refresh.
        selection = [i for i, r in runs.iterrows()
                     if r["run"] in kept or (r["status"] == "succeeded" and r["run"] not in seen)]
        self._curves = pd.DataFrame(curves(self.sweep))
        n_running = int((runs["status"] == "running").sum())
        with pn.io.hold():
            self._status.param.update(
                object=f"**{len(runs)}** runs in `{self.sweep}`, {n_running} still running. "
                       "Curves appear when a run finishes; each run's card shows them live.",
                severity="info",
            )
            self._table.value = runs
            if self._table.selection == selection:
                self._update_plots()
            else:
                self._table.selection = selection  # fires _update_plots

    def _update_plots(self, *events) -> None:
        df = self._table.value
        picked = set(df["run"].iloc[self._table.selection]) if len(df) else set()
        data = self._curves[self._curves["run"].isin(picked)] if len(self._curves) else self._curves
        with pn.io.hold():
            if data.empty:
                for pane in (self._val, self._train, self._lr):
                    pane.object = None
                return
            common = dict(x="iter", by="label", xlabel="iteration", responsive=True, height=PLOT_HEIGHT)
            self._val.object = data.hvplot.line(y="val_loss", ylabel="loss", title="Validation loss", **common)
            self._train.object = data.hvplot.line(y="train_loss", ylabel="loss", title="Train loss", **common)
            self._lr.object = data.hvplot.line(y="lr", ylabel="learning rate", title="Learning rate", **common)

    def __panel__(self) -> pmui.Page:
        return self._page


CompareApp().servable()
