#!/usr/bin/env python3
"""Build a single self-contained HTML dashboard for CroCo experiment progress.

The dashboard overlays the construction modes (e.g. max_reward vs gold_chosen) so
they can be compared while the runs are still in progress:

* DPO training dynamics (loss, preference accuracy, reward margin, chosen/rejected
  rewards), read from the latest ``checkpoint-*/trainer_state.json`` in each model
  directory.
* EuroEval learning curves over checkpoints, when checkpoint evaluations exist.
* The final before/after score comparison with confidence intervals.

Plots are rendered with Plotly (loaded from a CDN), so each chart has a built-in
"download as PNG" button. The page carries a meta-refresh, so regenerating the
file in place keeps an open browser tab live.

The data files can live locally or on a remote host: pass ``--ssh-host`` to read
the model directories and results file from that host over plain ``ssh``/``cat``,
so nothing extra needs to run there. With ``--watch`` the script regenerates the
HTML on an interval, giving a live dashboard driven entirely from this repo.
"""

import datetime as dt
import json
import logging
import os
import re
import subprocess
import time
import typing as t
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")
# Ordered most-specific first: each construction-mode / ablation maps to the
# unique tail of its model directory. ``max_reward`` is the bare suffix, so it
# must be matched last or it would swallow the ``-gold``/``-ls``/``-simpo`` ones.
_MODE_MARKERS = (
    ("croco-munin-apertus-8b-da-gold", "gold_chosen"),
    ("croco-munin-apertus-8b-da-generated", "generated"),
    ("croco-munin-apertus-8b-da-ls", "label_smoothing"),
    ("croco-munin-apertus-8b-da-simpo", "sigmoid_norm"),
    ("croco-munin-apertus-8b-da", "max_reward"),
)
_TRAINING_KEYS = {
    "loss": "loss",
    "rewards/accuracies": "acc",
    "rewards/margins": "margins",
    "rewards/chosen": "chosen",
    "rewards/rejected": "rejected",
    "grad_norm": "grad_norm",
}


@click.command()
@click.option(
    "--model-dir",
    "-m",
    "model_dirs",
    multiple=True,
    help="DPO output directory to read training dynamics from (repeatable).",
)
@click.option(
    "--results",
    "-r",
    default="euroeval_benchmark_results.jsonl",
    help="Path to the EuroEval results JSONL file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("croco_dashboard.html"),
    help="Output HTML path.",
)
@click.option(
    "--ssh-host",
    default=None,
    help="If set, read all paths from this host over ssh (paths are then "
    "relative to --remote-root).",
)
@click.option(
    "--remote-root",
    default="~/croco",
    help="Remote working directory that paths are relative to. Defaults to ~/croco.",
)
@click.option(
    "--watch",
    type=int,
    default=0,
    help="Regenerate every N seconds in a loop. 0 (default) builds once and exits.",
)
@click.option(
    "--refresh-seconds",
    type=int,
    default=60,
    help="Browser meta-refresh interval embedded in the page. Kept independent of "
    "--watch so the display tracks the file without racing the regeneration.",
)
def main(
    *,
    model_dirs: tuple[str, ...],
    results: str,
    output: Path,
    ssh_host: str | None,
    remote_root: str,
    watch: int,
    refresh_seconds: int,
) -> None:
    """Build (and optionally keep refreshing) the dashboard HTML.

    Args:
        model_dirs:
          DPO output directories to read training dynamics from.
        results:
          Path to the EuroEval results JSONL file.
        output:
          Output HTML path.
        ssh_host (optional):
          Host to read the data from over ssh. Defaults to None (local).
        remote_root:
          Remote directory that paths are relative to when reading over ssh.
        watch:
          Regenerate every N seconds when positive, else build once.
        refresh_seconds:
          Browser meta-refresh interval embedded in the page.
    """
    reader = _Reader(ssh_host=ssh_host, root=remote_root)

    while True:
        _build_once(
            reader=reader,
            model_dirs=model_dirs,
            results=results,
            output=output,
            refresh_seconds=refresh_seconds,
        )
        if watch <= 0:
            return
        time.sleep(watch)


def _build_once(
    *,
    reader: "_Reader",
    model_dirs: tuple[str, ...],
    results: str,
    output: Path,
    refresh_seconds: int,
) -> None:
    """Assemble the data and write the dashboard HTML atomically.

    Args:
        reader:
          File reader (local or ssh-backed).
        model_dirs:
          DPO output directories to read training dynamics from.
        results:
          Path to the EuroEval results JSONL file.
        output:
          Output HTML path.
        refresh_seconds:
          Browser meta-refresh interval to embed in the page.
    """
    data = {
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training": {
            _mode_label(Path(model_dir).name): series
            for model_dir in model_dirs
            if (series := _training_series(reader=reader, model_dir=model_dir))
        },
        **_eval_series(reader=reader, results=results),
    }
    html = _render_html(data=data, refresh_seconds=refresh_seconds)

    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(html)
    os.replace(tmp, output)
    logger.info("Wrote dashboard to %s (modes: %s)", output, list(data["training"]))


def _training_series(*, reader: "_Reader", model_dir: str) -> dict[str, t.Any] | None:
    """Extract per-step training metrics from a model directory.

    Args:
        reader:
          File reader (local or ssh-backed).
        model_dir:
          DPO output directory containing checkpoint-* subdirectories.

    Returns:
        Mapping with a ``steps`` list, one list per tracked metric, plus the
        latest step and total step count, or None when no state is available.
    """
    state = _latest_trainer_state(reader=reader, model_dir=model_dir)
    if state is None:
        return None
    rows = [entry for entry in state["log_history"] if "loss" in entry]

    series: dict[str, t.Any] = {"steps": [entry["step"] for entry in rows]}
    for source_key, out_key in _TRAINING_KEYS.items():
        series[out_key] = [entry.get(source_key) for entry in rows]

    max_steps = int(state.get("max_steps", 0) or 0)
    # Prefer the optimiser's ``global_step`` over the last logged step: logging
    # runs on a fixed cadence, so a finished run whose final step is not a
    # logging multiple (e.g. 625 with cadence 10) would otherwise read 620/625.
    last_logged_step = series["steps"][-1] if series["steps"] else 0
    series["latest_step"] = int(state.get("global_step", 0) or 0) or last_logged_step
    series["total"] = max_steps or series["latest_step"]
    return series


def _latest_trainer_state(
    *, reader: "_Reader", model_dir: str
) -> dict[str, t.Any] | None:
    """Return the trainer state from the highest-step checkpoint, if any.

    Falls back to a top-level ``trainer_state.json`` (written when training
    finishes) when no checkpoint directories are present.

    Args:
        reader:
          File reader (local or ssh-backed).
        model_dir:
          DPO output directory.

    Returns:
        The parsed trainer state, or None when none is available.
    """
    candidates = [
        f"{model_dir}/checkpoint-{step}/trainer_state.json"
        for step in reader.checkpoint_steps(model_dir=model_dir)
    ]
    candidates.append(f"{model_dir}/trainer_state.json")
    for path in reversed(candidates):
        text = reader.read_text(path=path)
        if text:
            return json.loads(text)
    return None


def _eval_series(*, reader: "_Reader", results: str) -> dict[str, t.Any]:
    """Parse EuroEval results into final scores and per-checkpoint curves.

    Args:
        reader:
          File reader (local or ssh-backed).
        results:
          Path to the EuroEval results JSONL file.

    Returns:
        Mapping with ``finals`` (model label -> metric -> score record),
        ``curves`` (mode -> metric -> list of step/score records) and the sorted
        list of dataset/metric keys present.
    """
    finals: dict[str, dict[str, t.Any]] = {}
    curves: dict[str, dict[str, list[dict[str, t.Any]]]] = {}
    keys: set[str] = set()
    base_entries: dict[str, dict[str, t.Any]] = {}

    text = reader.read_text(path=results)
    if not text:
        return {"finals": finals, "curves": curves, "metric_keys": []}

    for line in text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        model_id = record["model_info"]["id"]
        mode = _result_mode(model_id=model_id)
        if mode is None:
            continue
        step = _checkpoint_step(model_id=model_id)
        for result in record["evaluation_results"]:
            dataset = result["source_data"]["dataset_name"]
            key = f"{dataset}||{result['evaluation_name']}"
            keys.add(key)
            entry = _score_entry(result=result)
            if mode == "base":
                base_entries[key] = entry
            if step is None:
                finals.setdefault(_final_label(mode=mode), {})[key] = entry
            else:
                curves.setdefault(mode, {}).setdefault(key, []).append(
                    {"step": step, **entry}
                )

    # Anchor every mode's curve at step 0 with the base policy's score: before any
    # DPO steps the LoRA-adapted model is the base model, so step 0 is a shared,
    # meaningful starting point for all construction modes.
    for metric_points in curves.values():
        for key, points in metric_points.items():
            if key in base_entries:
                points.append({"step": 0, **base_entries[key]})
            points.sort(key=lambda point: point["step"])

    return {"finals": finals, "curves": curves, "metric_keys": sorted(keys)}


def _score_entry(*, result: dict[str, t.Any]) -> dict[str, t.Any]:
    """Extract score, confidence interval and direction from one result.

    Args:
        result:
          A single entry from a record's ``evaluation_results``.

    Returns:
        Mapping with ``score``, optional ``lower``/``upper`` CI bounds and the
        ``lower_is_better`` flag.
    """
    details = result["score_details"]
    interval = details.get("uncertainty", {}).get("confidence_interval", {})
    return {
        "score": details["score"],
        "lower": interval.get("lower"),
        "upper": interval.get("upper"),
        "lower_is_better": result["metric_config"]["lower_is_better"],
    }


def _result_mode(*, model_id: str) -> str | None:
    """Classify a model id into a comparison series, or None to ignore it.

    Args:
        model_id:
          The ``model_info.id`` from a results record.

    Returns:
        The construction-mode/ablation label, ``base``, or None for unrelated
        models.
    """
    if model_id.rstrip("/").endswith("munin-apertus-8b"):
        return "base"
    for marker, mode in _MODE_MARKERS:
        if marker in model_id:
            return mode
    return None


def _final_label(*, mode: str) -> str:
    """Return the display label for a final (non-checkpoint) model.

    Args:
        mode:
          The classified mode.

    Returns:
        A human-readable label for the final-comparison series.
    """
    return mode


def _checkpoint_step(*, model_id: str) -> int | None:
    """Return the checkpoint step encoded in a model id, if present.

    Args:
        model_id:
          The ``model_info.id`` from a results record.

    Returns:
        The checkpoint step, or None when the id is not a checkpoint.
    """
    match = _CHECKPOINT_RE.search(model_id)
    return int(match.group(1)) if match else None


def _mode_label(name: str) -> str:
    """Map a model-directory name to its construction-mode label.

    Args:
        name:
          The model directory name.

    Returns:
        The construction-mode/ablation label for the directory.
    """
    for marker, mode in _MODE_MARKERS:
        if name.endswith(marker):
            return mode
    return "max_reward"


class _Reader:
    """Read text files and list checkpoints, locally or over ssh."""

    def __init__(self, *, ssh_host: str | None, root: str) -> None:
        """Initialise the reader.

        Args:
            ssh_host:
              Host to read from over ssh, or None to read the local filesystem.
            root:
              Remote directory that relative paths resolve against (ssh only).
        """
        self.ssh_host = ssh_host
        self.root = root

    def read_text(self, *, path: str) -> str | None:
        """Return the contents of a file, or None if it is missing or empty.

        Args:
            path:
              File path (local, or relative to the remote root over ssh).

        Returns:
            The file contents, or None.
        """
        if self.ssh_host is None:
            local = Path(path)
            return local.read_text() if local.exists() else None
        out = self._ssh(command=f"cat {self.root}/{path} 2>/dev/null")
        return out or None

    def checkpoint_steps(self, *, model_dir: str) -> list[int]:
        """Return the checkpoint step numbers present in a model directory.

        Args:
            model_dir:
              DPO output directory (local, or relative to the remote root).

        Returns:
            Sorted checkpoint step numbers (ascending), empty when none exist.
        """
        if self.ssh_host is None:
            base = Path(model_dir)
            names = (
                [path.name for path in base.glob("checkpoint-*") if path.is_dir()]
                if base.exists()
                else []
            )
        else:
            out = self._ssh(
                command=f"ls -d {self.root}/{model_dir}/checkpoint-* 2>/dev/null"
            )
            names = out.splitlines()
        steps = [
            int(match.group(1))
            for name in names
            if (match := _CHECKPOINT_RE.search(name))
        ]
        return sorted(steps)

    def _ssh(self, *, command: str) -> str:
        """Run a command on the ssh host and return stdout (empty on failure).

        Args:
            command:
              Shell command to execute on the remote host.

        Returns:
            The command's standard output, or an empty string on any error.
        """
        host = self.ssh_host
        assert host is not None  # noqa: S101 - _ssh is only reached in remote mode
        result = subprocess.run(
            ["ssh", host, command],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout


def _render_html(*, data: dict[str, t.Any], refresh_seconds: int) -> str:
    """Render the dashboard HTML with the experiment data embedded.

    Args:
        data:
          The assembled dashboard data.
        refresh_seconds:
          Browser meta-refresh interval to embed.

    Returns:
        A complete, self-contained HTML document.
    """
    return _HTML_TEMPLATE.replace("__DATA__", json.dumps(data)).replace(
        "__REFRESH__", str(refresh_seconds)
    )


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="__REFRESH__">
<title>CroCo dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  body { font-family: -apple-system, system-ui, sans-serif;
         margin: 24px; color: #1a1a1a; }
  h1 { margin-bottom: 2px; }
  h2 { margin-top: 32px; border-bottom: 1px solid #ddd; }
  .meta { color: #666; font-size: 13px; margin-bottom: 8px; }
  .grid { display: grid;
          grid-template-columns: repeat(2, minmax(360px, 1fr)); gap: 12px; }
  .plot { width: 100%; height: 320px; }
  .full { height: 420px; }
  .note { color: #888; font-style: italic; padding: 12px 0; }
  select { font-size: 14px; padding: 4px; margin: 8px 0; }
  .pill { display:inline-block; padding:2px 8px; border-radius:10px; font-size:12px;
          background:#eef; color:#225; margin-left:6px; }
</style>
</head>
<body>
<h1>CroCo experiment dashboard</h1>
<div class="meta">Generated <span id="gen"></span> &middot; auto-refreshes every
  __REFRESH__s &middot; hover a chart and use the camera icon to save a PNG</div>

<h2>Training progress</h2>
<div id="progress" class="meta"></div>

<h2>DPO training dynamics</h2>
<div class="grid">
  <div id="loss" class="plot"></div>
  <div id="acc" class="plot"></div>
  <div id="margins" class="plot"></div>
</div>

<h2>EuroEval learning curves <span class="pill">per checkpoint</span></h2>
<div id="curveControls"></div>
<div id="curve" class="plot full"></div>

<h2>Final comparison <span class="pill">with 95% CIs</span></h2>
<div id="finals" style="width:100%"></div>

<script>
const DATA = __DATA__;
const COLOURS = {max_reward: "#1f77b4", gold_chosen: "#d62728", base: "#7f7f7f",
  generated: "#ff7f0e", label_smoothing: "#2ca02c", sigmoid_norm: "#9467bd"};
// Significance of a score relative to the base policy via non-overlapping 95%
// CIs: returns +1 (significantly better), -1 (significantly worse) or 0.
function sigVsBase(rec, baseRec) {
  if (!rec || !baseRec) return 0;
  if ([rec.lower, rec.upper, baseRec.lower, baseRec.upper].some(v => v == null))
    return 0;
  if (rec.lower > baseRec.upper) return rec.lower_is_better ? -1 : 1;
  if (rec.upper < baseRec.lower) return rec.lower_is_better ? 1 : -1;
  return 0;
}
document.getElementById("gen").textContent = DATA.generated;

const layout = (title, xlab, ylab, extra) => Object.assign({
  title: {text: title, font: {size: 14}}, margin: {t: 36, r: 12, b: 40, l: 52},
  xaxis: {title: xlab}, yaxis: {title: ylab}, legend: {orientation: "h", y: -0.2},
}, extra || {});
const CFG = {responsive: true, displaylogo: false,
  toImageButtonOptions: {format: "png", scale: 2}};

function progress() {
  const parts = Object.entries(DATA.training).map(([mode, s]) =>
    `${mode}: step ${s.latest_step}/${s.total}` +
    (s.total ? ` (${Math.round(100*s.latest_step/s.total)}%)` : ""));
  document.getElementById("progress").textContent =
    parts.length ? parts.join("  ·  ") : "no training data yet";
}

function lineTrace(mode, x, y) {
  return {x, y, mode: "lines+markers", name: mode, marker: {size: 4},
          line: {color: COLOURS[mode] || undefined}};
}

function trainingPlots() {
  const specs = [
    ["loss", "loss", "DPO loss", "loss"],
    ["acc", "acc", "Preference accuracy (chosen > rejected)", "accuracy"],
    ["margins", "margins", "Reward margin (chosen - rejected)", "margin"],
  ];
  for (const [div, key, title, ylab] of specs) {
    const traces = Object.entries(DATA.training)
      .map(([mode, s]) => lineTrace(mode, s.steps, s[key]));
    const lay = layout(title, "step", ylab);
    if (key === "acc") lay.yaxis.range = [0, 1];
    Plotly.newPlot(div, traces, lay, CFG);
  }
}

function curveMetrics() {
  const set = new Set();
  for (const metrics of Object.values(DATA.curves))
    Object.keys(metrics).forEach(k => set.add(k));
  return [...set].sort();
}

function drawCurve(metric) {
  const traces = [];
  for (const [mode, metrics] of Object.entries(DATA.curves)) {
    const pts = metrics[metric]; if (!pts) continue;
    const hasCI = pts.every(p => p.lower != null && p.upper != null);
    traces.push({x: pts.map(p => p.step), y: pts.map(p => p.score),
      mode: "lines+markers", name: mode, line: {color: COLOURS[mode]},
      error_y: hasCI ? {type: "data", symmetric: false,
        array: pts.map(p => p.upper - p.score),
        arrayminus: pts.map(p => p.score - p.lower),
        color: "black", thickness: 2.5, width: 4} : undefined});
  }
  const [ds, m] = metric.split("||");
  Plotly.newPlot("curve", traces, layout(`${ds} - ${m}`, "checkpoint step", m), CFG);
}

function curves() {
  const metrics = curveMetrics();
  const controls = document.getElementById("curveControls");
  if (!metrics.length) {
    controls.innerHTML = "";
    document.getElementById("curve").innerHTML =
      '<div class="note">No checkpoint evaluations yet - these appear once a ' +
      'training run finishes and its checkpoints are benchmarked.</div>';
    return;
  }
  const sel = document.createElement("select");
  metrics.forEach(m => { const o = document.createElement("option");
    o.value = m; o.textContent = m.replace("||", "  /  "); sel.appendChild(o); });
  sel.onchange = () => drawCurve(sel.value);
  controls.innerHTML = "Dataset / metric: "; controls.appendChild(sel);
  drawCurve(metrics[0]);
}

function finals() {
  const labels = Object.keys(DATA.finals);
  if (!labels.length) {
    document.getElementById("finals").innerHTML =
      '<div class="note">No final evaluations yet.</div>'; return;
  }
  const baseLabel = labels.find(l => l.startsWith("base"));
  const base = baseLabel ? DATA.finals[baseLabel] : null;
  // Group by dataset: the y label is just the dataset name; each metric is a
  // separate bar whose name shows on hover. The group arrow is set when ANY
  // (method, metric) in the dataset is significantly better / worse than base.
  const allKeys = [...new Set(labels.flatMap(l => Object.keys(DATA.finals[l])))];
  const byDs = {};
  for (const key of allKeys) {
    const ds = key.split("||")[0];
    if (!byDs[ds]) byDs[ds] = [];
    byDs[ds].push(key);
  }
  const datasets = Object.keys(byDs).sort();
  for (const ds of datasets) byDs[ds].sort();
  const maxSlots = Math.max(1, ...datasets.map(d => byDs[d].length));
  const catOf = {}, cats = [];
  for (const ds of datasets) {
    let better = false, worse = false;
    for (const key of byDs[ds]) {
      for (const label of labels) {
        if (label === baseLabel) continue;
        const s = sigVsBase(DATA.finals[label][key], base ? base[key] : null);
        if (s > 0) better = true; else if (s < 0) worse = true;
      }
    }
    const arrows = (better ? "▲" : "") + (worse ? "▼" : "");
    const suffix = arrows.padStart(2, "\u00A0");
    const cat = `${ds}\u00A0${suffix}`;
    catOf[ds] = cat; cats.push(cat);
  }
  const traces = [];
  const shown = new Set();
  for (let slot = 0; slot < maxSlots; slot++) {
    if (slot > 0) {
      traces.push({type: "bar", orientation: "h", showlegend: false,
        legendgroup: "_gap" + slot, hoverinfo: "skip",
        y: datasets.map(ds => catOf[ds]), x: datasets.map(() => null),
        marker: {color: "rgba(0,0,0,0)"}});
    }
    for (const label of labels) {
      const mode = label.startsWith("base") ? "base" : label;
      const y = [], x = [], up = [], dn = [], cd = [];
      for (const ds of datasets) {
        if (slot >= byDs[ds].length) continue;
        const key = byDs[ds][slot];
        const r = DATA.finals[label][key];
        if (!r) continue;
        y.push(catOf[ds]); x.push(r.score);
        up.push(r.upper != null ? r.upper - r.score : 0);
        dn.push(r.lower != null ? r.score - r.lower : 0);
        cd.push(key.split("||")[1].replace(/^test_/, ""));
      }
      if (!x.length) continue;
      const showlegend = !shown.has(label); shown.add(label);
      traces.push({type: "bar", orientation: "h", name: label,
        legendgroup: label, showlegend, y, x, customdata: cd,
        marker: {color: COLOURS[mode]},
        hovertemplate: "%{customdata}: %{x:.3f}<extra>" + label + "</extra>",
        error_x: {type: "data", symmetric: false, color: "black", thickness: 1,
          array: up, arrayminus: dn}});
    }
  }
  const slots = labels.length * maxSlots + Math.max(0, maxSlots - 1);
  const height = Math.max(420, 60 + datasets.length * slots * 14);
  Plotly.newPlot("finals", traces,
    layout("Final EuroEval scores (▲ better / ▼ worse than base in group, 95% CI)",
      "score", "", {barmode: "group", bargroupgap: 0.25, height,
      margin: {t: 40, r: 40, b: 40, l: 10},
      yaxis: {automargin: true, categoryorder: "array", categoryarray: cats,
        autorange: "reversed", tickfont: {family: "monospace", size: 11}}}), CFG);
}

progress(); trainingPlots(); curves(); finals();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
