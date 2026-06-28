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
"""

import datetime as dt
import json
import logging
import re
import typing as t
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")
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
    type=click.Path(path_type=Path),
    help="DPO output directory to read training dynamics from (repeatable).",
)
@click.option(
    "--results",
    "-r",
    type=click.Path(path_type=Path),
    default=Path("euroeval_benchmark_results.jsonl"),
    help="Path to the EuroEval results JSONL file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path, allow_dash=True),
    default=Path("croco_dashboard.html"),
    help="Output HTML path, or '-' for stdout.",
)
@click.option(
    "--total-steps",
    type=int,
    default=0,
    help="Planned total training steps, for progress display. 0 to infer.",
)
def main(
    *,
    model_dirs: tuple[Path, ...],
    results: Path,
    output: Path,
    total_steps: int,
) -> None:
    """Build the dashboard HTML from training states and EuroEval results.

    Args:
        model_dirs:
          DPO output directories to read training dynamics from.
        results:
          Path to the EuroEval results JSONL file.
        output:
          Output HTML path, or '-' for stdout.
        total_steps:
          Planned total training steps for progress display, or 0 to infer.
    """
    data = {
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training": {
            _mode_label(model_dir.name): _training_series(
                model_dir=model_dir, total_steps=total_steps
            )
            for model_dir in model_dirs
            if _latest_trainer_state(model_dir=model_dir) is not None
        },
        **_eval_series(results=results),
    }

    html = _render_html(data=data)
    if str(output) == "-":
        click.echo(html, nl=False)
    else:
        output.write_text(html)
        logger.info("Wrote dashboard to %s", output)


def _training_series(*, model_dir: Path, total_steps: int) -> dict[str, t.Any]:
    """Extract per-step training metrics from a model directory.

    Args:
        model_dir:
          DPO output directory containing checkpoint-* subdirectories.
        total_steps:
          Planned total training steps, or 0 to infer from the log.

    Returns:
        Mapping with a ``steps`` list, one list per tracked metric, plus the
        latest step and the (possibly inferred) total step count.
    """
    state = _latest_trainer_state(model_dir=model_dir)
    history = state["log_history"] if state else []
    rows = [entry for entry in history if "loss" in entry]

    series: dict[str, t.Any] = {"steps": [entry["step"] for entry in rows]}
    for source_key, out_key in _TRAINING_KEYS.items():
        series[out_key] = [entry.get(source_key) for entry in rows]

    inferred_total = 0
    if state is not None:
        max_steps = state.get("max_steps", 0)
        inferred_total = int(max_steps) if max_steps else 0
    series["latest_step"] = series["steps"][-1] if series["steps"] else 0
    series["total"] = total_steps or inferred_total or series["latest_step"]
    return series


def _latest_trainer_state(*, model_dir: Path) -> dict[str, t.Any] | None:
    """Return the trainer state from the highest-step checkpoint, if any.

    Falls back to a top-level ``trainer_state.json`` (written when training
    finishes) when no checkpoint directories are present.

    Args:
        model_dir:
          DPO output directory.

    Returns:
        The parsed trainer state, or None when none is available.
    """
    if not model_dir.exists():
        return None
    checkpoints = sorted(
        (path for path in model_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    for candidate in (*reversed(checkpoints), model_dir):
        state_path = candidate / "trainer_state.json"
        if state_path.exists():
            return json.loads(state_path.read_text())
    return None


def _eval_series(*, results: Path) -> dict[str, t.Any]:
    """Parse EuroEval results into final scores and per-checkpoint curves.

    Args:
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

    if not results.exists():
        return {"finals": finals, "curves": curves, "metric_keys": []}

    for line in results.read_text().splitlines():
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
            if step is None:
                finals.setdefault(_final_label(mode=mode), {})[key] = entry
            else:
                point = {"step": step, **entry}
                curves.setdefault(mode, {}).setdefault(key, []).append(point)

    for metric_points in curves.values():
        for points in metric_points.values():
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
        ``base``, ``gold_chosen``, ``max_reward`` or None for unrelated models.
    """
    if model_id.rstrip("/").endswith("munin-apertus-8b"):
        return "base"
    if "croco-munin-apertus-8b-da-gold" in model_id:
        return "gold_chosen"
    if "croco-munin-apertus-8b-da" in model_id:
        return "max_reward"
    return None


def _final_label(*, mode: str) -> str:
    """Return the display label for a final (non-checkpoint) model.

    Args:
        mode:
          The classified mode.

    Returns:
        A human-readable label for the final-comparison series.
    """
    return {"base": "base (munin-apertus-8b)"}.get(mode, mode)


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
        ``gold_chosen`` for ``*-gold`` directories, else ``max_reward``.
    """
    return "gold_chosen" if name.endswith("-gold") else "max_reward"


def _render_html(*, data: dict[str, t.Any]) -> str:
    """Render the dashboard HTML with the experiment data embedded.

    Args:
        data:
          The assembled dashboard data.

    Returns:
        A complete, self-contained HTML document.
    """
    payload = json.dumps(data)
    return _HTML_TEMPLATE.replace("__DATA__", payload)


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="120">
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
<div class="meta">Generated <span id="gen"></span> &middot; auto-refreshes every 120s
  &middot; hover a chart and use the camera icon to save a PNG</div>

<h2>Training progress</h2>
<div id="progress" class="meta"></div>

<h2>DPO training dynamics</h2>
<div class="grid">
  <div id="loss" class="plot"></div>
  <div id="acc" class="plot"></div>
  <div id="margins" class="plot"></div>
  <div id="rewards" class="plot"></div>
</div>

<h2>EuroEval learning curves <span class="pill">per checkpoint</span></h2>
<div id="curveControls"></div>
<div id="curve" class="plot full"></div>

<h2>Final comparison <span class="pill">with 95% CIs</span></h2>
<div id="finals" class="plot full"></div>

<script>
const DATA = __DATA__;
const COLOURS = {max_reward: "#1f77b4", gold_chosen: "#d62728", base: "#7f7f7f"};
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
  const rewardTraces = [];
  for (const [mode, s] of Object.entries(DATA.training)) {
    rewardTraces.push(Object.assign(lineTrace(mode + " chosen", s.steps, s.chosen),
      {line: {color: COLOURS[mode], dash: "solid"}}));
    rewardTraces.push(Object.assign(lineTrace(mode + " rejected", s.steps, s.rejected),
      {line: {color: COLOURS[mode], dash: "dot"}}));
  }
  Plotly.newPlot("rewards", rewardTraces,
    layout("Implicit rewards: chosen (solid) vs rejected (dotted)",
      "step", "reward"), CFG);
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
        arrayminus: pts.map(p => p.score - p.lower)} : undefined});
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
  const keys = [...new Set(labels.flatMap(l => Object.keys(DATA.finals[l])))].sort();
  const traces = labels.map(label => {
    const recs = keys.map(k => DATA.finals[label][k]);
    return {type: "bar", name: label,
      x: keys.map(k => k.replace("||", "/")),
      y: recs.map(r => r ? r.score : null),
      error_y: {type: "data", symmetric: false,
        array: recs.map(r => (r && r.upper != null) ? r.upper - r.score : 0),
        arrayminus: recs.map(r => (r && r.lower != null) ? r.score - r.lower : 0)}};
  });
  Plotly.newPlot("finals", traces,
    layout("Final EuroEval scores by model", "", "score", {barmode: "group"}), CFG);
}

progress(); trainingPlots(); curves(); finals();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
