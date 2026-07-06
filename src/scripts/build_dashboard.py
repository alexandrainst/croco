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
# must be matched last or it would swallow the ``-gold``/``-ls``/``-simpo`` ones;
# likewise ``-simpo`` must come after ``-simpo-tuned``/``-simpo-full`` so it does
# not swallow those SimPO ablation tails.
_MODE_MARKERS = (
    ("croco-munin-apertus-8b-da-gold", "gold_chosen"),
    ("croco-munin-apertus-8b-da-generated", "generated"),
    ("croco-munin-apertus-8b-da-ls", "label_smoothing"),
    ("croco-munin-apertus-8b-da-simpo-tuned", "simpo_tuned"),
    ("croco-munin-apertus-8b-da-simpo-full", "simpo_full"),
    ("croco-munin-apertus-8b-da-simpo", "sigmoid_norm"),
    ("croco-munin-apertus-8b-da-grpo", "grpo"),
    ("croco-munin-apertus-8b-da-llamarm", "llama_rm"),
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
_ITERATION_FIELDS = ("num_iterations", "iterations", "n_iterations")
_SAMPLE_COUNT_FIELDS = ("num_samples",)


def _discover_model_dirs(*, reader: "_Reader") -> tuple[str, ...]:
    """Auto-discover model directories from config/*.yaml output_dir fields.

    Args:
        reader:
          File reader (local or ssh-backed).

    Returns:
        Tuple of model directory paths (relative to reader root).
    """
    try:
        if reader.ssh_host:
            # List and parse remote configs in one SSH call
            remote_cmd = (
                f"cd {reader.root} && "
                "grep -h '^  output_dir:' config/*.yaml 2>/dev/null"
            )
            result = subprocess.run(
                ["ssh", reader.ssh_host, remote_cmd],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                logger.warning(
                    "Could not read remote configs: %s", result.stderr.strip()
                )
                return ()
            model_dirs = []
            for line in result.stdout.strip().split("\n"):
                if ":" in line:
                    output_dir = line.split(":", 1)[1].strip()
                    # Skip micro ablations and smoke tests
                    if _include_discovered_model_dir(output_dir=output_dir):
                        model_dirs.append(output_dir)
            return tuple(sorted(set(model_dirs)))
        else:
            import glob

            model_dirs = []
            for config_path in glob.glob("config/*.yaml"):
                try:
                    with open(config_path) as f:
                        for line in f:
                            if line.strip().startswith("output_dir:"):
                                output_dir = line.split(":", 1)[1].strip()
                                # Skip micro ablations and smoke tests
                                if _include_discovered_model_dir(output_dir=output_dir):
                                    model_dirs.append(output_dir)
                                break
                except Exception as e:
                    logger.debug("Could not parse %s: %s", config_path, e)
            return tuple(sorted(set(model_dirs)))
    except Exception as e:
        logger.warning("Could not discover configs: %s", e)
        return ()


def _include_discovered_model_dir(*, output_dir: str) -> bool:
    """Return whether an auto-discovered output dir belongs on the dashboard.

    Args:
        output_dir:
          Model output directory read from a config file.

    Returns:
        True for full experiment directories, false for micro/smoke fixtures.
    """
    name = Path(output_dir).name.lower()
    return "micro" not in name and "smoke" not in name


@click.command()
@click.option(
    "--model-dir",
    "-m",
    "model_dirs",
    multiple=True,
    help="DPO output directory to read training dynamics from (repeatable).",
)
@click.option(
    "--configs/--no-configs",
    "-c",
    default=True,
    help=(
        "Auto-discover model directories from config/*.yaml output_dir fields "
        "(default). Use --no-configs to disable and only use -m."
    ),
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
    configs: bool,
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
        configs:
          If set, auto-discover model directories from config/*.yaml.
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

    # Auto-discover model directories from config/*.yaml (default behavior)
    # Can be disabled with --no-configs, or supplemented with -m flags
    if configs:
        config_dirs = _discover_model_dirs(reader=reader)
        # Merge with any manually specified -m directories
        model_dirs = tuple(sorted(set(config_dirs + model_dirs)))
        logger.info(
            "Auto-discovered %d model directories from configs", len(config_dirs)
        )
        if model_dirs:
            logger.info("Total model directories: %d", len(model_dirs))

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
    final_entries: dict[tuple[str, str], tuple[int | None, int, dict[str, t.Any]]] = {}
    curve_entries: dict[
        tuple[str, int, str], tuple[int | None, int, dict[str, t.Any]]
    ] = {}
    result_order = 0

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
            result_order += 1
            dataset = result["source_data"]["dataset_name"]
            key = f"{dataset}||{result['evaluation_name']}"
            keys.add(key)
            entry = _score_entry(result=result)
            iterations = _iteration_count(record=record, result=result)
            if step is None:
                final_key = (_final_label(mode=mode), key)
                current = final_entries.get(final_key)
                if _should_replace_eval(
                    current=current, iterations=iterations, order=result_order
                ):
                    final_entries[final_key] = (iterations, result_order, entry)
            else:
                curve_key = (mode, step, key)
                current = curve_entries.get(curve_key)
                point = {"step": step, **entry}
                if _should_replace_eval(
                    current=current, iterations=iterations, order=result_order
                ):
                    curve_entries[curve_key] = (iterations, result_order, point)

    for (label, key), (_, _, entry) in final_entries.items():
        finals.setdefault(label, {})[key] = entry
    for (mode, _, key), (_, _, point) in curve_entries.items():
        curves.setdefault(mode, {}).setdefault(key, []).append(point)

    base_entries = finals.get("base", {})
    # Anchor every mode's curve at step 0 with the base policy's score: before any
    # DPO steps the LoRA-adapted model is the base model, so step 0 is a shared,
    # meaningful starting point for all construction modes.
    for metric_points in curves.values():
        for key, points in metric_points.items():
            if key in base_entries:
                points.append({"step": 0, **base_entries[key]})
            points.sort(key=lambda point: point["step"])

    return {"finals": finals, "curves": curves, "metric_keys": sorted(keys)}


def _should_replace_eval(
    *,
    current: tuple[int | None, int, dict[str, t.Any]] | None,
    iterations: int | None,
    order: int,
) -> bool:
    """Return whether a duplicate EuroEval result should replace the current one.

    Args:
        current:
          The currently selected ``(iterations, order, entry)`` tuple, if any.
        iterations:
          Iteration count for the candidate result, if present in the row.
        order:
          Monotonic result order within the JSONL file.

    Returns:
        True when the candidate should be kept.
    """
    if current is None:
        return True
    current_iterations, current_order, _ = current
    if current_iterations is not None and iterations is not None:
        if iterations != current_iterations:
            return iterations > current_iterations
    return order > current_order


def _iteration_count(
    *, record: dict[str, t.Any], result: dict[str, t.Any]
) -> int | None:
    """Extract the EuroEval iteration count from known row metadata locations.

    Args:
        record:
          The top-level JSONL row.
        result:
          One entry from the row's ``evaluation_results`` list.

    Returns:
        Iteration count when available, otherwise None.
    """
    details = result.get("score_details", {})
    uncertainty = details.get("uncertainty", {}) if isinstance(details, dict) else {}
    value = _int_field(data=uncertainty, fields=_SAMPLE_COUNT_FIELDS)
    if value is not None:
        return value

    for container in (result, record):
        value = _raw_results_count(data=container)
        if value is not None:
            return value

    containers = (
        record,
        record.get("metadata"),
        record.get("benchmark_config"),
        record.get("config"),
        result,
        details,
        uncertainty,
    )
    for container in containers:
        value = _int_field(data=container, fields=_ITERATION_FIELDS)
        if value is not None:
            return value
    return None


def _raw_results_count(*, data: object) -> int | None:
    """Return the number of raw EuroEval per-iteration results if present.

    Args:
        data:
          Candidate mapping containing EuroEval library details.

    Returns:
        Number of raw result entries, or None when unavailable.
    """
    if not isinstance(data, dict):
        return None
    eval_library = data.get("eval_library")
    if not isinstance(eval_library, dict):
        return None
    additional_details = eval_library.get("additional_details")
    if not isinstance(additional_details, dict):
        return None
    raw_results = additional_details.get("raw_results")
    if isinstance(raw_results, str):
        try:
            raw_results = json.loads(raw_results)
        except json.JSONDecodeError:
            return None
    if isinstance(raw_results, list):
        return len(raw_results)
    return None


def _int_field(*, data: object, fields: tuple[str, ...]) -> int | None:
    """Return the first integer-like field in a mapping.

    Args:
        data:
          Candidate metadata mapping.
        fields:
          Field names to inspect.

    Returns:
        The parsed integer value, or None.
    """
    if not isinstance(data, dict):
        return None
    for field in fields:
        value = data.get(field)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdecimal():
            return int(value)
    return None


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
    if not _include_result_model_id(model_id=model_id):
        return None
    if model_id.rstrip("/").endswith("munin-apertus-8b"):
        return "base"
    for marker, mode in _MODE_MARKERS:
        if marker in model_id:
            return mode
    return None


def _include_result_model_id(*, model_id: str) -> bool:
    """Return whether a EuroEval model id belongs on the dashboard.

    Args:
        model_id:
          The ``model_info.id`` from a results record.

    Returns:
        True for full experiment rows, false for micro/smoke fixture rows.
    """
    path = Path(model_id.rstrip("/"))
    name = path.parent.name if _CHECKPOINT_RE.fullmatch(path.name) else path.name
    return "micro" not in name.lower() and "smoke" not in name.lower()


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
            ["ssh", host, command], capture_output=True, text=True, check=False
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
  .mode-selector { margin: 16px 0; padding: 12px; background: #f5f5f5;
                   border-radius: 8px; }
  .mode-selector label { margin-right: 14px; cursor: pointer; user-select: none; }
  .mode-selector input[type="checkbox"] { margin-right: 4px; }
</style>
</head>
<body>
<h1>CroCo experiment dashboard</h1>
<div class="meta">Generated <span id="gen"></span> &middot; auto-refreshes every
  __REFRESH__s &middot; hover a chart and use the camera icon to save a PNG</div>

<div class="mode-selector">
  <strong>Show modes:</strong>
  <div id="modeCheckboxes" style="margin-top: 8px;"></div>
</div>

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
  generated: "#ff7f0e", label_smoothing: "#2ca02c", sigmoid_norm: "#9467bd",
  grpo: "#8c564b", simpo_tuned: "#e377c2", simpo_full: "#17becf",
  llama_rm: "#bcbd22"};
const MODE_LABELS = {
  max_reward: "max_reward",
  gold_chosen: "gold_chosen",
  generated: "generated",
  label_smoothing: "label_smoothing",
  sigmoid_norm: "SimPO",
  grpo: "GRPO",
  llama_rm: "Llama RM",
  grpo: "GRPO",
  simpo_tuned: "SimPO-tuned",
  simpo_full: "SimPO-full"
};

function getSelectedModes() {
  const stored = localStorage.getItem("croco_selected_modes");
  if (stored) return JSON.parse(stored);
  return Object.keys(COLOURS); // all selected by default
}

function setSelectedModes(modes) {
  localStorage.setItem("croco_selected_modes", JSON.stringify(modes));
}

function renderModeSelector() {
  const container = document.getElementById("modeCheckboxes");
  const selected = getSelectedModes();
  // Only show modes that have data in at least one of: training, curves, or finals
  const modesWithResults = new Set();
  for (const mode of Object.keys(DATA.training || {})) {
    if (!mode.startsWith("base")) modesWithResults.add(mode);
  }
  for (const mode of Object.keys(DATA.curves || {})) {
    if (!mode.startsWith("base")) modesWithResults.add(mode);
  }
  for (const mode of Object.keys(DATA.finals || {})) {
    if (!mode.startsWith("base")) modesWithResults.add(mode);
  }
  const modes = Object.keys(COLOURS)
    .filter(m => !m.startsWith("base"))
    .filter(m => modesWithResults.has(m));
  if (!modes.length) {
    container.innerHTML = '<span class="note">No experiment results yet.</span>';
    return;
  }
  modes.forEach(mode => {
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = mode;
    cb.checked = selected.includes(mode);
    cb.onchange = () => {
      const nowSelected = Array.from(container.querySelectorAll("input:checked"))
        .map(el => el.value);
      setSelectedModes(nowSelected);
      updateAllPlots();
    };
    label.appendChild(cb);
    label.appendChild(document.createTextNode(MODE_LABELS[mode] || mode));
    label.style.color = COLOURS[mode];
    container.appendChild(label);
  });
}

function updateAllPlots() {
  trainingPlots();
  curves();
  finals();
}
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
  const selected = getSelectedModes();
  for (const [div, key, title, ylab] of specs) {
    const traces = Object.entries(DATA.training)
      .filter(([mode]) => selected.includes(mode) || mode.startsWith("base"))
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
  // Dodge each run by a small horizontal offset so the (vertical) confidence
  // intervals sit side-by-side instead of overlapping; hover still reports the
  // true checkpoint step.
  const selected = getSelectedModes();
  const present = Object.entries(DATA.curves)
    .filter(([, mm]) => mm[metric])
    .filter(([mode]) => selected.includes(mode) || mode.startsWith("base"));
  const spread = 4;
  present.forEach(([mode, metrics], i) => {
    const pts = metrics[metric];
    const dx = (i - (present.length - 1) / 2) * spread;
    const hasCI = pts.every(p => p.lower != null && p.upper != null);
    traces.push({x: pts.map(p => p.step + dx), y: pts.map(p => p.score),
      customdata: pts.map(p => p.step),
      mode: "lines+markers", name: mode, line: {color: COLOURS[mode]},
      hovertemplate: "step %{customdata}: %{y:.2f}<extra>" + mode + "</extra>",
      error_y: hasCI ? {type: "data", symmetric: false,
        array: pts.map(p => p.upper - p.score),
        arrayminus: pts.map(p => p.score - p.lower),
        color: "black", thickness: 2.5, width: 4} : undefined});
  });
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
  const selected = getSelectedModes();
  const labels = Object.keys(DATA.finals)
    .filter(l => selected.includes(l) || l.startsWith("base"));
  if (!labels.length) {
    document.getElementById("finals").innerHTML =
      '<div class="note">No final evaluations yet.</div>'; return;
  }
  const baseLabel = labels.find(l => l.startsWith("base"));
  const base = baseLabel ? DATA.finals[baseLabel] : null;
  // Nested groups via a multicategory y-axis: outer = dataset, inner = metric,
  // construction modes are the grouped bars at each (dataset, metric) leaf.
  // Plotly draws a divider and wider gap between datasets automatically. The
  // significance arrow rides the metric (inner) label and is set when ANY
  // non-base method at that (dataset, metric) leaf is significantly better /
  // worse than base.
  const allKeys = [...new Set(labels.flatMap(l => Object.keys(DATA.finals[l])))];
  const byDs = {};
  for (const key of allKeys) {
    const ds = key.split("||")[0];
    if (!byDs[ds]) byDs[ds] = [];
    byDs[ds].push(key);
  }
  const datasets = Object.keys(byDs).sort();
  for (const ds of datasets) byDs[ds].sort();
  // One triangle per non-base mode, coloured by that mode and pinned to a fixed
  // column, so a single metric can show several at once (e.g. one mode
  // significantly better, another significantly worse). Each mode keeps the same
  // column across rows; modes with no significant difference at a leaf get a
  // blank (NBSP) slot so the coloured triangles stay vertically aligned per mode.
  const nonBase = labels.filter(l => l !== baseLabel).sort();
  const innerOf = {};
  for (const key of allKeys) {
    const metric = key.split("||")[1].replace(/^test_/, "");
    const slots = nonBase.map(label => {
      const s = sigVsBase(DATA.finals[label][key], base ? base[key] : null);
      if (s === 0) return "\u00A0";
      const colour = COLOURS[label] || "#000";
      return `<span style="color:${colour}">${s > 0 ? "▲" : "▼"}</span>`;
    });
    innerOf[key] = nonBase.length ? `${metric}\u00A0${slots.join("")}` : metric;
  }
  const leaves = [];
  for (const ds of datasets) {
    for (const key of byDs[ds]) {
      leaves.push({ds, key, metric: key.split("||")[1].replace(/^test_/, "")});
    }
  }
  const traces = [];
  for (const label of labels) {
    const mode = label.startsWith("base") ? "base" : label;
    const outer = [], inner = [], x = [], up = [], dn = [], cd = [];
    for (const leaf of leaves) {
      const r = DATA.finals[label][leaf.key];
      outer.push(leaf.ds); inner.push(innerOf[leaf.key]);
      x.push(r ? r.score : null);
      up.push(r && r.upper != null ? r.upper - r.score : 0);
      dn.push(r && r.lower != null ? r.score - r.lower : 0);
      cd.push(`${leaf.ds} / ${leaf.metric}`);
    }
    traces.push({type: "bar", orientation: "h", name: label,
      legendgroup: label, y: [outer, inner], x, customdata: cd,
      marker: {color: COLOURS[mode]},
      hovertemplate: "%{customdata}: %{x:.3f}<extra>" + label + "</extra>",
      error_x: {type: "data", symmetric: false, color: "black", thickness: 1,
        array: up, arrayminus: dn}});
  }
  const height = Math.max(420, 60 + leaves.length * labels.length * 22);
  document.getElementById("finals").style.height = `${height}px`;
  Plotly.newPlot("finals", traces,
    layout("Final EuroEval scores (▲ better / ▼ worse than base, " +
      "by mode colour; 95% CI)",
      "score", "", {barmode: "group", bargap: 0.15, bargroupgap: 0.12, height,
      margin: {t: 40, r: 40, b: 40, l: 10},
      yaxis: {automargin: true, autorange: "reversed",
        tickfont: {family: "monospace", size: 11}}}), CFG);
}

renderModeSelector();
trainingPlots(); curves(); finals();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
