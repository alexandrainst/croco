#!/usr/bin/env bash
# Export dashboard plots and update experiment documentation.
#
# Usage:
#   ./src/scripts/update_docs.sh
#   # or from repo root:
#   uv run src/scripts/update_docs.sh
#
# This script:
#   1. Regenerates croco_dashboard.html with latest training/benchmark data
#   2. Exports training dynamics plots (loss, accuracy, margins)
#   3. Exports learning curves for all 10 EuroEval benchmarks
#   4. Exports final comparison bar chart
#   5. Copies PNGs to docs/gfx/
#
# Prerequisites:
#   - Dashboard must exist or be regeneratable via build_dashboard.py
#   - plotly, kaleido installed in current environment

set -euo pipefail
cd "$(dirname "$0")/../.."

SCRIPT_DIR="$(pwd)"
GFX_DIR="docs/gfx"
DASHBOARD="croco_dashboard.html"

echo "[1/4] Checking dashboard..."
if [[ ! -f "$DASHBOARD" ]]; then
    echo "  Dashboard not found. Run: python src/scripts/build_dashboard.py -m models/croco-munin-apertus-8b-da -r euroeval_benchmark_results.jsonl"
    exit 1
fi

echo "[2/4] Exporting plots..."
.venv/bin/python3 << 'PYEOF'
import json, re
from pathlib import Path
import plotly.graph_objects as go

import logging
logging.getLogger("kaleido").setLevel(logging.ERROR)

# Read dashboard
html = Path("croco_dashboard.html").read_text()
match = re.search(r'const DATA = ({.+?});', html, re.DOTALL)
DATA = json.loads(match.group(1))

gfx = Path("docs/gfx")
gfx.mkdir(exist_ok=True)

COLOURS = {
    "base": "#7f7f7f", "max_reward": "#1f77b4", "gold_chosen": "#d62728",
    "generated": "#ff7f0e", "label_smoothing": "#2ca02c", "sigmoid_norm": "#9467bd",
}

# 1. Training dynamics
for plot_name, key, title, ylab in [
    ("loss", "loss", "DPO loss", "loss"),
    ("accuracy", "acc", "Preference accuracy", "accuracy"),
    ("margins", "margins", "Reward margin", "margin"),
]:
    fig = go.Figure()
    for mode, data in DATA["training"].items():
        if key in data and data[key]:
            fig.add_trace(go.Scatter(x=data["steps"], y=data[key],
                mode="lines+markers", name=mode, marker=dict(size=4)))
    fig.update_layout(title=title, xaxis_title="step", yaxis_title=ylab,
        height=400, width=1200, legend=dict(orientation="h", y=-0.2))
    if plot_name == "accuracy": fig.update_yaxes(range=[0, 1])
    fig.write_image(gfx / f"training_{plot_name}.png", scale=2)
    print(f"  Exported: training_{plot_name}.png")

# 2. Learning curves (all 10 benchmarks)
PRIMARY_METRICS = {
    "angry-tweets": "test_mcc", "danish-citizen-tests": "test_mcc",
    "danske-talemaader": "test_mcc", "dansk": "test_micro_f1",
    "hellaswag-da": "test_mcc", "ifeval-da": "test_instruction_accuracy",
    "multi-wiki-qa-da": "test_f1", "nordjylland-news": "test_chr_f4pp",
    "scala-da": "test_mcc", "valeu-da": "test_european_values",
}
friendly_names = {
    "angry-tweets": "Angry Tweets", "danish-citizen-tests": "Danish Citizen Tests",
    "danske-talemaader": "Danske Talemåder", "dansk": "Dansk (NER)",
    "hellaswag-da": "Hellaswag-da", "ifeval-da": "IFEval-da",
    "multi-wiki-qa-da": "Multi-Wiki QA-da", "nordjylland-news": "Nordjylland News",
    "scala-da": "ScaLA-da", "valeu-da": "ValEU-da",
}
all_metrics = set()
for mode, metrics in DATA["curves"].items(): all_metrics.update(metrics.keys())

for dataset, primary_metric in PRIMARY_METRICS.items():
    metric_key = f"{dataset}||{primary_metric}"
    if metric_key not in all_metrics: continue
    fig = go.Figure()
    present = [(m, d[metric_key]) for m, d in DATA["curves"].items() if metric_key in d]
    spread = 4
    for i, (mode, pts) in enumerate(present):
        dx = (i - (len(present) - 1) / 2) * spread
        has_ci = all(p.get("lower") and p.get("upper") for p in pts)
        fig.add_trace(go.Scatter(x=[p["step"]+dx for p in pts], y=[p["score"] for p in pts],
            customdata=[p["step"] for p in pts], mode="lines+markers", name=mode,
            line=dict(color=COLOURS.get(mode)), marker=dict(size=5),
            error_y=dict(type='data', symmetric=False,
                array=[p["upper"]-p["score"] for p in pts],
                arrayminus=[p["score"]-p["lower"] for p in pts],
                width=3, thickness=2) if has_ci else None))
    fig.update_layout(title=f"{friendly_names.get(dataset, dataset)} - Learning Curve",
        xaxis_title="checkpoint step", yaxis_title=primary_metric,
        height=400, width=1200, legend=dict(orientation="h", y=-0.2))
    fig.write_image(gfx / f"curve_{dataset}.png", scale=2)
    print(f"  Exported: curve_{dataset}.png")

# 3. Final comparison
all_benchmarks = set()
for exp, scores in DATA["finals"].items(): all_benchmarks.update(scores.keys())
all_benchmarks = sorted(all_benchmarks)
experiments = [k for k in DATA["finals"].keys() if not k.startswith("base")]
base_data = DATA["finals"].get("base", {})
fig = go.Figure()
for exp in experiments:
    scores = DATA["finals"][exp]
    y_vals, err_min, err_max, labels = [], [], [], []
    for bench in all_benchmarks:
        if bench in scores:
            rec = scores[bench]
            y_vals.append(rec["score"])
            err_min.append(rec["score"] - rec.get("lower", rec["score"]))
            err_max.append(rec.get("upper", rec["score"]) - rec["score"])
            labels.append(bench)
    fig.add_trace(go.Bar(name=exp, x=labels, y=y_vals,
        error_y=dict(type='data', symmetric=False, array=err_max, arrayminus=err_min,
                    width=0.5, thickness=1.5),
        marker_color=COLOURS.get(exp, "#000")))
fig.update_layout(title="Final EuroEval Comparison (vs Base Model)",
    xaxis_title="Benchmark", yaxis_title="Score (with 95% CI)",
    barmode="group", height=500, width=1400,
    legend=dict(orientation="h", y=-0.15), xaxis=dict(tickangle=-45, tickfont=dict(size=8)))
fig.write_image(gfx / "final_comparison.png", scale=2)
print(f"  Exported: final_comparison.png")

print(f"\nExported {len(list(gfx.glob('*.png')))} plots to {gfx}/")
PYEOF

echo "[3/4] Cleaning up old plot files..."
rm -f docs/gfx/curve_*-test_*.png 2>/dev/null || true

echo "[4/4] Done! Commit with:"
echo "  git add docs/gfx/*.png docs/*.md"
echo "  git commit -m 'docs: update plots'"
