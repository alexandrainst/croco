#!/usr/bin/env bash
# Export dashboard plots and update experiment documentation.
#
# Usage:
#   uv run src/scripts/update_docs.sh
#
# This script:
#   1. Reads croco_dashboard.html (must exist)
#   2. Exports 22 PNG plots to docs/gfx/:
#      - Training: loss, accuracy, margins (3)
#      - Learning curves: all 18 dataset-metric combinations
#      - Final: comparison bar chart (1)
#   3. Cleans up outdated plot files
#
# Prerequisites:
#   - Dashboard exists: python src/scripts/build_dashboard.py
#   - plotly, kaleido installed

set -euo pipefail
cd "$(dirname "$0")/../.."

GFX_DIR="docs/gfx"
DASHBOARD="croco_dashboard.html"

echo "[1/4] Checking dashboard..."
if [[ ! -f "$DASHBOARD" ]]; then
    echo "  Dashboard not found. Run: python src/scripts/build_dashboard.py"
    exit 1
fi

echo "[2/4] Exporting plots..."
.venv/bin/python3 << 'PYEOF'
import json, re
from pathlib import Path
import plotly.graph_objects as go
import logging
logging.getLogger("kaleido").setLevel(logging.ERROR)

html = Path("croco_dashboard.html").read_text()
match = re.search(r'const DATA = ({.+?});', html, re.DOTALL)
DATA = json.loads(match.group(1))

gfx = Path("docs/gfx")
gfx.mkdir(exist_ok=True)

# Keep in sync with the COLOURS map in build_dashboard.py's HTML template.
COLOURS = {"max_reward": "#1f77b4", "gold_chosen": "#d62728", "base": "#7f7f7f",
           "generated": "#ff7f0e", "label_smoothing": "#2ca02c",
           "sigmoid_norm": "#9467bd", "grpo": "#8c564b",
           "simpo_tuned": "#e377c2", "simpo_full": "#17becf"}

# 1. Training dynamics
for name, key, title, ylab in [
    ("loss", "loss", "DPO loss", "loss"),
    ("accuracy", "acc", "Preference accuracy", "accuracy"),
    ("margins", "margins", "Reward margin", "margin"),
]:
    fig = go.Figure()
    for mode, data in DATA["training"].items():
        if key in data and data[key]:
            fig.add_trace(go.Scatter(x=data["steps"], y=data[key],
                mode="lines+markers", name=mode, marker=dict(size=4),
                line=dict(color=COLOURS.get(mode))))
    fig.update_layout(title=title, xaxis_title="step", yaxis_title=ylab,
        height=400, width=1200, legend=dict(orientation="h", y=-0.2))
    if name == "accuracy": fig.update_yaxes(range=[0, 1])
    fig.write_image(gfx / f"training_{name}.png", scale=2)
    print(f"  Exported: training_{name}.png")

# 2. All 18 learning curves (dataset-metric combinations)
ALL_METRICS = [
    ("angry-tweets", "test_macro_f1"), ("angry-tweets", "test_mcc"),
    ("danish-citizen-tests", "test_accuracy"), ("danish-citizen-tests", "test_mcc"),
    ("dansk", "test_micro_f1"), ("dansk", "test_micro_f1_no_misc"),
    ("danske-talemaader", "test_accuracy"), ("danske-talemaader", "test_mcc"),
    ("hellaswag-da", "test_accuracy"), ("hellaswag-da", "test_mcc"),
    ("ifeval-da", "test_instruction_accuracy"),
    ("multi-wiki-qa-da", "test_em"), ("multi-wiki-qa-da", "test_f1"),
    ("nordjylland-news", "test_chr_f3pp"), ("nordjylland-news", "test_chr_f4pp"),
    ("scala-da", "test_macro_f1"), ("scala-da", "test_mcc"),
    ("valeu-da", "test_european_values"),
]
for dataset, metric in ALL_METRICS:
    key = f"{dataset}||{metric}"
    present = [(m, d[key]) for m, d in DATA["curves"].items() if key in d]
    if not present: continue
    fig = go.Figure()
    spread = 4
    for i, (mode, pts) in enumerate(present):
        dx = (i - (len(present)-1)/2) * spread
        has_ci = all(p.get("lower") is not None and p.get("upper") is not None for p in pts)
        fig.add_trace(go.Scatter(x=[p["step"]+dx for p in pts], y=[p["score"] for p in pts],
            customdata=[p["step"] for p in pts], mode="lines+markers", name=mode,
            line=dict(color=COLOURS.get(mode)), marker=dict(size=5),
            error_y=dict(type='data', symmetric=False,
                array=[p["upper"]-p["score"] for p in pts],
                arrayminus=[p["score"]-p["lower"] for p in pts],
                width=3, thickness=2) if has_ci else None))
    fig.update_layout(title=f"{dataset} - {metric}", xaxis_title="step", yaxis_title=metric,
        height=400, width=1200, legend=dict(orientation="h", y=-0.2))
    safe = f"{dataset}-{metric}".replace("/", "-")
    fig.write_image(gfx / f"curve_{safe}.png", scale=2)
    print(f"  Exported: curve_{safe}.png")

# 3. Final comparison — base model plus every mode evaluated on all its benchmarks.
# Base is included so the before/after gap (and thus significance) is visible; a
# mode still being benchmarked (partial coverage) is skipped until it is complete.
all_benchmarks = sorted(set().union(*(s.keys() for s in DATA["finals"].values())))
base_label = next((k for k in DATA["finals"] if k.startswith("base")), None)
base_keys = set(DATA["finals"][base_label]) if base_label else set(all_benchmarks)
modes = ([base_label] if base_label else []) + sorted(
    label for label in DATA["finals"]
    if label != base_label and base_keys.issubset(DATA["finals"][label]))
pretty = lambda b: f'{b.split("||")[0]} / {b.split("||")[1].replace("test_", "")}'
fig = go.Figure()
for label in modes:
    scores = DATA["finals"][label]
    y, emin, emax = [], [], []
    for b in all_benchmarks:
        r = scores.get(b)
        y.append(r["score"] if r else None)
        emin.append(r["score"] - r["lower"] if r and r.get("lower") is not None else 0)
        emax.append(r["upper"] - r["score"] if r and r.get("upper") is not None else 0)
    colour = COLOURS.get("base" if label.startswith("base") else label, "#000")
    fig.add_trace(go.Bar(name=label, x=[pretty(b) for b in all_benchmarks], y=y,
        error_y=dict(type='data', symmetric=False, array=emax, arrayminus=emin,
                    width=0.5, thickness=1.5), marker_color=colour))
fig.update_layout(title="Final EuroEval comparison (mean ± 95% CI)",
    xaxis_title="Benchmark / metric", yaxis_title="Score", barmode="group",
    height=500, width=1400, legend=dict(orientation="h", y=-0.15),
    xaxis=dict(tickangle=-45, tickfont=dict(size=8)))
fig.write_image(gfx / "final_comparison.png", scale=2)
print(f"  Exported: final_comparison.png")

print(f"\nExported {len(list(gfx.glob('*.png')))} plots to {gfx}/")
PYEOF

echo "[3/4] Cleaning up old plot files..."
rm -f docs/gfx/curve_*_no_metric.png 2>/dev/null || true

echo "[4/4] Done! Commit with:"
echo "  git add docs/gfx/*.png docs/*.md"
echo "  git commit -m 'docs: update plots'"
