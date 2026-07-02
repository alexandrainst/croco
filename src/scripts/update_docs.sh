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

COLOURS = {"max_reward": "#1f77b4", "gold_chosen": "#d62728",
           "generated": "#ff7f0e", "label_smoothing": "#2ca02c"}

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
                mode="lines+markers", name=mode, marker=dict(size=4)))
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
        has_ci = all(p.get("lower") and p.get("upper") for p in pts)
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

# 3. Final comparison
all_benchmarks = set()
for exp, scores in DATA["finals"].items(): all_benchmarks.update(scores.keys())
experiments = [k for k in DATA["finals"] if not k.startswith("base")]
fig = go.Figure()
for exp in experiments:
    scores = DATA["finals"][exp]
    y, emin, emax, labels = [], [], [], []
    for b in sorted(all_benchmarks):
        if b in scores:
            r = scores[b]
            y.append(r["score"])
            emin.append(r["score"] - r.get("lower", r["score"]))
            emax.append(r.get("upper", r["score"]) - r["score"])
            labels.append(b)
    fig.add_trace(go.Bar(name=exp, x=labels, y=y,
        error_y=dict(type='data', symmetric=False, array=emax, arrayminus=emin,
                    width=0.5, thickness=1.5), marker_color=COLOURS.get(exp, "#000")))
fig.update_layout(title="Final EuroEval Comparison", xaxis_title="Benchmark",
    yaxis_title="Score (95% CI)", barmode="group", height=500, width=1400,
    legend=dict(orientation="h", y=-0.15), xaxis=dict(tickangle=-45, tickfont=dict(size=8)))
fig.write_image(gfx / "final_comparison.png", scale=2)
print(f"  Exported: final_comparison.png")

print(f"\nExported {len(list(gfx.glob('*.png')))} plots to {gfx}/")
PYEOF

echo "[3/4] Cleaning up old plot files..."
rm -f docs/gfx/curve_*_no_metric.png 2>/dev/null || true

echo "[4/4] Done! Commit with:"
echo "  git add docs/gfx/*.png docs/*.md"
echo "  git commit -m 'docs: update plots'"
