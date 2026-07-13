"""Tests for the dashboard model-id / directory classification and EuroEval data."""

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_dashboard",
    Path(__file__).resolve().parent.parent / "src" / "scripts" / "build_dashboard.py",
)
assert _SPEC is not None and _SPEC.loader is not None
build_dashboard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_dashboard)


class _FakeReader:
    """Reader stub returning a fixed JSONL payload."""

    def __init__(self, *, text: str) -> None:
        """Store the payload returned by ``read_text``."""
        self.text = text

    def read_text(self, *, path: str) -> str | None:
        """Return the configured payload."""
        return self.text


def _result_record(
    *,
    model_id: str,
    score: float,
    num_samples: int | None = None,
    raw_results: str | None = None,
    lower: float | None = None,
    upper: float | None = None,
    dataset: str = "angry-tweets-test",
    metric: str = "macro_f1",
) -> str:
    """Build one minimal EuroEval JSONL record for parser tests.

    Returns:
        JSON-encoded EuroEval result row.
    """
    if lower is None:
        lower = score - 0.01
    if upper is None:
        upper = score + 0.01
    uncertainty: dict[str, object] = {
        "confidence_interval": {"lower": lower, "upper": upper}
    }
    if num_samples is not None:
        uncertainty["num_samples"] = num_samples
    result: dict[str, object] = {
        "source_data": {"dataset_name": dataset},
        "evaluation_name": metric,
        "score_details": {
            "score": score,
            "uncertainty": uncertainty,
        },
        "metric_config": {"lower_is_better": False},
    }
    if raw_results is not None:
        result["eval_library"] = {
            "additional_details": {"raw_results": raw_results}
        }
    record: dict[str, object] = {
        "model_info": {"id": model_id},
        "evaluation_results": [result],
    }
    return json.dumps(record)


def _eval_series(*, rows: tuple[str, ...]) -> dict[str, object]:
    """Parse minimal EuroEval rows through the dashboard parser.

    Returns:
        Parsed dashboard eval data.
    """
    return build_dashboard._eval_series(
        reader=_FakeReader(text="\n".join(rows)), results="unused.jsonl"
    )


class TestResultMode:
    """The ablation suffixes must not be swallowed by the bare max_reward tail."""

    def test_base_model(self) -> None:
        """The base policy maps to ``base``."""
        assert (
            build_dashboard._result_mode(
                model_id="danish-foundation-models/munin-apertus-8b"
            )
            == "base"
        )

    def test_each_mode_is_distinct(self) -> None:
        """Each model directory classifies to its own mode, not max_reward."""
        cases = {
            "models/croco-munin-apertus-8b-da": "max_reward",
            "models/croco-munin-apertus-8b-da-gold": "gold_chosen",
            "models/croco-munin-apertus-8b-da-generated": "generated",
            "models/croco-munin-apertus-8b-da-ls": "label_smoothing",
            "models/croco-munin-apertus-8b-da-simpo-tuned": "simpo_tuned",
            "models/croco-munin-apertus-8b-da-simpo-full": "simpo_full",
            "models/croco-munin-apertus-8b-da-simpo": "sigmoid_norm",
            "models/croco-munin-apertus-8b-da-grpo": "grpo",
        }
        for model_id, expected in cases.items():
            assert build_dashboard._result_mode(model_id=model_id) == expected

    def test_checkpoint_ids_classify_like_their_dir(self) -> None:
        """A checkpoint id keeps its parent directory's mode."""
        assert (
            build_dashboard._result_mode(
                model_id="models/croco-munin-apertus-8b-da-ls/checkpoint-100"
            )
            == "label_smoothing"
        )
        assert (
            build_dashboard._result_mode(
                model_id="models/croco-munin-apertus-8b-da-simpo-tuned/checkpoint-100"
            )
            == "simpo_tuned"
        )

    def test_bare_simpo_dir_still_sigmoid_norm(self) -> None:
        """The bare ``-simpo`` dir must keep its sigmoid_norm classification.

        Regression guard for the most-specific-first marker ordering: the new
        ``-simpo-tuned``/``-simpo-full`` markers must not shadow it, and the
        bare ``-simpo`` marker must not swallow them.
        """
        assert (
            build_dashboard._result_mode(
                model_id="models/croco-munin-apertus-8b-da-simpo"
            )
            == "sigmoid_norm"
        )
        assert (
            build_dashboard._mode_label("croco-munin-apertus-8b-da-simpo")
            == "sigmoid_norm"
        )

    def test_micro_and_smoke_result_ids_are_ignored(self) -> None:
        """Micro/smoke result ids do not fall through to real modes."""
        assert (
            build_dashboard._result_mode(
                model_id="models/croco-munin-apertus-8b-da-micro/checkpoint-100"
            )
            is None
        )
        assert (
            build_dashboard._result_mode(
                model_id="models/croco-munin-apertus-8b-da-simpo-tuned-smoke/checkpoint-100"
            )
            is None
        )

    def test_unrelated_model_is_ignored(self) -> None:
        """Unrelated models classify to None so they are dropped."""
        assert build_dashboard._result_mode(model_id="google/gemma-3-12b-it") is None


class TestModeLabel:
    """Directory names map to the same mode labels as result ids."""

    def test_directory_names(self) -> None:
        """Each directory name maps to its construction-mode label."""
        assert build_dashboard._mode_label("croco-munin-apertus-8b-da") == "max_reward"
        assert (
            build_dashboard._mode_label("croco-munin-apertus-8b-da-gold")
            == "gold_chosen"
        )
        assert (
            build_dashboard._mode_label("croco-munin-apertus-8b-da-simpo")
            == "sigmoid_norm"
        )
        assert (
            build_dashboard._mode_label("croco-munin-apertus-8b-da-simpo-tuned")
            == "simpo_tuned"
        )
        assert (
            build_dashboard._mode_label("croco-munin-apertus-8b-da-simpo-full")
            == "simpo_full"
        )
        assert build_dashboard._mode_label("croco-munin-apertus-8b-da-grpo") == "grpo"


class TestEvalSeries:
    """EuroEval rows are grouped and deduplicated for dashboard data."""

    def test_simpo_tuned_checkpoint_and_final_rows_are_included(self) -> None:
        """SimPO-tuned checkpoint and final evals use the simpo_tuned key."""
        key = "angry-tweets-test||macro_f1"
        series = _eval_series(
            rows=(
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da-simpo-tuned/checkpoint-100",
                    score=0.70,
                ),
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da-simpo-tuned", score=0.80
                ),
            )
        )

        assert series["curves"]["simpo_tuned"][key][0]["step"] == 100
        assert series["curves"]["simpo_tuned"][key][0]["score"] == 0.70
        assert series["finals"]["simpo_tuned"][key]["score"] == 0.80

    def test_micro_eval_rows_are_ignored(self) -> None:
        """Micro eval rows do not pollute max_reward curves or finals."""
        key = "angry-tweets-test||macro_f1"
        series = _eval_series(
            rows=(
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da/checkpoint-100",
                    score=0.10,
                    lower=0.09,
                    upper=0.11,
                ),
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da-micro/checkpoint-100",
                    score=0.90,
                ),
                _result_record(model_id="models/croco-munin-apertus-8b-da", score=0.50),
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da-micro",
                    score=0.99,
                ),
            )
        )

        assert series["curves"]["max_reward"][key] == [
            {
                "step": 100,
                "score": 0.10,
                "lower": 0.09,
                "upper": 0.11,
                "lower_is_better": False,
            }
        ]
        assert series["finals"]["max_reward"][key]["score"] == 0.50

    def test_smoke_eval_rows_are_ignored(self) -> None:
        """Smoke eval rows do not pollute SimPO-tuned curves or finals."""
        key = "angry-tweets-test||macro_f1"
        series = _eval_series(
            rows=(
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da-simpo-tuned/checkpoint-100",
                    score=0.70,
                ),
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da-simpo-tuned-smoke/checkpoint-100",
                    score=0.96,
                ),
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da-simpo-tuned",
                    score=0.80,
                ),
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da-simpo-tuned-smoke",
                    score=0.95,
                ),
            )
        )

        assert series["curves"]["simpo_tuned"][key][0]["step"] == 100
        assert series["curves"]["simpo_tuned"][key][0]["score"] == 0.70
        assert len(series["curves"]["simpo_tuned"][key]) == 1
        assert series["finals"]["simpo_tuned"][key]["score"] == 0.80

    def test_duplicate_rows_prefer_higher_sample_count(self) -> None:
        """Duplicate checkpoint/final scores keep the higher sample count row."""
        key = "angry-tweets-test||macro_f1"
        ten_checkpoint = _result_record(
            model_id="models/croco-munin-apertus-8b-da/checkpoint-100",
            score=0.10,
            num_samples=10,
            lower=0.09,
            upper=0.11,
        )
        three_checkpoint = _result_record(
            model_id="models/croco-munin-apertus-8b-da/checkpoint-100",
            score=0.30,
            num_samples=3,
            lower=0.20,
            upper=0.40,
        )
        ten_final = _result_record(
            model_id="models/croco-munin-apertus-8b-da",
            score=0.50,
            num_samples=10,
            lower=0.49,
            upper=0.51,
        )
        three_final = _result_record(
            model_id="models/croco-munin-apertus-8b-da",
            score=0.70,
            num_samples=3,
            lower=0.60,
            upper=0.80,
        )

        for rows in (
            (ten_checkpoint, three_checkpoint, ten_final, three_final),
            (three_checkpoint, ten_checkpoint, three_final, ten_final),
        ):
            series = _eval_series(rows=rows)
            point = series["curves"]["max_reward"][key][0]
            final = series["finals"]["max_reward"][key]
            assert point["score"] == 0.10
            assert point["lower"] == 0.09
            assert point["upper"] == 0.11
            assert final["score"] == 0.50
            assert final["lower"] == 0.49
            assert final["upper"] == 0.51

    def test_eee_num_samples_prefers_higher_count(self) -> None:
        """EEE uncertainty sample counts keep the higher-count duplicate."""
        key = "angry-tweets-test||macro_f1"
        ten_checkpoint = _result_record(
            model_id="models/croco-munin-apertus-8b-da/checkpoint-100",
            score=0.10,
            num_samples=10,
        )
        three_checkpoint = _result_record(
            model_id="models/croco-munin-apertus-8b-da/checkpoint-100",
            score=0.30,
            num_samples=3,
        )
        ten_final = _result_record(
            model_id="models/croco-munin-apertus-8b-da",
            score=0.50,
            num_samples=10,
        )
        three_final = _result_record(
            model_id="models/croco-munin-apertus-8b-da",
            score=0.70,
            num_samples=3,
        )

        series = _eval_series(
            rows=(ten_checkpoint, three_checkpoint, ten_final, three_final)
        )

        assert series["curves"]["max_reward"][key][0]["score"] == 0.10
        assert series["finals"]["max_reward"][key]["score"] == 0.50

    def test_eee_raw_results_fallback_prefers_higher_count(self) -> None:
        """EEE raw per-iteration results are used when sample counts are absent."""
        key = "angry-tweets-test||macro_f1"
        ten_raw_results = json.dumps(
            [{"iteration": iteration} for iteration in range(10)]
        )
        three_raw_results = json.dumps(
            [{"iteration": iteration} for iteration in range(3)]
        )
        series = _eval_series(
            rows=(
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da/checkpoint-100",
                    score=0.10,
                    raw_results=ten_raw_results,
                ),
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da/checkpoint-100",
                    score=0.30,
                    raw_results=three_raw_results,
                ),
            )
        )

        assert series["curves"]["max_reward"][key][0]["score"] == 0.10

    def test_invalid_raw_results_keep_later_row_wins(self) -> None:
        """Invalid EEE raw results preserve later-row-wins replacement."""
        key = "angry-tweets-test||macro_f1"
        series = _eval_series(
            rows=(
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da/checkpoint-100",
                    score=0.10,
                    raw_results="not JSON",
                ),
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da/checkpoint-100",
                    score=0.20,
                    raw_results=json.dumps({"iteration": 1}),
                ),
            )
        )

        assert series["curves"]["max_reward"][key][0]["score"] == 0.20

    def test_tied_iteration_count_keeps_later_row_wins(self) -> None:
        """Rows with equal iteration counts preserve the old later-row-wins rule."""
        key = "angry-tweets-test||macro_f1"
        series = _eval_series(
            rows=(
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da/checkpoint-100",
                    score=0.10,
                    num_samples=10,
                ),
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da/checkpoint-100",
                    score=0.20,
                    num_samples=10,
                ),
            )
        )

        assert series["curves"]["max_reward"][key][0]["score"] == 0.20

    def test_missing_sample_count_keeps_later_row_wins(self) -> None:
        """Rows missing sample count metadata preserve the old later-row-wins rule."""
        key = "angry-tweets-test||macro_f1"
        series = _eval_series(
            rows=(
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da/checkpoint-100",
                    score=0.10,
                ),
                _result_record(
                    model_id="models/croco-munin-apertus-8b-da/checkpoint-100",
                    score=0.20,
                ),
            )
        )

        assert series["curves"]["max_reward"][key][0]["score"] == 0.20


class TestRenderingModes:
    """The browser rendering template knows every eval-only dashboard mode."""

    def test_simpo_tuned_is_available_to_curve_and_final_filters(self) -> None:
        """SimPO-tuned eval data can appear in the mode selector."""
        html = build_dashboard._render_html(
            data={
                "generated": "now",
                "training": {},
                "curves": {"simpo_tuned": {}},
                "finals": {"simpo_tuned": {}},
                "metric_keys": [],
            },
            refresh_seconds=60,
        )

        assert 'simpo_tuned: "#e377c2"' in html
        assert 'simpo_tuned: "SimPO-tuned"' in html


class TestDiscoveredModelDirs:
    """Auto-discovered fixture runs are excluded before mode labelling."""

    def test_micro_and_smoke_dirs_are_excluded(self) -> None:
        """Micro/smoke output dirs must not fall through to max_reward."""
        assert not build_dashboard._include_discovered_model_dir(
            output_dir="models/croco-munin-apertus-8b-da-micro"
        )
        assert not build_dashboard._include_discovered_model_dir(
            output_dir="models/croco-munin-apertus-8b-da-simpo-tuned-smoke"
        )
        assert not build_dashboard._include_discovered_model_dir(
            output_dir="models/_smoke_grpo"
        )
        assert build_dashboard._include_discovered_model_dir(
            output_dir="models/croco-munin-apertus-8b-da-simpo-tuned"
        )
