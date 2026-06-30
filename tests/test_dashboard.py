"""Tests for the dashboard model-id / directory classification."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_dashboard",
    Path(__file__).resolve().parent.parent / "src" / "scripts" / "build_dashboard.py",
)
assert _SPEC is not None and _SPEC.loader is not None
build_dashboard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_dashboard)


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
        assert build_dashboard._mode_label("croco-munin-apertus-8b-da-grpo") == "grpo"
