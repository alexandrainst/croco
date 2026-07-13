"""Tests for the checkpoint EuroEval wrapper script."""

import importlib.util
import subprocess
import types
from pathlib import Path

import pytest
from click.testing import CliRunner


def _load_eval_checkpoints() -> types.ModuleType:
    """Load the script module from src/scripts.

    Returns:
        Loaded eval_checkpoints module.

    Raises:
        RuntimeError:
          If the script module cannot be loaded.
    """
    script_path = Path("src/scripts/eval_checkpoints.py")
    spec = importlib.util.spec_from_file_location("eval_checkpoints", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_model_dir(tmp_path: Path) -> Path:
    """Create a model directory with one evaluable checkpoint.

    Args:
        tmp_path:
          Temporary test directory.

    Returns:
        Model directory containing one checkpoint adapter.
    """
    checkpoint = tmp_path / "model" / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}\n")
    return tmp_path / "model"


@pytest.mark.parametrize(("force_args", "expected"), [([], False), (["--force"], True)])
def test_cli_forwards_force_only_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    force_args: list[str],
    expected: bool,
) -> None:
    """The CLI should forward force only when the user passes --force."""
    module = _load_eval_checkpoints()
    model_dir = _checkpoint_model_dir(tmp_path=tmp_path)
    seen_force: list[object] = []

    def fake_run_euroeval(**kwargs: object) -> None:
        seen_force.append(kwargs["force"])

    monkeypatch.setattr(module, "_run_euroeval", fake_run_euroeval)

    result = CliRunner().invoke(
        module.main, ["-m", str(model_dir), "--no-include-final", *force_args]
    )

    assert result.exit_code == 0, result.output
    assert seen_force == [expected]


def test_run_euroeval_appends_force_only_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EuroEval receives --force only when the wrapper force flag is true."""
    module = _load_eval_checkpoints()
    commands: list[list[str]] = []

    def fake_run(cmd: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        assert check is True
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    for force in (False, True):
        module._run_euroeval(
            model_path=tmp_path / "checkpoint-100",
            language="da",
            tasks=(),
            gpu_memory_utilization=0.5,
            force=force,
        )

    assert "--force" not in commands[0]
    assert commands[1][-1] == "--force"
