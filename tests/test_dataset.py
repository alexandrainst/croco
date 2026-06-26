"""Tests for CroCo dataset save/load and TRL conversion."""

from pathlib import Path

from croco.data_models import PreferencePair
from croco.dataset import load_pairs, save_pairs, to_trl_records


class TestSavePairs:
    """Tests for save_pairs function."""

    def test_save_pairs_basic(self, tmp_path: Path) -> None:
        """Test saving a basic list of preference pairs."""
        pairs = [
            PreferencePair(
                prompt="Test prompt 1",
                chosen="Chosen response 1",
                rejected="Rejected response 1",
                rejected_score=0.4,
                chosen_score=0.8,
                pool_size=4,
                mode="generated",
            ),
            PreferencePair(
                prompt="Test prompt 2",
                chosen="Chosen response 2",
                rejected="Rejected response 2",
                rejected_score=0.3,
                chosen_score=0.9,
                pool_size=5,
                mode="gold_chosen",
            ),
        ]

        output_path = tmp_path / "pairs.jsonl"
        save_pairs(pairs=pairs, path=output_path)

        assert output_path.exists()
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2

    def test_save_pairs_empty_list(self, tmp_path: Path) -> None:
        """Test saving an empty list of pairs."""
        output_path = tmp_path / "empty.jsonl"
        save_pairs(pairs=[], path=output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert content == ""

    def test_save_pairs_with_all_fields(self, tmp_path: Path) -> None:
        """Test saving pairs with all optional fields."""
        pairs = [
            PreferencePair(
                prompt="Complete prompt",
                chosen="Complete chosen",
                rejected="Complete rejected",
                rejected_score=0.5,
                chosen_score=0.7,
                evolution=3,
                pool_size=6,
                mode="generated",
                hash="abc123",
            )
        ]

        output_path = tmp_path / "complete.jsonl"
        save_pairs(pairs=pairs, path=output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "abc123" in content
        assert '"evolution": 3' in content or '"evolution":3' in content

    def test_save_pairs_with_nested_path(self, tmp_path: Path) -> None:
        """Test saving pairs to a nested path (parent dirs created by test)."""
        pairs = [
            PreferencePair(
                prompt="Test",
                chosen="Chosen",
                rejected="Rejected",
                rejected_score=0.5,
                pool_size=2,
                mode="generated",
            )
        ]

        output_path = tmp_path / "subdir" / "nested" / "pairs.jsonl"
        output_path.parent.mkdir(parents=True)
        save_pairs(pairs=pairs, path=output_path)

        assert output_path.exists()


class TestLoadPairs:
    """Tests for load_pairs function."""

    def test_load_pairs_basic(self, tmp_path: Path) -> None:
        """Test loading a basic list of preference pairs."""
        pairs = [
            PreferencePair(
                prompt="Test prompt 1",
                chosen="Chosen response 1",
                rejected="Rejected response 1",
                rejected_score=0.4,
                chosen_score=0.8,
                pool_size=4,
                mode="generated",
            ),
            PreferencePair(
                prompt="Test prompt 2",
                chosen="Chosen response 2",
                rejected="Rejected response 2",
                rejected_score=0.3,
                chosen_score=0.9,
                pool_size=5,
                mode="gold_chosen",
            ),
        ]

        output_path = tmp_path / "pairs.jsonl"
        save_pairs(pairs=pairs, path=output_path)

        loaded = load_pairs(path=output_path)

        assert len(loaded) == 2
        assert all(isinstance(pair, PreferencePair) for pair in loaded)

    def test_load_pairs_roundtrip(self, tmp_path: Path) -> None:
        """Test that save and load roundtrip preserves data."""
        original_pairs = [
            PreferencePair(
                prompt=f"Prompt {i}",
                chosen=f"Chosen {i}",
                rejected=f"Rejected {i}",
                rejected_score=0.3 + i * 0.1,
                chosen_score=0.7 + i * 0.1,
                evolution=i,
                pool_size=4 + i,
                mode="generated" if i % 2 == 0 else "gold_chosen",
                hash=f"hash_{i}" if i % 3 == 0 else None,
            )
            for i in range(5)
        ]

        output_path = tmp_path / "roundtrip.jsonl"
        save_pairs(pairs=original_pairs, path=output_path)
        loaded = load_pairs(path=output_path)

        assert len(loaded) == len(original_pairs)
        for orig, load in zip(original_pairs, loaded):
            assert orig.prompt == load.prompt
            assert orig.chosen == load.chosen
            assert orig.rejected == load.rejected
            assert orig.rejected_score == load.rejected_score
            assert orig.chosen_score == load.chosen_score
            assert orig.evolution == load.evolution
            assert orig.pool_size == load.pool_size
            assert orig.mode == load.mode
            assert orig.hash == load.hash

    def test_load_pairs_empty_file(self, tmp_path: Path) -> None:
        """Test loading from an empty file."""
        output_path = tmp_path / "empty.jsonl"
        output_path.write_text("")

        loaded = load_pairs(path=output_path)
        assert loaded == []

    def test_load_pairs_validates_structure(self, tmp_path: Path) -> None:
        """Test that loading validates the pair structure."""
        output_path = tmp_path / "invalid.jsonl"
        # Write incomplete JSON - missing required fields
        output_path.write_text('{"prompt": "test"}\n')

        import pytest

        # Should raise validation error due to missing required fields
        with pytest.raises(Exception):  # noqa: B017
            load_pairs(path=output_path)


class TestToTlrRecords:
    """Tests for to_trl_records function."""

    def test_to_trl_records_basic(self) -> None:
        """Test converting basic preference pairs to TRL format."""
        pairs = [
            PreferencePair(
                prompt="What is the capital of Denmark?",
                chosen="Copenhagen is the capital.",
                rejected="Aarhus is the capital.",
                rejected_score=0.3,
                chosen_score=0.9,
                pool_size=4,
                mode="generated",
            )
        ]

        records = to_trl_records(pairs=pairs)

        assert len(records) == 1
        record = records[0]
        assert record["prompt"] == [{"role": "user", "content": "What is the capital of Denmark?"}]
        assert record["chosen"] == [{"role": "assistant", "content": "Copenhagen is the capital."}]
        assert record["rejected"] == [{"role": "assistant", "content": "Aarhus is the capital."}]

    def test_to_trl_records_multiple_pairs(self) -> None:
        """Test converting multiple pairs to TRL format."""
        pairs = [
            PreferencePair(
                prompt="Prompt 1",
                chosen="Chosen 1",
                rejected="Rejected 1",
                rejected_score=0.4,
                chosen_score=0.8,
                pool_size=3,
                mode="generated",
            ),
            PreferencePair(
                prompt="Prompt 2",
                chosen="Chosen 2",
                rejected="Rejected 2",
                rejected_score=0.3,
                chosen_score=0.9,
                pool_size=5,
                mode="gold_chosen",
            ),
        ]

        records = to_trl_records(pairs=pairs)

        assert len(records) == 2

        assert records[0]["prompt"] == [{"role": "user", "content": "Prompt 1"}]
        assert records[0]["chosen"] == [{"role": "assistant", "content": "Chosen 1"}]
        assert records[0]["rejected"] == [{"role": "assistant", "content": "Rejected 1"}]

        assert records[1]["prompt"] == [{"role": "user", "content": "Prompt 2"}]
        assert records[1]["chosen"] == [{"role": "assistant", "content": "Chosen 2"}]
        assert records[1]["rejected"] == [{"role": "assistant", "content": "Rejected 2"}]

    def test_to_trl_records_empty_list(self) -> None:
        """Test converting empty list to TRL format."""
        records = to_trl_records(pairs=[])
        assert records == []

    def test_to_trl_records_conversation_structure(self) -> None:
        """Test that TRL records have correct conversation structure."""
        pairs = [
            PreferencePair(
                prompt="Multi-turn instruction",
                chosen="Multi-turn response",
                rejected="Bad response",
                rejected_score=0.2,
                pool_size=2,
                mode="generated",
            )
        ]

        records = to_trl_records(pairs=pairs)
        record = records[0]

        # Check structure
        assert isinstance(record["prompt"], list)
        assert isinstance(record["chosen"], list)
        assert isinstance(record["rejected"], list)

        # Check each message has role and content
        assert len(record["prompt"]) == 1
        assert record["prompt"][0]["role"] == "user"
        assert "content" in record["prompt"][0]

        assert len(record["chosen"]) == 1
        assert record["chosen"][0]["role"] == "assistant"
        assert "content" in record["chosen"][0]

        assert len(record["rejected"]) == 1
        assert record["rejected"][0]["role"] == "assistant"
        assert "content" in record["rejected"][0]

    def test_to_trl_records_preserves_content(self) -> None:
        """Test that content is preserved exactly."""
        long_prompt = "This is a very long prompt with special characters: @#$%^&*()_+"
        long_chosen = "This is a very long chosen response\nwith multiple\nlines."
        long_rejected = "This is a bad response."

        pairs = [
            PreferencePair(
                prompt=long_prompt,
                chosen=long_chosen,
                rejected=long_rejected,
                rejected_score=0.3,
                pool_size=3,
                mode="generated",
            )
        ]

        records = to_trl_records(pairs=pairs)
        record = records[0]

        assert record["prompt"][0]["content"] == long_prompt
        assert record["chosen"][0]["content"] == long_chosen
        assert record["rejected"][0]["content"] == long_rejected
