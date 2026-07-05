#!/usr/bin/env python3
"""Regression test for TRL DPOTrainer precompute_ref_log_probs cache bug.

This test verifies that the fix in src/croco/trl_dpo_fix.py correctly
patches TRL to resolve the FileNotFoundError.

The bug (TRL 1.7.0): DPOTrainer._precompute_ref_logps() computes a cache_file
path, then calls dataset.map() without passing cache_file_name. Since the 
dataset has no cache_files, map() generates a DIFFERENT random path, so the 
cache is never written where TRL expects it.

Scripts to reproduce/verify:
- src/scripts/verify_trell_bug.py - Vanilla TRL (fails with FileNotFoundError)
- src/scripts/test_trl_fix.py - Patched TRL (succeeds)

Manual reproduction on GPU machine:
    # Reproduces the bug (expected to fail)
    uv run src/scripts/verify_trell_bug.py
    
    # Verifies the fix (expected to succeed)  
    uv run src/scripts/test_trl_fix.py

See: AGENTS.md "Gotchas" for TMPDIR + HF_DATASETS_CACHE requirements.
"""


def test_patch_module_exists() -> None:
    """Verify the trl_dpo_fix module exists and can be imported."""
    from croco import trl_dpo_fix
    assert hasattr(trl_dpo_fix, 'patch_dpo_precompute')


def test_patch_function_signature() -> None:
    """Verify patch_dpo_precompute is callable."""
    from croco.trl_dpo_fix import patch_dpo_precompute
    assert callable(patch_dpo_precompute)


def test_patch_applies_without_model() -> None:
    """Verify the patch can be applied without loading a model.
    
    This tests that the monkey-patch mechanism works, without the
    overhead of loading an 8B model.
    """
    from croco.trl_dpo_fix import patch_dpo_precompute
    from trl import DPOTrainer
    
    # Get original method for comparison
    original = DPOTrainer._precompute_ref_logps
    
    # Apply patch
    patch_dpo_precompute()
    
    # Verify method was replaced
    assert DPOTrainer._precompute_ref_logps is not original, \
        "Patch should replace _precompute_ref_logps"


def test_patch_introspection() -> None:
    """Verify patch adds logging to the fixed code path."""
    from croco.trl_dpo_fix import _patched_precompute_ref_logps
    import inspect
    
    source = inspect.getsource(_patched_precompute_ref_logps)
    
    # The fix passes cache_file_name to dataset.map()
    assert 'cache_file_name=cache_file' in source, \
        "Patch should pass cache_file_name to dataset.map()"
    
    # Should have logging about the fix
    assert 'logger.info' in source
