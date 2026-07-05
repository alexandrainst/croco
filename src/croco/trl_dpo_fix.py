"""Patch for TRL's DPOTrainer._precompute_ref_logps() cache bug.

The bug: When precompute_ref_log_probs=True, TRL computes a cache_file path,
then calls dataset.map() without passing cache_file_name. Since the dataset
has no cache_files, map() generates a DIFFERENT random path, so the cache
is never written where TRL expects it.

This patch monkey-patches DPOTrainer._precompute_ref_logps() to pass
cache_file_name to dataset.map().

Usage (at module import, before creating DPOTrainer):
    from croco.trl_dpo_fix import patch_dpo_precompute
    patch_dpo_precompute()
"""

import logging
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from trl import DPOTrainer

logger = logging.getLogger(__name__)


def _patched_precompute_ref_logps(
    self: DPOTrainer,
    dataset: Dataset,
    name: str,
    batch_size: int,
) -> Dataset:
    """Patched version of DPOTrainer._precompute_ref_logps().

    Fix: Pass cache_file_name to dataset.map() so cache is written correctly.
    """
    from datasets.fingerprint import Hasher
    from trl.trainer.utils import hash_module

    model_hash = hash_module(self.ref_model or self.model)
    fingerprint = Hasher.hash((dataset._fingerprint, model_hash))
    cache_file = dataset._get_cache_file_path(fingerprint)

    logger.info(f"Precomputing ref log probs for {name} dataset, cache_file={cache_file}")

    if os.path.exists(cache_file):
        logger.info(f"Cache already exists, loading from {cache_file}")
        return concatenate_datasets([dataset, Dataset.from_file(cache_file)], axis=1)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=self.data_collator,
        num_workers=self.args.dataloader_num_workers,
        pin_memory=self.args.dataloader_pin_memory,
        shuffle=False,
    )
    data_loader = self.accelerator.prepare(dataloader)
    ref_chosen_logps = []
    ref_rejected_logps = []
    for padded_batch in tqdm(
        iterable=data_loader, desc=f"Computing reference log probs for {name} dataset"
    ):
        ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(padded_batch)
        ref_chosen_logp, ref_rejected_logp = self.accelerator.gather_for_metrics(
            (ref_chosen_logp, ref_rejected_logp)
        )
        ref_chosen_logps.append(ref_chosen_logp.cpu())
        ref_rejected_logps.append(ref_rejected_logp.cpu())

    ref_chosen_logps = torch.cat(ref_chosen_logps)
    ref_rejected_logps = torch.cat(ref_rejected_logps)

    if self.accelerator.is_main_process:

        def add_ref_logps(batch, indices):
            return {
                "ref_chosen_logps": ref_chosen_logps[indices],
                "ref_rejected_logps": ref_rejected_logps[indices],
            }

        # FIX: Pass cache_file_name so cache is written to the expected location
        logger.info(f"Writing cache to {cache_file}")
        dataset.map(
            add_ref_logps,
            with_indices=True,
            batched=True,
            remove_columns=dataset.column_names,
            new_fingerprint=fingerprint,
            cache_file_name=cache_file,  # <-- THE FIX
            desc=f"Caching reference log probs for {name} dataset",
        )
        logger.info(f"Cache written successfully")
    self.accelerator.wait_for_everyone()

    return concatenate_datasets([dataset, Dataset.from_file(cache_file)], axis=1)


def patch_dpo_precompute() -> None:
    """Apply the monkey-patch to DPOTrainer._precompute_ref_logps().

    Call this before creating any DPOTrainer instances.
    """
    import os

    # Import here to avoid circular dependency
    from datasets.fingerprint import Hasher
    from trl.trainer.utils import hash_module

    # Add missing imports to the function
    global _patched_precompute_ref_logps
    original_func = _patched_precompute_ref_logps

    # Create wrapper with proper imports
    def patched(self, dataset, name, batch_size):
        import os
        import torch
        from datasets import Dataset, concatenate_datasets
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from datasets.fingerprint import Hasher
        from trl.trainer.utils import hash_module

        model_hash = hash_module(self.ref_model or self.model)
        fingerprint = Hasher.hash((dataset._fingerprint, model_hash))
        cache_file = dataset._get_cache_file_path(fingerprint)

        logger.info(f"Precomputing ref log probs for {name} dataset, cache_file={cache_file}")

        if os.path.exists(cache_file):
            logger.info(f"Cache already exists, loading from {cache_file}")
            return concatenate_datasets([dataset, Dataset.from_file(cache_file)], axis=1)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False,
        )
        data_loader = self.accelerator.prepare(dataloader)
        ref_chosen_logps = []
        ref_rejected_logps = []
        for padded_batch in tqdm(
            iterable=data_loader, desc=f"Computing reference log probs for {name} dataset"
        ):
            ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(padded_batch)
            ref_chosen_logp, ref_rejected_logp = self.accelerator.gather_for_metrics(
                (ref_chosen_logp, ref_rejected_logp)
            )
            ref_chosen_logps.append(ref_chosen_logp.cpu())
            ref_rejected_logps.append(ref_rejected_logp.cpu())

        ref_chosen_logps = torch.cat(ref_chosen_logps)
        ref_rejected_logps = torch.cat(ref_rejected_logps)

        if self.accelerator.is_main_process:

            def add_ref_logps(batch, indices):
                return {
                    "ref_chosen_logps": ref_chosen_logps[indices],
                    "ref_rejected_logps": ref_rejected_logps[indices],
                }

            logger.info(f"Writing cache to {cache_file}")
            dataset.map(
                add_ref_logps,
                with_indices=True,
                batched=True,
                remove_columns=dataset.column_names,
                new_fingerprint=fingerprint,
                cache_file_name=cache_file,  # THE FIX
                desc=f"Caching reference log probs for {name} dataset",
            )
            logger.info(f"Cache written successfully")
        self.accelerator.wait_for_everyone()

        return concatenate_datasets([dataset, Dataset.from_file(cache_file)], axis=1)

    DPOTrainer._precompute_ref_logps = patched
    logger.info("Patched DPOTrainer._precompute_ref_logps() to fix cache bug")
