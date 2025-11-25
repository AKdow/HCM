import os
import json

import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata


def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):
    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_path: str
    global_batch_size: int
    test_set_mode: bool

    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.

    rank: int
    num_replicas: int


class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split
        self.metadata = self._load_metadata()
        
        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self) -> PuzzleDatasetMetadata:
        with open(os.path.join(self.config.dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        # For regression dataset we want:
        # - inputs: memory-map OK (int ids)
        # - labels: load as numpy float32 (no mmap) shape (N,1) or (N,)
        # - indices: in-memory
        field_mmap_modes = {
            "inputs": "r",
            # labels: regression -> load fully as float32 (no mmap)
            "labels": None,
            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data
        self._data = {}
        for set_name in self.metadata.sets:
            subset = {}
            for field_name, mmap_mode in field_mmap_modes.items():
                path = os.path.join(self.config.dataset_path, self.split, f"{set_name}__{field_name}.npy")
                arr = np.load(path, mmap_mode=mmap_mode)
                # If labels, ensure float32 and shape (N,1)
                if field_name == "labels":
                    arr = arr.astype(np.float32, copy=False)
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                subset[field_name] = arr
            self._data[set_name] = subset

    def _collate_batch(self, batch):
        """
        Expect batch keys: inputs(int), labels(float), puzzle_identifiers(int)
        """

        # ---- Inputs 和 puzzle_identifiers 强制 int32 ----
        inputs = batch["inputs"].astype(np.int32)
        puzzle_identifiers = batch["puzzle_identifiers"].astype(np.int32)

        # ---- Labels: 保证 float32，不做 int32 cast ----
        labels_np = batch["labels"]

        # 如果是 (N,1) → squeeze 到 (N,)
        if labels_np.ndim == 2 and labels_np.shape[1] == 1:
            labels_np = labels_np.reshape(-1)

        # labels 保持 float32
        labels = labels_np.astype(np.float32)

        # ---- Pad to local_batch_size ----
        current = labels.shape[0]
        if current < self.local_batch_size:
            pad_size = self.local_batch_size - current

            inputs = np.pad(inputs, ((0, pad_size), (0, 0)), constant_values=self.metadata.pad_id)
            labels = np.pad(labels, (0, pad_size), constant_values=0.0)
            puzzle_identifiers = np.pad(
                puzzle_identifiers,
                (0, pad_size),
                constant_values=self.metadata.blank_identifier_id
            )

        # ---- Convert to tensors ----
        return {
            "inputs": torch.from_numpy(inputs),
            "labels": torch.from_numpy(labels),  # float32 tensor
            "puzzle_identifiers": torch.from_numpy(puzzle_identifiers),
        }

    
    def _iter_test(self):
        for set_name, dataset in self._data.items():  # type: ignore
            total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start: local_end],
                    "labels": dataset["labels"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                })

                yield set_name, batch, end_index - start_index
                
                # Advance to next batch
                start_index += self.config.global_batch_size

    def _iter_train(self):
        for set_name, dataset in self._data.items():  # type: ignore
            # Increase epoch count
            self._iters += 1

            # Randomly shuffle groups
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            group_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1) for _i in range(self.config.epochs_per_iter)])
            start_index = 0
            
            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )

                # Select current rank and collate
                global_effective_batch_size = batch_puzzle_indices.size  # Global effective batch size, excluding pads

                # Drop last batch
                if global_effective_batch_size < self.config.global_batch_size:
                    break

                batch_indices        = batch_indices       [self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_puzzle_indices = batch_puzzle_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]
                })

                yield set_name, batch, global_effective_batch_size
                
    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."
        
        self._lazy_load_dataset()
        
        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()
