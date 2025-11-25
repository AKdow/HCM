# quick_check_dataset.py
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
import torch

cfg = PuzzleDatasetConfig(
    seed=0,
    dataset_path="./data/MATH-401/exp6",  # 改成你 build_math_dataset.py 的 output_dir
    global_batch_size=8,
    test_set_mode=True,
    epochs_per_iter=1,
    rank=0,
    num_replicas=1
)

ds = PuzzleDataset(cfg, split="train")
it = iter(ds)
set_name, batch, gb = next(it)
print("set:", set_name)
print("inputs shape/dtype:", batch["inputs"].shape, batch["inputs"].dtype)
print("labels shape/dtype:", batch["labels"].shape, batch["labels"].dtype)
print("puzzle_identifiers shape/dtype:", batch["puzzle_identifiers"].shape, batch["puzzle_identifiers"].dtype)
# print first example
print("first labels:", batch["labels"][:4])
