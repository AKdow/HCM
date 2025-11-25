#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import numpy as np
from pathlib import Path

# 导入你修改后的模块（确保在 PYTHONPATH 中）
from build_math_dataset import DataProcessConfig, convert_math_subset
from mytokenizers.hf_math_tokenizer import HFMathTokenizer

def prepare_sample(source_dir: str):
    os.makedirs(source_dir, exist_ok=True)
    # 写入少量样本（注意保留等号）
    sample_lines = [
        "27 + 64 = 91",
        "10 + 10 = 20",
        "84.5 + 54.5 = 139.0",
        "3 + 4 = 7",
        "99 + 1 = 100"
    ]
    file_path = os.path.join(source_dir, "1_decimal_additions.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        for ln in sample_lines:
            f.write(ln + "\n")
    print("Wrote sample file:", file_path)
    return file_path

def run_test(tmp_root="./tmp_math_test"):
    src_corpus = os.path.join(tmp_root, "corpus", "random", "addition")
    out_dir = os.path.join(tmp_root, "exp_out")
    # cleanup
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    os.makedirs(src_corpus, exist_ok=True)

    prepare_sample(src_corpus)

    cfg = DataProcessConfig(
        source_dir=os.path.abspath(src_corpus),
        output_dir=os.path.abspath(out_dir),
        max_seq_len=32,
        subsample_size=None,
        train_file="1_decimal_additions.txt",
        test_file=None,
    )

    # run preprocess (train + test)
    convert_math_subset("train", cfg)
    convert_math_subset("test", cfg)

    # load and inspect
    tokenizer = HFMathTokenizer()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    arr_in = np.load(os.path.join(out_dir, "train", "all__inputs.npy"))
    arr_lab = np.load(os.path.join(out_dir, "train", "all__labels.npy"))
    print("Saved shapes:", arr_in.shape, arr_lab.shape)

    for i in range(min(5, arr_in.shape[0])):
        toks = [int(x) for x in arr_in[i] if int(x) != pad_id]
        decoded = tokenizer.decode(toks)
        supervised_idx = np.where(arr_lab[i] != -100)[0]
        targets = [int(arr_lab[i, p]) for p in supervised_idx] if supervised_idx.size > 0 else []
        decoded_targets = tokenizer.decode(targets) if targets else ""
        print(f"Sample {i}:")
        print("  decoded input:", decoded)
        print("  supervised positions (first 10):", supervised_idx[:10])
        print("  decoded targets:", decoded_targets)
        print("-" * 40)

if __name__ == "__main__":
    run_test()
