#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 手动指定需要添加的路径（根据实际目录层级调整）
# 假设当前在 HRM-main/notebooks，上级目录是 HRM-main，用 ".." 表示
target_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(target_dir)  # 将 HRM-main 目录添加到Python路径

# 验证是否添加成功（可选）
print("已添加到路径的目录：", target_dir)
print("当前Python路径包含：", target_dir in sys.path)  # 应输出 True

"""
build_math_dataset.py
预处理 MATH 加法数据为 causal-LM 格式：
  inputs = context + "=" + completion
  labels = [-100]*len(context+"=") + completion

保存结构：
  <output_dir>/<set_name>/all__inputs.npy
  <output_dir>/<set_name>/all__labels.npy
  <output_dir>/<set_name>/dataset.json
同时在 output_dir/saved_hf_tokenizer 保存 tokenizer（train 集时）。
"""

import os
import json
from typing import Optional, List

import numpy as np
from tqdm import tqdm
from pydantic import BaseModel
from argdantic import ArgParser

from mytokenizers.hf_math_tokenizer import HFMathTokenizer
from common import PuzzleDatasetMetadata

with open('../data/MATH-401/corpus/random/addition/2_digit_additions.txt', 'r') as f:
    lines = f.readlines()
    print(len(lines))  # 输出行数


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_dir: str = "../data/MATH-401/corpus/random/addition"
    output_dir: str = "../data/MATH-401/exp5"
    max_seq_len: int = 64
    subsample_size: Optional[int] = None
    train_file: str = "1_decimal_additions.txt"
    test_file: Optional[str] = None


def _strip_special_tokens(ids: List[int], bos_id: Optional[int], eos_id: Optional[int]) -> List[int]:
    """去掉序列开头/结尾的 BOS/EOS（如果存在）"""
    if not ids:
        return ids
    start = 0
    end = len(ids)
    if bos_id is not None and len(ids) > 0 and ids[0] == bos_id:
        start = 1
    if eos_id is not None and len(ids) > 0 and ids[-1] == eos_id:
        end = end - 1
    return ids[start:end]


def convert_math_subset(set_name: str, config: DataProcessConfig):
    tokenizer = HFMathTokenizer()

    PAD = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    BOS = getattr(tokenizer, "bos_token_id", None)
    EOS = getattr(tokenizer, "eos_token_id", None)
    IGNORE = -100

    # get "=" token id (if the tokenizer has it)
    eq_ids = tokenizer.encode("=", add_special_tokens=False)
    EQ = eq_ids[0] if len(eq_ids) > 0 else None

    # choose file
    if set_name == "train":
        file_path = os.path.join(config.source_dir, config.train_file)
    else:
        if config.test_file:
            file_path = os.path.join(config.source_dir, config.test_file)
        else:
            file_path = os.path.join(config.source_dir, config.train_file)

    # read lines
    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n\r") for ln in f if ln.strip()]

    # optional split if no explicit test file
    if set_name == "test" and not config.test_file:
        split_idx = int(len(raw_lines) * 0.9)
        lines = raw_lines[split_idx:]
    elif set_name == "train" and not config.test_file:
        split_idx = int(len(raw_lines) * 0.9)
        lines = raw_lines[:split_idx]
    else:
        lines = raw_lines

    if config.subsample_size:
        lines = lines[: config.subsample_size]

    input_seqs = []
    label_seqs = []

    for line in tqdm(lines, desc=f"Processing {set_name}"):
        if "=" in line:
            parts = line.split("=")
            context = parts[0].strip()
            completion = parts[1].strip()
        else:
            # 如果数据行没有 '=', 我们把文件中左侧当 context，右侧当 completion（但通常会有 '='）
            # 这里跳过无法解析的行
            # 也可以选择尝试按空格最后一列为答案，但为安全起见这里只 skip
            # print(f"⚠️ skip line (no '='): {line}")
            continue

        # tokenization: 不添加 special tokens，手动控制 BOS/EOS/SEP
        ctx_ids = tokenizer.encode(context, add_special_tokens=False)
        ans_ids = tokenizer.encode(completion, add_special_tokens=False)

        # strip any BOS/EOS if tokenizer might add them (we used add_special_tokens=False but以防)
        ctx_ids = _strip_special_tokens(ctx_ids, BOS, EOS)
        ans_ids = _strip_special_tokens(ans_ids, BOS, EOS)

        # build sequence: context + [EQ] + completion
        seq = list(ctx_ids)
        if EQ is not None:
            seq.append(EQ)
        else:
            # 如果 tokenizer 没有 "=" 的 token，则通过 encode(" = ") 尝试插入空格+等号
            # 但通常 tokenizer 对 "=" 应该有标记；若仍无则不插入
            pass
        seq += list(ans_ids)

        # labels: prefix masked (-100)，completion kept
        prefix_len = len(ctx_ids) + (1 if EQ is not None else 0)
        labels = [IGNORE] * prefix_len + list(ans_ids)

        # truncate if longer than max_seq_len, prefer keeping completion (truncate context if needed)
        if len(seq) > config.max_seq_len:
            # how many to drop
            overflow = len(seq) - config.max_seq_len
            # drop from the left of context part (not from completion)
            drop_from_context = min(overflow, len(ctx_ids))
            ctx_ids = ctx_ids[drop_from_context:]
            # rebuild seq and labels
            seq = list(ctx_ids)
            if EQ is not None:
                seq.append(EQ)
            seq += list(ans_ids)
            prefix_len = len(ctx_ids) + (1 if EQ is not None else 0)
            labels = [IGNORE] * prefix_len + list(ans_ids)
            # if still overflow (very unlikely), truncate the completion from right
            if len(seq) > config.max_seq_len:
                seq = seq[: config.max_seq_len]
                labels = labels[: config.max_seq_len]

        input_seqs.append(seq)
        label_seqs.append(labels)

    if len(input_seqs) == 0:
        raise RuntimeError(f"No valid lines found for set {set_name} in {file_path}")

    # pad to uniform length (use config.max_seq_len as final length)
    L = config.max_seq_len
    N = len(input_seqs)
    inputs_arr = np.full((N, L), PAD, dtype=np.int64)
    labels_arr = np.full((N, L), IGNORE, dtype=np.int64)

    for i, (seq, lab) in enumerate(zip(input_seqs, label_seqs)):
        length = min(len(seq), L)
        inputs_arr[i, :length] = np.array(seq[:length], dtype=np.int64)
        labels_arr[i, :length] = np.array(lab[:length], dtype=np.int64)

    # prepare results
    results = {
        "inputs": inputs_arr,
        "labels": labels_arr,
        "group_indices": np.arange(len(inputs_arr), dtype=np.int32),
        "puzzle_indices": np.arange(len(inputs_arr), dtype=np.int32),
        "puzzle_identifiers": np.zeros(len(inputs_arr), dtype=np.int32),
    }

    print(f"Saved shapes for {set_name}: inputs {inputs_arr.shape}, labels {labels_arr.shape}")

    # metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=inputs_arr.shape[1],
        vocab_size=tokenizer.vocab_size,
        pad_id=int(PAD),
        ignore_label_id=int(IGNORE),
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(inputs_arr),
        mean_puzzle_examples=1,
        sets=["all"],
    )

    # save
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    # write dataset.json
    with open(os.path.join(save_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, ensure_ascii=False, indent=2)

    # save arrays
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # save tokenizer for train set
    if set_name == "train":
        tokenizer_save_path = os.path.join(config.output_dir, "saved_hf_tokenizer")
        os.makedirs(tokenizer_save_path, exist_ok=True)
        try:
            tokenizer.save_pretrained(tokenizer_save_path)
            print(f"✅ Saved tokenizer to {tokenizer_save_path}")
        except Exception:
            # 如果 tokenizer 没有 save_pretrained 方法则跳过
            pass

    print(f"✅ Saved preprocessed {set_name} to {save_dir}")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_math_subset("train", config)
    convert_math_subset("test", config)


if __name__ == "__main__":
    cli()
