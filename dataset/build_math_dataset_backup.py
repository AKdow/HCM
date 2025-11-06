# from mytokenizers.math_tokenizer import MathTokenizer

# tokenizer = MathTokenizer.from_file("mytokenizers/math_tokenizer_vocab.json")
# print("✅ 词表大小:", tokenizer.vocab_size)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel
from argdantic import ArgParser
from typing import Optional

from mytokenizers.hf_math_tokenizer import HFMathTokenizer
from common import PuzzleDatasetMetadata 

cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_dir: str = "../data/MATH-401/corpus/random/addition"  # ✅ 修改：指向实际数据路径
    output_dir: str = "../data/MATH-401/exp2"  # 预处理后数据保存路径

    max_seq_len: int = 64  
    subsample_size: Optional[int] = None
    
    # ✅ 新增：指定具体的数据文件
    train_file: str = "1_decimal_additions.txt"  
    test_file: Optional[str] = None  # 如果没有单独的测试集，可以从训练集切分


def pad_sequences(seqs, pad_id=0):
    """统一序列长度"""
    max_len = max(len(s) for s in seqs)
    padded = np.array([s + [pad_id] * (max_len - len(s)) for s in seqs], dtype=np.int32)
    return padded


def convert_math_subset(set_name: str, config: DataProcessConfig):
    tokenizer = HFMathTokenizer()
    
    input_list, label_list = [], []

    # ✅ 修改：根据 set_name 选择文件
    if set_name == "train":
        file_path = os.path.join(config.source_dir, config.train_file)
    else:  # test
        if config.test_file:
            file_path = os.path.join(config.source_dir, config.test_file)
        else:
            # 如果没有测试集文件，从训练集切分
            file_path = os.path.join(config.source_dir, config.train_file)

    # ✅ 修改：读取纯文本格式
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # ✅ 如果没有单独的测试集，按比例切分
    if set_name == "test" and not config.test_file:
        split_idx = int(len(lines) * 0.9)  # 90% 训练，10% 测试
        lines = lines[split_idx:]
    elif set_name == "train" and not config.test_file:
        split_idx = int(len(lines) * 0.9)
        lines = lines[:split_idx]

    if config.subsample_size:
        lines = lines[:config.subsample_size]

    for line in tqdm(lines, desc=f"Processing {set_name}"):
        # ✅ 修改：解析算式格式 "84.5 + 54.5 = 139"
        if "=" in line:
            parts = line.split("=")
            context = parts[0].strip()  # "84.5 + 54.5"
            completion = parts[1].strip()  # "139"
        else:
            print(f"⚠️ 跳过格式错误的行: {line}")
            continue

        # 检查输出
        # print("context:", context)
        # print("completion:", completion)

        inp_ids = tokenizer.encode(context, add_special_tokens=True)
        out_ids = tokenizer.encode(completion, add_special_tokens=True)

        # ✅ 打印分词后的结果（可选，调试用）
        # print("输入原文:", context)
        # print("输入分词ID:", inp_ids)
        # print("输出原文:", completion)
        # print("输出分词ID:", out_ids)
        # print("-----")

        # 截断
        inp_ids = inp_ids[:config.max_seq_len]
        out_ids = out_ids[:config.max_seq_len]

        input_list.append(inp_ids)
        label_list.append(out_ids)

    # Padding
    inputs = pad_sequences(input_list, pad_id=tokenizer.pad_token_id)
    labels = pad_sequences(label_list, pad_id=tokenizer.pad_token_id)

    # 统一到相同长度
    if inputs.shape[1] != labels.shape[1]:
        max_len = max(inputs.shape[1], labels.shape[1])
        inputs = np.pad(inputs, ((0, 0), (0, max_len - inputs.shape[1])), constant_values=tokenizer.pad_token_id)
        labels = np.pad(labels, ((0, 0), (0, max_len - labels.shape[1])), constant_values=tokenizer.pad_token_id)

    # 组织成 HRM 通用结构
    results = {
        "inputs": inputs,
        "labels": labels,
        "group_indices": np.arange(len(inputs) + 1, dtype=np.int32),
        "puzzle_indices": np.arange(len(inputs) + 1, dtype=np.int32),
        "puzzle_identifiers": np.zeros(len(inputs), dtype=np.int32),
    }

    print(results["inputs"].shape, results["labels"].shape)

    # Dataset metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=inputs.shape[1],
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_token_id,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(inputs),
        mean_puzzle_examples=1,
        sets=["all"]
    )

    # 保存
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, ensure_ascii=False, indent=2)

    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # 保存 tokenizer（只保存一次）
    if set_name == "train":
        tokenizer_save_path = os.path.join(config.output_dir, "saved_hf_tokenizer")
        os.makedirs(tokenizer_save_path, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_save_path)
        print(f"✅ Saved tokenizer to {tokenizer_save_path}")

    print(f"✅ Saved preprocessed {set_name} to {save_dir}")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_math_subset("train", config)
    convert_math_subset("test", config)


if __name__ == "__main__":
    cli()
