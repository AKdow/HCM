# preprocess_math_ops_A.py
import os, json, random, numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from pydantic import BaseModel
from argdantic import ArgParser
from typing import List, cast

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mytokenizers.hf_math_tokenizer import HFMathTokenizer

@dataclass
class PuzzleDatasetMetadata:
    seq_len: int
    vocab_size: int
    pad_id: int
    ignore_label_id: int
    blank_identifier_id: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    sets: List[str]

cli = ArgParser()

class DataProcessConfig(BaseModel):
    input_txt: str = "../data/MATH-401/corpus/random/addition/1_digit_additions.txt"
    output_dir: str = "../data/MATH-401/exp3"
    test_ratio: float = 0.1
    seed: int = 42

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def parse_eq(line: str) -> Tuple[str, str]:
    left, right = line.split("=")
    return left.strip(), right.strip()

def tokenize_ids(tok: HFMathTokenizer, text: str) -> List[int]:
    # 先最长匹配分词，再转 id
    ids = [tok._convert_token_to_id(t) for t in tok._tokenize(text)]
    # 手动拼接 BOS/EOS
    bos = tok.bos_token_id
    eos = tok.eos_token_id
    if bos is not None:
        ids = [bos] + ids
    if eos is not None:
        ids = ids + [eos]
    # 使用 cast 强制转换为 List[int]
    return cast(List[int], ids)

def build_samples(tok: HFMathTokenizer, lines: List[str]) -> List[Tuple[List[int], List[int]]]:
    samples = []
    for ln in lines:
        lhs, rhs = parse_eq(ln)
        inp = tokenize_ids(tok, lhs + " =")
        lab = tokenize_ids(tok, rhs)
        samples.append((inp, lab))
        
        # Debugging: Print input and tokenized output
        print(f"Original Input: {ln}")
        print(f"Tokenized Input IDs: {inp}")
        print(f"Tokenized Label IDs: {lab}")
        print(f"Input Length: {len(inp)}, Label Length: {len(lab)}")
        
    return samples

def pack_and_save(out_dir, split, pairs, vocab_size):
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    seq_len = max(max(len(i), len(l)) for i,l in pairs)
    pad_id = 0
    pad = lambda x: x + [pad_id]*(seq_len - len(x))
    inputs = [pad(i) for i,l in pairs]
    labels = [pad(l) for i,l in pairs]
    
    # Debugging: Print shapes/dimensions of the arrays
    print(f"Inputs shape: {np.array(inputs).shape}")
    print(f"Labels shape: {np.array(labels).shape}")

    np.save(os.path.join(out_dir, split, "all__inputs.npy"), np.array(inputs))
    np.save(os.path.join(out_dir, split, "all__labels.npy"), np.array(labels))
    np.save(os.path.join(out_dir, split, "all__puzzle_indices.npy"), np.array([0,len(pairs)]))
    np.save(os.path.join(out_dir, split, "all__group_indices.npy"), np.array([0,len(pairs)]))
    np.save(os.path.join(out_dir, split, "all__puzzle_identifiers.npy"), np.zeros(len(pairs),dtype=np.int32))
    
    meta = PuzzleDatasetMetadata(seq_len, vocab_size,0,0,0,1,1,len(pairs),["all"])
    json.dump(meta.__dict__, open(os.path.join(out_dir,split,"dataset.json"),"w"),indent=2)

@cli.command(singleton=True)
def preprocess(config: DataProcessConfig):
    random.seed(config.seed)
    lines = read_lines(config.input_txt)
    tok = HFMathTokenizer()
    pairs = build_samples(tok, lines)
    random.shuffle(pairs)
    n_test = int(len(pairs)*config.test_ratio)
    
    # Save train/test splits
    pack_and_save(config.output_dir, "train", pairs[n_test:], tok.vocab_size)
    pack_and_save(config.output_dir, "test", pairs[:n_test], tok.vocab_size)
    
    # Save identifier (for visualization only)
    json.dump(["<blank>"], open(os.path.join(config.output_dir, "identifiers.json"), "w"))

if __name__=="__main__":
    cli()
