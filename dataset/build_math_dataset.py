# preprocess_math_ops_A.py
import os, json, random, numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from pydantic import BaseModel
from argdantic import ArgParser
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
    ids = [tok._convert_token_to_id(t) for t in tok._tokenize(text)]
    return tok.build_inputs_with_special_tokens(ids)

def build_samples(tok: HFMathTokenizer, lines: List[str]) -> List[Tuple[List[int], List[int]]]:
    samples = []
    for ln in lines:
        lhs, rhs = parse_eq(ln)
        inp = tokenize_ids(tok, lhs + " =")
        lab = tokenize_ids(tok, rhs)
        samples.append((inp, lab))
    return samples

def pack_and_save(out_dir, split, pairs, vocab_size):
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    seq_len = max(max(len(i), len(l)) for i,l in pairs)
    pad_id = 0
    pad = lambda x: x + [pad_id]*(seq_len - len(x))
    inputs = [pad(i) for i,l in pairs]
    labels = [pad(l) for i,l in pairs]
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
    pack_and_save(config.output_dir,"train",pairs[n_test:],tok.vocab_size)
    pack_and_save(config.output_dir,"test",pairs[:n_test],tok.vocab_size)
    json.dump(["<blank>"], open(os.path.join(config.output_dir,"identifiers.json"),"w"))

if __name__=="__main__":
    cli()
