import os
import json
import argparse
import pandas as pd
from datasets import Dataset


def textworld_make_map_fn(split, game_dir, dataset_id, reward_method):
    """
    Add a row to each data item that represents a unique id
    """
    def process_fn(example, idx):
        data = {
            "data_source": f"textworld_{dataset_id}",
            "prompt": [
                {
                    "role": "user",
                    "content": example["prompt"],
                }
            ],
            "reward_model": {
                "style": "rule", 
                "ground_truth": f"{example['game_size']}_{example['game_id']}"
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "response": example["response"],
                "prompt": example["prompt"],
                "game_path": game_dir,
                "chunk_len": example["chunk_len"] if "chunk_len" in example.keys() else -1,
                "chunk_id": example["chunk_id"] if "chunk_id" in example.keys() else -1,
                "reward_method": reward_method
            },
        }
        return data

    return process_fn


def main(args):
    
    with open(os.path.join(args.data_dir, "train.jsonl"), 'r') as f:
        train_dataset = []
        for line in f:
            train_dataset.append(json.loads(line))
    with open(os.path.join(args.data_dir, "validation.jsonl"), 'r') as f:
        val_dataset = []
        for line in f:
            val_dataset.append(json.loads(line))
    with open(os.path.join(args.data_dir, "test.jsonl"), 'r') as f:
        test_dataset = []
        for line in f:
            test_dataset.append(json.loads(line))

    train_dataset = Dataset.from_pandas(pd.DataFrame(train_dataset))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_dataset))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_dataset))

    if args.task == "textworld":
        train_dataset = train_dataset.map(function=textworld_make_map_fn("train", args.game_dir, args.dataset_id, args.reward_method), with_indices=True)
        val_dataset = val_dataset.map(function=textworld_make_map_fn("validation", args.game_dir, args.dataset_id, args.reward_method), with_indices=True)
        test_dataset = test_dataset.map(function=textworld_make_map_fn("test", args.game_dir, args.dataset_id, args.reward_method), with_indices=True)
    else:
        raise NotImplementedError

    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(args.local_dir, "validation.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_dir, "test.parquet"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PPO data for verl post-training.")
    parser.add_argument("--task", type=str, required=True, choices=["textworld"], help="Choose the task.")
    parser.add_argument("--dataset_id", type=str, required=True, help="The specific dataset of the task.")
    parser.add_argument("--game_dir", type=str, required=True, help="Path to the games directory.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory containing train.jsonl, validation.jsonl, and test.jsonl.")
    parser.add_argument("--local_dir", type=str, required=True, help="Path to the local directory to save the data parquet.")
    parser.add_argument("--reward_method", type=str, required=True, choices=["game_winning", "exact_match", "game_winning_dense"], help="0/1 reward computation method. ")
    
    args = parser.parse_args()
    main(args)