import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import polars as pl
import argparse

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from concurrent.futures import ThreadPoolExecutor, as_completed
import datasets
from datasets import Dataset, concatenate_datasets, Features, Value, Sequence
import pandas as pd



def convert_dataset_to_executor_format(dataset: pd.DataFrame) -> Dataset:
    def convert_func(example):
        instruction = example['instruction']
        
        student_prompt = example['extra_info']['student_prompt']
        student_prompt[-1]['content'] = f'<teacher_response>{instruction}</teacher_response>'
            
        return {
                "data_source": "student",
                "prompt": student_prompt,
                "ability": "math",
                "reward_model": example['reward_model'],
                "extra_info": example['extra_info'],
            }
            

    dataset = dataset.to_dict(orient="records")
    grounding_dataset = []
    with ThreadPoolExecutor(max_workers=112) as executor:
        futures = [executor.submit(convert_func, it) for it in dataset]

        for future in as_completed(futures):
            try:
                d = future.result()
                if d is not None:
                    grounding_dataset.append(d)
            except Exception as e:
                print("task failed:", e)

    print(f"Processed {len(grounding_dataset)} items successfully, out of {len(dataset)} click items.")

    features = Features({
        'data_source': Value(dtype='string', id=None),
        'prompt': [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None)}],
        'ability': Value(dtype='string', id=None),
        'reward_model': {'ground_truth': Value(dtype='string', id=None), 'style': Value(dtype='string', id=None)},
        'extra_info': {
            'split': Value(dtype='string', id=None),
            'index': Value(dtype='string', id=None),
            "question": Value(dtype='string', id=None),
            "student_prompt": [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None)}],
        }})

    def build_chunk(start_idx):
        end_idx = min(start_idx + CHUNK_SIZE, len(grounding_dataset))
        batch = grounding_dataset[start_idx:end_idx]
        ds = Dataset.from_list(batch, features=features)
        print(f"  ▶ chunk {start_idx}–{end_idx} built in PID {os.getpid()}")
        return ds

    CHUNK_SIZE = 100
    starts = list(range(0, len(grounding_dataset), CHUNK_SIZE))

    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 32)) as executor:
        futures = [executor.submit(build_chunk, s) for s in starts]
        ds_chunks = [f.result() for f in as_completed(futures)]

    grounding_dataset = concatenate_datasets(ds_chunks)
    print("✅ all chunks concatenated →", len(grounding_dataset), "samples")

    return grounding_dataset


def planner2executor(filepath: str, savepath: str, response_key: str):
    dataset = pl.read_parquet(filepath).to_pandas()
    if not type(dataset[response_key][0])  is str:
        dataset["instruction"] = dataset[response_key].apply(lambda x: x[0])
    else:
        dataset["instruction"] = dataset[response_key]
        
    dataset = convert_dataset_to_executor_format(dataset)
    dataset.to_parquet(savepath)

def get_args():
    parser = argparse.ArgumentParser(description="Convert planner dataset to executor dataset.")
    parser.add_argument("--filepath", type=str)
    parser.add_argument("--savepath", type=str)
    parser.add_argument("--response_key", type=str, default="responses", help="Key in the dataset containing the response.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    planner2executor(
        filepath=args.filepath,
        savepath=args.savepath,
        response_key=args.response_key,
    )
    print(f"Converted dataset saved to {args.savepath}")
