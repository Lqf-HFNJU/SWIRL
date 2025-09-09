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
from verl.tools.utils.tool_registry import initialize_tools_from_config
from sglang.srt.openai_api.protocol import Tool

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser


def initialize_tools(tools_config_file):
    if tools_config_file is None:
        return [], {}, None, [], None

    tool_list = initialize_tools_from_config(tools_config_file)

    tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
    tool_call_parser_type = 'qwen25'
    sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
    function_call_parser = FunctionCallParser(
        sgl_tools,
        tool_call_parser_type,
    )
    return function_call_parser


def get_instruction_from_tool_call(response: str, function_call_parser: FunctionCallParser) -> str:
    try:
        normed_content, tool_calls = function_call_parser.parse_non_stream(response)
        instruction = eval(tool_calls[0].parameters)['instruction']
        return instruction
    except Exception as e:
        print(f"Error parsing tool call: {e}")
        return ""


def convert_dataset_to_executor_format(dataset: pd.DataFrame) -> Dataset:
    def convert_func(example):
        instruction = example['instruction']
        answer = example['extra_info']['answer']
        prompt = example['extra_info']['tools_kwargs']['executor_agent']['create_kwargs']['executor_prompt']

        assert len(example['images']) == 1
        img = example['images'][-1]

        prompt = [
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": prompt.format(instruction=instruction)}
        ]

        oldbbox = example['extra_info']['bbox']
        if len(oldbbox) == 0:
            newbbox = []
        else:
            assert type(oldbbox[0]) is np.ndarray and len(oldbbox) == 2, oldbbox
            newbbox = [oldbbox[0].tolist(), oldbbox[1].tolist()]

        return {
            "data_source": "executor",
            "prompt": prompt,
            "images": [img],
            "ability": "gui_grounding",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "index": example['extra_info']['index'],
                "answer": answer,
                "high_level_instruction": example['extra_info']["high_level_instruction"],
                "low_level_instruction": example['extra_info']["low_level_instruction"],
                "planner_instruction": instruction,
                "width": example['extra_info']['width'],
                "height": example['extra_info']['height'],
                "bbox": newbbox,
            },
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
        'images': Sequence(feature=datasets.Image(mode=None, decode=True, id=None), length=-1, id=None),
        'ability': Value(dtype='string', id=None),
        'reward_model': {'ground_truth': Value(dtype='string', id=None), 'style': Value(dtype='string', id=None)},
        'extra_info': {
            'index': Value(dtype='string', id=None),
            'answer': Value(dtype='string', id=None),
            "high_level_instruction": Value(dtype='string', id=None),
            "low_level_instruction": Value(dtype='string', id=None),
            "planner_instruction": Value(dtype='string', id=None),
            "width": Value("int32", id=None),
            "height": Value("int32", id=None),
            "bbox": [[Value("int32", id=None)]],
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


def planner2executor(filepath: str, savepath: str, response_key: str, tools_config_file: str = None):
    function_call_parser = initialize_tools(tools_config_file)
    dataset = pl.read_parquet(filepath).to_pandas()
    if not type(dataset[response_key][0])  is str:
        dataset[response_key] = dataset[response_key].apply(lambda x: x[0])
        
    dataset["instruction"] = dataset[response_key].apply(lambda resp: get_instruction_from_tool_call(resp, function_call_parser))

    dataset = convert_dataset_to_executor_format(dataset)
    dataset.to_parquet(savepath)

def get_args():
    parser = argparse.ArgumentParser(description="Convert planner dataset to executor dataset.")
    parser.add_argument("--filepath", type=str)
    parser.add_argument("--savepath", type=str)
    parser.add_argument("--response_key", type=str, default="responses", help="Key in the dataset containing the response.")
    parser.add_argument("--tools_config_file", type=str)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    planner2executor(
        filepath=args.filepath,
        savepath=args.savepath,
        response_key=args.response_key,
        tools_config_file=args.tools_config_file
    )
    print(f"Converted dataset saved to {args.savepath}")
