import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import hydra
import numpy as np
import pickle
import ray
from tqdm import tqdm

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed
import datasets
from datasets import Dataset, concatenate_datasets, Features, Value, Sequence
import pandas as pd
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.trainer.main_ppo import create_rl_dataset
from verl.utils.dataset.rl_dataset import collate_fn
from omegaconf import OmegaConf
from verl.tools.utils.tool_registry import initialize_tools_from_config
from sglang.srt.openai_api.protocol import Tool

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser


def get_tool_call_parser_type(processing_class) -> str:
    items = FunctionCallParser.ToolCallParserEnum.items()
    for parser_type, parser_cls in items:
        parser = parser_cls()
        try:
            # This is when processing_class is a tokenizer
            tokenizer_vocab = processing_class.get_vocab()
        except AttributeError:
            try:
                # This is when processing_class is a processor
                tokenizer_vocab = processing_class.tokenizer.get_vocab()
            except AttributeError as e:
                raise ValueError(f"Cannot get vocab from processing_class {processing_class}") from e

        if parser.bot_token.strip() in tokenizer_vocab and (
                parser.eot_token == "" or parser.eot_token.strip() in tokenizer_vocab):
            return parser_type
    else:
        raise ValueError(f"No tool call parser found for processing_class {processing_class}")


def initialize_tools(config, processing_class):
    if config.multi_turn.tool_config_path is None:
        return [], {}, None, [], None

    tools_config_file = config.multi_turn.tool_config_path
    tool_list = initialize_tools_from_config(tools_config_file)

    tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
    tool_map = {tool.name: tool for tool in tool_list}
    tool_call_parser_type = get_tool_call_parser_type(processing_class)
    sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
    function_call_parser = FunctionCallParser(
        sgl_tools,
        tool_call_parser_type,
    )

    return (
        tool_schemas,
        tool_map,
        tool_call_parser_type,
        sgl_tools,
        function_call_parser,
    )


def get_instruction_from_tool_call(response: str, function_call_parser: FunctionCallParser) -> str:
    try:
        normed_content, tool_calls = function_call_parser.parse_non_stream(response)
        instruction = eval(tool_calls[0].parameters)['instruction']
        return instruction
    except Exception as e:
        print(f"Error parsing tool call: {e}")
        return ""


def convert_dataset_to_grounding_format(dataset: pd.DataFrame) -> Dataset:
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

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(build_chunk, s) for s in starts]
        ds_chunks = [f.result() for f in as_completed(futures)]

    grounding_dataset = concatenate_datasets(ds_chunks)
    print("✅ all chunks concatenated →", len(grounding_dataset), "samples")

    return grounding_dataset


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    (
        tool_schemas,
        tool_map,
        tool_call_parser_type,
        sgl_tools,
        function_call_parser,
    ) = initialize_tools(config.rollout, processor)

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)

    out_path = config.data.output_path
    ckpt_path = out_path + ".ckpt"
    makedirs(os.path.dirname(out_path), exist_ok=True)

    rl_dataset = create_rl_dataset(config.data.path, config.data, tokenizer, processor)
    test_dataloader = StatefulDataLoader(
        dataset=rl_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.get("dataloader_num_workers", 8),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            ck = pickle.load(f)
        output_lst = ck["output_lst"]
        start_batch = ck["last_idx"] + 1
        print(f"🔄 Resume from batch {start_batch}")
    else:
        output_lst = []
        start_batch = 0

    num_batch = len(test_dataloader)
    print(f"Total {num_batch} batches, starting from {start_batch}")
    for idx, test_data in enumerate(tqdm(test_dataloader, desc="Generating")):
        if idx < start_batch:
            continue

        test_batch = DataProto.from_single_dict(test_data)

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        test_gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        test_gen_batch.meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": config.rollout.val_kwargs.do_sample,
            "validate": True,
        }

        data_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, wg.world_size)

        print(f"[{idx + 1}/{num_batch}] Start to generate.", flush=True)
        output_padded = wg.generate_sequences(data_padded)
        output = unpad_dataproto(output_padded, pad_size=pad_size)

        output_texts = []
        for i in range(len(output)):
            data_item = output[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = data_item.batch["responses"][:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            output_texts.append(response_str)

        output_lst.extend(output_texts)
        with open(ckpt_path, "wb") as f:
            pickle.dump({"output_lst": output_lst, "last_idx": idx}, f)

    # add to the data frame
    dataset["responses"] = output_lst
    dataset["instruction"] = dataset["responses"].apply(
        lambda resp: get_instruction_from_tool_call(resp, function_call_parser))
    dataset = dataset.loc[dataset["instruction"] != ""].reset_index(drop=True)

    dataset = convert_dataset_to_grounding_format(dataset)
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)

    os.remove(ckpt_path)


if __name__ == "__main__":
    main()
