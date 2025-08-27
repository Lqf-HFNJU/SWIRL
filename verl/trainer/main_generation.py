# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""

import os

import pickle
import hydra
import numpy as np
import ray
from tqdm import tqdm

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.trainer.main_ppo import create_rl_dataset
from verl.utils.dataset.rl_dataset import collate_fn

import random
import torch
seed = 201830168
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


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

    if config.rollout.val_kwargs.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

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
        try:
            with open(ckpt_path, "rb") as f:
                ck = pickle.load(f)
            output_lst = ck["output_lst"]
            start_batch = ck["last_idx"] + 1
            print(f"🔄 Resume from batch {start_batch}")
        except:
            output_lst = [[] for _ in range(config.data.n_samples)]
            start_batch = 0
    else:
        output_lst = [[] for _ in range(config.data.n_samples)]
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

        # START TO GENERATE FOR n_samples TIMES
        print(f"[{idx + 1}/{num_batch}] Start to generate.", flush=True)
        for n_sample in range(config.data.n_samples):
            try:
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
            except:
                raise ValueError('error')
                output_texts = ['error'] * len(output)

            output_lst[n_sample].extend(output_texts)
            print(f'[response]:{output_texts[0]}')
            with open(ckpt_path, "wb") as f:
                pickle.dump({"output_lst": output_lst, "last_idx": idx}, f)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    dataset["responses"] = output_lst

    dataset.to_parquet(config.data.output_path)
    os.remove(ckpt_path)


if __name__ == "__main__":
    main()