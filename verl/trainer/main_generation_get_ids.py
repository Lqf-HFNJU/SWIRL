
import os
import json
import hydra
import ray
from tqdm import tqdm
from verl.utils.fs import copy_to_local
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils import hf_tokenizer, hf_processor
from verl import DataProto
from verl.utils.hdfs_io import makedirs
from verl.trainer.main_ppo import create_rl_dataset
from verl.utils.dataset.rl_dataset import collate_fn


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


    rl_dataset = create_rl_dataset(config.data.path, config.data, tokenizer, processor)
    test_dataloader = StatefulDataLoader(
        dataset=rl_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.get("dataloader_num_workers", 8),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    all_index = []

    for idx, test_data in enumerate(tqdm(test_dataloader, desc="Generating")):
        test_batch = DataProto.from_single_dict(test_data)

        index = test_batch.non_tensor_batch['index'].tolist()
        all_index += index

    savefp = 'tmp/all_index.json'
    
    json.dump(all_index, open(savefp, 'w'), indent=4)
    print(len(savefp))


if __name__ == "__main__":
    main()