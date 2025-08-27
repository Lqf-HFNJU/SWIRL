from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch
import json
from openai import OpenAI
from verl import DataProto
from verl.workers.reward_manager import register
from functools import partial

import numpy.random as npr




@register("teacher")
class TeacherRewardManager:

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", executor_agent_config="") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

        self.executor_creater = []

        executor_agent_config = json.load(open(executor_agent_config))
        executor_agent_urls = executor_agent_config['executor_agent_urls']
        executor_agent_path = executor_agent_config['executor_agent_path']

        for url in executor_agent_urls:
            client = OpenAI(base_url=url, api_key="None", timeout=10.0)
            msg_create = partial(
                client.chat.completions.create,
                model=executor_agent_path,
            )
            self.executor_creater.append(msg_create)

        self.client_num = len(self.executor_creater)


    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        jobs = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            if len(data) < self.client_num:
                url_choice_index = npr.randint(0, self.client_num)
            else:
                url_choice_index = i % self.client_num
            executor_client = self.executor_creater[url_choice_index]

            jobs.append((i, valid_response_length, response_str, ground_truth, extra_info, executor_client, data_source))

        def _compute_one(job):
            i, valid_response_length, response_str, ground_truth, extra_info, executor_client, data_source = job
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                creater=executor_client,
            )
            return i, valid_response_length, score

        with ThreadPoolExecutor(max_workers=min(16, len(jobs))) as exe:
            futures = [exe.submit(_compute_one, job) for job in jobs]
            for future in as_completed(futures):
                i, valid_response_length, score = future.result()

                if isinstance(score, dict):
                    reward = score["score"]
                    for k, v in score.items():
                        reward_extra_info[k].append(v)
                else:
                    reward = score

                reward_tensor[i, valid_response_length - 1] = reward
        

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
