import logging
import os, json
import base64
import subprocess
import random
import time
from typing import Any, Optional, Tuple
from uuid import uuid4
from urllib.parse import urlparse
from openai import OpenAI
from functools import partial

import socket
import time
import asyncio
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def encode_image_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    else:
        mime = "image/png"
    return f"data:{mime};base64,{img_b64}"


class ExecutingTool(BaseTool):
    """A demo tool for calculating the reward of gsm8k.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "executor_agent",
                "description": "an Executor Agent capable of executing low-level instruction",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "A clear and precise low-level instruction for the executor agent",
                        },
                    },
                    "required": ["instruction"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self.max_tries = config.get("max_tries", 5)
        self.executor_image_max_tokens = config.get("executor_image_max_tokens", 2048)

        executor_agent_config = config.get("executor_agent_config")
        executor_agent_config = json.load(open(executor_agent_config))
        executor_agent_urls = executor_agent_config['executor_agent_urls']
        executor_agent_path = executor_agent_config['executor_agent_path']
        if len(executor_agent_urls) == 0:
            raise ValueError('could not find executor_agent_urls.')

        self.executor_creater = []
        for url in self.executor_creater:
            client = OpenAI(base_url=url, api_key="None", timeout=5.0)
            msg_create = partial(
                client.chat.completions.create,
                model=executor_agent_path,
            )
            self.executor_creater.append(msg_create)

        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        if len(self.executor_creater) == 0:
            raise ValueError("did not find available urls")
        _executor_creater = random.choice(self.executor_creater)

        img_path = kwargs.get("img_path")
        excutor_prompt = kwargs.get("excutor_prompt")
        self._instance_dict[instance_id] = {
            "creater": _executor_creater,
            "executor_output": "",
            "excutor_prompt": excutor_prompt,
            "img_path": img_path,
            "timeout": 5.0
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        executor_output = None
        creater = self._instance_dict[instance_id]["creater"]
        img_path = self._instance_dict[instance_id]["img_path"]
        excutor_prompt = self._instance_dict[instance_id]["excutor_prompt"]
        instruction = parameters.get("instruction", "")

        img = img_path if img_path.startswith('data:') else encode_image_to_data_uri(img_path)

        num_try = 0
        while num_try < self.max_tries:
            try:
                text = excutor_prompt.format(instruction)
                response = creater(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": img}},
                                {"type": "text", "text": text, },
                            ],
                        }
                    ],
                    max_tokens=128,
                    temperature=0.0,
                )
                executor_output = response.choices[0].message.content

                # TODO: resize coordinate

                self._instance_dict[instance_id]["executor_output"] = executor_output
                break
            except Exception as e:
                num_try += 1
                logger.warning(f"[{instance_id}] executor try {num_try}/{self.max_tries} failed: {e}")
                await asyncio.sleep(0.5)

        if executor_output is None:
            logger.error(f"[{instance_id}] executor failed after {self.max_tries} attempts, instruction={instruction}")
            executor_output = "Unable to provide the coordinate at the moment. Please try again later."
            self._instance_dict[instance_id]["executor_output"] = executor_output
        print('[planning_agent] input:', instruction, flush=True)
        print("[executor_agent] output:", executor_output, flush=True)

        return executor_output, 0.0, {"executor_agent_output": executor_output}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
