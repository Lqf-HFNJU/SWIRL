import logging
import os, json
import base64
import subprocess
import random
import atexit
import time
from typing import Any, Optional, Tuple
from uuid import uuid4
from urllib.parse import urlparse
from openai import OpenAI

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


class GroundingTool(BaseTool):
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
                "name": "grounding_agent",
                "description": "A GUI grounding agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "A clear and precise instruction for the GUI grounding agent",
                        },
                    },
                    "required": ["instruction"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self.max_tries = config.get("max_tries", 5)
        self.grounding_image_max_tokens = config.get("grounding_image_max_tokens", 2048)
        self.grounding_agent_config = config.get("grounding_agent_config", None)
        
        # self.grounding_agent_urls = config.get("grounding_agent_urls", [])
        # self.grounding_agent_path = config.get("grounding_agent_path", "Qwen/Qwen2.5-VL-3B-Instruct")
        # self.grounding_agent_ports = config.get("grounding_agent_ports", [8000, 8001])
        if self.grounding_agent_config:
            self.grounding_agent_config = json.load(open(self.grounding_agent_config))
            self.grounding_agent_urls = self.grounding_agent_config['grounding_agent_urls']
            self.grounding_agent_path = self.grounding_agent_config['grounding_agent_path']
            self.grounding_agent_ports = self.grounding_agent_config['grounding_agent_ports']
        if len(self.grounding_agent_urls) == []:
            raise ValueError('could not find grounding_agent_urls.')
        self._instance_dict = {}

        atexit.register(self._cleanup)

    def _cleanup(self):
        for p in getattr(self, "grounding_agent_processes", []):
            try:
                p.terminate()
                p.wait()
            except Exception:
                pass

    def __del__(self):
        self._cleanup()

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        if len(self.grounding_agent_urls) == 0:
            raise ValueError("did not find available urls")
        _url = random.choice(self.grounding_agent_urls)
        client = OpenAI(base_url=_url, api_key="None")

        img_path = kwargs.get("img_path")
        self._instance_dict[instance_id] = {
            "client": client,
            "grounding_output": "",
            "img_path": img_path,
            "timeout": 10.0
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        grounding_output = None
        client = self._instance_dict[instance_id]["client"]
        grounding_model = self.grounding_agent_path
        img_path = self._instance_dict[instance_id]["img_path"]
        instruction = parameters.get("instruction", "")
        
        img = img_path if img_path.startswith('data:') else encode_image_to_data_uri(img_path)

        num_try = 0
        while num_try < self.max_tries:
            try:
                text=(
                        f'\nYou are a reasoning GUI Grounding Agent. Given the attached UI screenshot and the instruction: "{instruction}", please determine the most likely coordinate to click in order to fulfill the instruction. \nPlease keep your reasoning in <think> </think> tags brief and focused. Output the suggested coordinates in <answer> </answer> tags:\n<think> ... </think><answer>(x, y)</answer>\n'
                    )
                response = client.chat.completions.create(
                    model=grounding_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": img}},
                                {"type": "text", "text": text,},
                            ],
                        }
                    ],
                    max_tokens=128,
                    temperature=1.0,
                    top_p=0.9,
                )
                grounding_output = response.choices[0].message.content
                
                # TODO: resize coordinate
                
                
                self._instance_dict[instance_id]["grounding_output"] = grounding_output
                break
            except Exception as e:
                num_try += 1
                logger.warning(f"[{instance_id}] grounding try {num_try}/{self.max_tries} failed: {e}")
                await asyncio.sleep(0.5)

        if grounding_output is None:
            logger.error(f"[{instance_id}] grounding failed after {self.max_tries} attempts, instruction={instruction}")
            grounding_output = "Unable to provide the coordinate at the moment. Please try again later."
            self._instance_dict[instance_id]["grounding_output"] = grounding_output
        print('[planning_agent] input:', instruction, flush=True)
        print("[grounding_agent] output:", grounding_output, flush=True)
        
        return grounding_output, 0.0, {"grounding_agent_output": grounding_output}


    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
