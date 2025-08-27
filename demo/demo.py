from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image
import re, json
from qwen_vl_utils import process_vision_info

TOOL_PROMPT = """You need to interact with the Executor Agent by making a tool call: \n<tools>\n{"type": "function", "function": {"name": "executor_agent", "description": "an Executor Agent capable of executing fine-grained instruction", "parameters": {"type": "object", "properties": {"instruction": {"type": "string", "description": "A clear and precise fine-grained instruction for the executor agent"}}, "required": ["instruction"]}, "strict": false}}\n</tools>\n\nReturn a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>"""

NAVIGATOR_PROMPT = """You are a GUI Planner Agent. Your role is to actively collaborate with the Executor Agent to complete complex GUI navigation tasks. Given a task description, the current screenshot, and the action history from the Executor Agent, your goal is to provide a clear and precise fine-grained instruction for the Executor Agent to help accomplish the task.

## Tools
{TOOL_PROMPT}


## Note
- You should first outline the overall task flow and clarify your next intention. Then, generate a fine-grained, precise, and unambiguous instruction that will guide the Executor Agent to execute one of its available actions: ['click', 'type', 'scroll', 'press_home', 'press_back', 'press_enter', 'finished'].
- Please keep your reasoning within <think> </think> tags, and then output the fine-grained instruction as a tool call in the following format:
<think>...</think><tool_call>...</tool_call>


## User Instruction
{instruction}
"""

INTERACTOR_PROMPT = """
You are a reasoning GUI Executor Agent. Given the attached UI screenshot and the instruction: "{instruction}", please determine the next action to fulfill the instruction. 

## Action Space

click(point='(x1, y1)')
type(content='xxx') # Use escape characters \\' , \\" , and \\n to ensure correct parsing in Python string format.
scroll(direction='down or up or right or left')
press_home()
press_back()
press_enter()
finished() # Submit the task regardless of whether it succeeds or fails.


## Note
- Please keep your reasoning in <think> </think> tags brief and focused. Output the final action in <answer> </answer> tags:
<think>...</think><answer>...</answer>
"""


def gen_text(llm: LLM, processor, prompt: str, image: str, sampling: SamplingParams) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "user", "content": [{"type": "image", "image": image}]},
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    llm_inputs = {"prompt": prompt, "multi_modal_data": {"image": image_inputs}}

    outputs = llm.generate([llm_inputs], sampling_params=sampling)
    return outputs[0].outputs[0].text


def extract_instruction(text: str) -> str | None:
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
        return payload.get("arguments", {}).get("instruction")
    except json.JSONDecodeError:
        return None


def build_navigator_prompt(task: str) -> str:
    return NAVIGATOR_PROMPT.format(TOOL_PROMPT=TOOL_PROMPT, instruction=task)


def build_interactor_prompt(nav_instruction: str) -> str:
    return INTERACTOR_PROMPT.format(instruction=nav_instruction)


if __name__ == "__main__":
    NAVIGATOR_PATH = "/mnt/data/luquanfeng/model/SWIRL/Navigator"
    INTERACTOR_PATH = "/mnt/data/luquanfeng/model/SWIRL/Interactor"

    COMMON_INIT_KW = dict(
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        max_model_len=4096,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )

    nav_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512)
    int_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512)

    img_path = "demo/demo.jpg"
    user_task = "Find the closest Subway restaurant and order their best-selling combo."


    navigator_llm = LLM(model=NAVIGATOR_PATH, **COMMON_INIT_KW)
    navigator_preprocessor = AutoProcessor.from_pretrained(NAVIGATOR_PATH)
    nav_prompt = build_navigator_prompt(user_task)
    nav_out = gen_text(navigator_llm, navigator_preprocessor, nav_prompt, img_path, nav_params)
    nav_instr = extract_instruction(nav_out)

    del navigator_llm

    interactor_llm = LLM(model=INTERACTOR_PATH, **COMMON_INIT_KW)
    interactor_preprocessor = AutoProcessor.from_pretrained(INTERACTOR_PATH)
    int_prompt = build_interactor_prompt(nav_instr)
    int_out = gen_text(interactor_llm, interactor_preprocessor, int_prompt, img_path, int_params)

    print('\n-------------\n')
    print("\n[NAVIGATOR]\n", nav_out)
    print("\n[INTERACTOR]\n", int_out)