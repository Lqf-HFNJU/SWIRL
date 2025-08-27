import math, re


AITW_ACTION_SPACE = """
click(point='(x1, y1)')
type(content='xxx') # Use escape characters \\' , \\" , and \\n to ensure correct parsing in Python string format.
scroll(direction='down or up or right or left')
press_home()
press_back()
press_enter()
finished() # Submit the task regardless of whether it succeeds or fails.
"""

AMEX_ACTION_SPACE = AITW_ACTION_SPACE

AC_ACTION_SPACE = """
click(point='(x1, y1)')
long_press(point='(x1, y1)')
type(content='xxx') # Use escape characters \\' , \\" , and \\n to ensure correct parsing in Python string format.
scroll(direction='down or up or right or left')
open_app(app_name='xxx')
press_home()
press_back()
wait()
finished() # Submit the task regardless of whether it succeeds or fails.
"""

GUIODYSSEY_ACTION_SPACE = """
click(point='(x1, y1)')
long_press(point='(x1, y1)')
type(content='xxx') # Use escape characters \\' , \\" , and \\n to ensure correct parsing in Python string format.
scroll(direction='down or up or right or left')
press_home()
press_back()
press_appselect() # go to app selection page to show all the opened apps.
error(content='xxx') # submit the task as failed.
finished()
"""

CLICK_ACTION_SPACE = """
click(point='(x1, y1)')
"""

GUIACT_ACTION_SPACE = """
click(point='(x1, y1)')
scroll(direction='down or up')
"""

OMNIACT_ACTION_SPACE = """
click(point='(x1, y1)')
rightclick(point='(x1, y1)')
scroll(direction='down or up')
"""

OMNIACT_DESKTOP_ACTION_SPACE = """
click(point='(x1, y1)') # Single left-click the mouse
rightclick(point='(x1, y1)') # Right-click the mouse
doubleclick(point='(x1, y1)') # Double-click the mouse
moveto(point='(x1, y1)') # Move the mouse cursor, without clicking
scroll(direction='down or up') # Scroll the mouse wheel
"""

ACTION_MAP = {
    "aitz": AITW_ACTION_SPACE,
    "amex": AMEX_ACTION_SPACE,
    "guiodyssey": GUIODYSSEY_ACTION_SPACE,
    "ac": AC_ACTION_SPACE,
    "AndroidControl": AC_ACTION_SPACE,
    "click": CLICK_ACTION_SPACE,
    "guiact": GUIACT_ACTION_SPACE,
    "omniact": OMNIACT_ACTION_SPACE,
    "omniact_desktop": OMNIACT_DESKTOP_ACTION_SPACE
}

pattern = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", re.MULTILINE)
ACTION_MAP_LIST = dict([(k, pattern.findall(v)) for k, v in ACTION_MAP.items()])

print(ACTION_MAP_LIST)


##############################################################################################################################



TOOL_PROMPT = """You need to interact with the Executor Agent by making a tool call: \n<tools>\n{"type": "function", "function": {"name": "executor_agent", "description": "an Executor Agent capable of executing fine-grained instruction", "parameters": {"type": "object", "properties": {"instruction": {"type": "string", "description": "A clear and precise fine-grained instruction for the executor agent"}}, "required": ["instruction"]}, "strict": false}}\n</tools>\n\nReturn a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>"""


NAVIGATOR_PROMPT = """You are a GUI Planner Agent. Your role is to actively collaborate with the Executor Agent to complete complex GUI navigation tasks. Given a task description, the current screenshot, and the action history from the Executor Agent, your goal is to provide a clear and precise fine-grained instruction for the Executor Agent to help accomplish the task.

## Tools
{TOOL_PROMPT}


## Note
- You should first outline the overall task flow and clarify your next intention. Then, generate a fine-grained, precise, and unambiguous instruction that will guide the Executor Agent to execute one of its available actions: {action_space}.
- Please keep your reasoning within <think> </think> tags, and then output the fine-grained instruction as a tool call in the following format:
<think>...</think><tool_call>...</tool_call>


## User Instruction
{instruction}
"""


INTERACTOR_PROMPT = """
You are a reasoning GUI Executor Agent. Given the attached UI screenshot and the instruction: "{instruction}", please determine the next action to fulfill the instruction. 

## Action Space
{action_space}

## Note
- Please keep your reasoning in <think> </think> tags brief and focused. Output the final action in <answer> </answer> tags:
<think>...</think><answer>...</answer>
"""

def make_prompt(category, data_source, instruction=None):
    if instruction is None:
        instruction = "{instruction}"
    if category == 'planner':
        action_space = ACTION_MAP_LIST[data_source]
        return NAVIGATOR_PROMPT.format(TOOL_PROMPT=TOOL_PROMPT, instruction=instruction, action_space=action_space)
    elif category == 'executor':
        action_space = ACTION_MAP[data_source]
        return INTERACTOR_PROMPT.format(instruction=instruction, action_space=action_space)
    else:
        raise ValueError(f"category {category} no find.")




##############################################################################################################################



IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 2048 * 28 * 28
MAX_HISTORY_PIXELS = 128 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
        height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

print(smart_resize(2208, 1840, max_pixels=1024*28*28))