import base64
import os, json
import re
import ast
import asyncio
import time
import sys
import numpy as np

CLICK_COORD_THRESHOLD = 0.14
TEXT_F1_THRESHOLD = 0.5


def is_thought_tool_block(text: str, function_call_parser) -> tuple[bool, str]:
    pattern = re.compile(
        r'^<think>.*?</think><tool_call>.*?</tool_call>$',
        re.DOTALL
    )

    m = pattern.match(text)
    instruction = get_instruction_from_tool_call(text, function_call_parser)

    if not m or instruction == "":
        return False, ""
    return True, instruction


def is_thought_action_block(text: str) -> tuple[bool, str]:
    pattern = re.compile(
        r'^<think>.*?</think><answer>(?P<answer>.*?)</answer>$',
        re.DOTALL
    )

    m = pattern.match(text)
    if not m:
        return False, ""
    return True, m.group('answer').strip()


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


def list_depth(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, list):
        return 0
    return 1 + max((list_depth(item) for item in x), default=0)


def get_instruction_from_tool_call(response: str, function_call_parser) -> str:
    try:
        normed_content, tool_calls = function_call_parser.parse_non_stream(response)
        instruction = json.loads(tool_calls[0].parameters)['instruction']
        # instruction = eval(tool_calls[0].parameters)['instruction']
        return instruction
    except Exception as e:
        print(f"Error parsing tool call: {e}, response: {response}")
        return ""


def is_valid_action(action: str) -> bool:
    BOX = r"\(\d+,\s*\d+\)"

    PATTERNS = [
        re.compile(rf"^click\(point='{BOX}'\)$"),
        re.compile(rf"^long_press\(point='{BOX}'\)$"),
        re.compile(r"^type\(content='[^']*'\)$"),
        re.compile(r"^scroll\(direction='(?:down|up|left|right)'\)$"),
        re.compile(r"^open_app\(app_name='[^']*'\)$"),
        re.compile(r"^press_home\(\)$"),
        re.compile(r"^press_back\(\)$"),
        re.compile(r"^press_enter\(\)$"),
        re.compile(r"^press_appselect\(\)$"),
        re.compile(r"^wait\(\)$"),
        re.compile(r"^finished\(\)$"),
        re.compile(r"^error\(content='[^']*'\)$"),
    ]
    return any(p.match(action) for p in PATTERNS)


def extract_answer_action(text: str) -> tuple[str | None, str | None]:
    pattern = re.compile(
        r'^[\s\S]*?<answer>\s*'
        r'(?P<name>[a-z_]+)\('
        r'(?P<params>.*?)'
        r'\)\s*</answer>\s*$',
        re.MULTILINE
    )

    m = pattern.match(text)
    if not m:
        return None, None
    return m.group('name'), m.group('params').strip()


def parse_params(param_str: str) -> dict:
    param_str = param_str.strip()
    pattern = re.compile(r"(\w+)\s*=\s*('(?:\\.|[^'])*')")

    result = {}
    for key, raw_val in pattern.findall(param_str):
        try:
            val = ast.literal_eval(raw_val)
        except Exception:
            val = raw_val[1:-1]
        result[key] = val
    return result


def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    precision = 0 if len(predicted_tokens) == 0 else len(common_tokens) / len(predicted_tokens)
    recall = 0 if len(ground_truth_tokens) == 0 else len(common_tokens) / len(ground_truth_tokens)
    f1_score = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
    return f1_score


### action score compute
def text_matching(gt, pred, pfx):
    pred_info = parse_params(pred)
    gt_info = parse_params(gt)

    gt_text = gt_info.get(pfx, None)
    pred_text = pred_info.get(pfx, None)
    if pred_text is None:
        return 0.0

    return calculate_f1_score(pred_text, gt_text) > TEXT_F1_THRESHOLD


def click_matching(gt_info, pred_info, extra_info=None):
    gt_info = parse_params(gt_info)
    pred_info = parse_params(pred_info)

    height, width = extra_info['height'], extra_info['width']

    try:
        pred_point = eval(pred_info['point'])
        gt_point = eval(gt_info['point'])

        bbox = extra_info.get('bbox', [])
        bbox_depth = list_depth(bbox)
        
        if len(bbox) != 0:
            if bbox_depth == 2:
                # print('bbox_depth: 2')
                if bbox[0][0] <= pred_point[0] <= bbox[1][0] and bbox[0][1] <= pred_point[1] <= bbox[1][1]:
                    return True
            elif bbox_depth == 3:
                print('bbox_depth: 3')
                for candidate in bbox:
                    if candidate[0][0] <= pred_point[0] <= candidate[1][0] and candidate[0][1] <= pred_point[1] <= candidate[1][1]:
                        return True
            else:
                print(f'bbox_depth: {bbox_depth} not match.')
            
            # if bbox[0][0] <= pred_point[0] <= bbox[1][0] and bbox[0][1] <= pred_point[1] <= bbox[1][1]:
            #     return True

        pred_point_normed = (pred_point[0] / width, pred_point[1] / height)
        gt_point_normed = (gt_point[0] / width, gt_point[1] / height)

        return (pred_point_normed[0] - gt_point_normed[0]) ** 2 + (
                pred_point_normed[1] - gt_point_normed[1]) ** 2 <= CLICK_COORD_THRESHOLD ** 2
    except Exception:
        print('parse error')
        return False


def scroll_matching(gt_info, pred_info):
    gt_info = parse_params(gt_info)
    pred_info = parse_params(pred_info)
    gt_direction = gt_info.get("direction", "").lower()
    pred_direction = pred_info.get("direction", "").lower()
    return gt_direction == pred_direction


def compute_match_score(pred_action: str, gt_action: str, pred_params: str, gt_params: str, extra_info=None) -> tuple[
    float, float]:
    action_match_score, param_match_score = -1, -1
    if gt_action != pred_action:
        action_match_score, param_match_score = 0.0, 0.0
    else:
        action_match_score = 1.0
        if pred_action in ['press_home', 'press_back', 'press_enter', 'wait', 'finished', 'press_appselect', 'error']:
            param_match_score = 1.0
        elif pred_action in ["click", "long_press"]:
            param_match_score = 1.0 if click_matching(gt_params, pred_params, extra_info) else 0.0
        elif pred_action in ["open_app"]:
            param_match_score = 1.0 if text_matching(gt_params, pred_params, pfx='app_name') else 0.0
        elif pred_action in ["type"]:
            param_match_score = 1.0 if text_matching(gt_params, pred_params, pfx='content') else 0.0
        elif pred_action in ["scroll"]:
            param_match_score = 1.0 if scroll_matching(gt_params, pred_params) else 0.0
        else:
            raise ValueError('unexpected action.')

    return action_match_score, param_match_score


def format_reward(predict_str: str) -> float:
    res, action = is_thought_action_block(predict_str)
    match_result = is_valid_action(action)

    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str, extra_info=None) -> float:
    pred_action, pred_params = extract_answer_action(predict_str)
    gt_action, gt_params = extract_answer_action(f"<answer>{ground_truth}</answer>")
    action_match_score, param_match_score = compute_match_score(pred_action, gt_action, pred_params, gt_params, extra_info)
    return 0.2 * action_match_score + 0.8 * param_match_score


def gui_interactor_score(predict_str: str, ground_truth: str, extra_info=None) -> float:
    return 0.9 * acc_reward(predict_str, ground_truth, extra_info) + 0.1 * format_reward(predict_str)


def gui_navigator_score(function_call_parser, creater, predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_correct, instruction = is_thought_tool_block(predict_str, function_call_parser)
    prompt_template = extra_info['tools_kwargs']['executor_agent']['create_kwargs']['executor_prompt']

    format_reward = 1.0 if is_format_correct else 0.0
    if instruction == "":
        accuracy_reward = 0.0
    else:
        executor_output = ""
        img = extra_info['tools_kwargs']['executor_agent']['create_kwargs']['img_path']
        img = img if img.startswith('data:') else encode_image_to_data_uri(img)
        num_try = 0

        max_retry_times = 5
        # import pdb; pdb.set_trace()
        while num_try < max_retry_times:
            try:
                text = prompt_template.format(instruction=instruction)
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
                break
            except Exception as e:
                num_try += 1
                print(f"[{instruction}] executing try {num_try}/{max_retry_times} failed: {e}")
                time.sleep(1)
                # await asyncio.sleep(0.5)

        executor_score = acc_reward(executor_output, ground_truth, extra_info)
        #print(f'[instruction]: {instruction}\n[executor_output]: {executor_output}\n[executor_score]: {executor_score}')
        accuracy_reward = executor_score

    return format_reward * 0.1 + accuracy_reward * 0.9


def compute_score(data_source, solution_str, ground_truth, extra_info=None, function_call_parser=None, creater=None):
    if data_source == 'executor':
        res = gui_interactor_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source == 'planner':
        res = gui_navigator_score(function_call_parser, creater, solution_str, ground_truth, extra_info)
    else:
        raise ValueError(f"Unknown data source: {data_source}")
    # print(
    #     f"----------\n[score]:{res}\n[response]:{solution_str}\n[gt]:{ground_truth}\n[high_level_instruction]:{extra_info['high_level_instruction']}\n[low_level_instruction]:{extra_info['low_level_instruction']}\n----------",
    #     flush=True)
    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


if __name__ == '__main__':
    pass
    guiri = """<think>Tap on the 'Rename' option in the menu.<\/think><answer>click(point='(402, 683)')<\/answer>"""
    gt = "click(point='(531, 684)')"
    # [[289, 662], [774, 711]]
    extra_info = {
            "answer":"click(point='(531, 684)')",
            "bbox": [[289, 662], [774, 711]],
            "height":1876,
            "high_level_instruction":"i want to change the first recording title to \"birthday song\" using the recorder app",
            "index":"AndroidControl-16464-2",
            "low_level_instruction":"choose  last third option (Rename )from a drop down  at the screen ",
            "width":840
        }

    res = gui_compute_score(guiri, gt, extra_info=extra_info)
    print(res)
