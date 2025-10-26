import argparse
import polars as pl
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

import os, json
import re
import ast
import numpy as np

CLICK_COORD_THRESHOLD = 0.14
TEXT_F1_THRESHOLD = 0.5


def list_depth(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, list):
        return 0
    return 1 + max((list_depth(item) for item in x), default=0)


def extract_answer_action(text: str) -> tuple[str | None, str | None]:
    pattern = re.compile(
        r'<answer>\s*'
        r'(?P<name>[a-z_]+)\('
        r'(?P<params>.*?)'
        r'\)\s*</answer>',
        re.DOTALL
    )

    m = pattern.search(text)
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
                if gt_point[0] == -1:
                    gt_point = list(gt_point)
                    gt_point[0] = (bbox[0][0] + bbox[1][0]) / 2
                    gt_point[1] = (bbox[0][1] + bbox[1][1]) / 2
                
                # if bbox[0][0] <= pred_point[0] <= bbox[1][0] and bbox[0][1] <= pred_point[1] <= bbox[1][1]:
                #     return True
            elif bbox_depth == 3:
                print('bbox_depth: 3')
                for candidate in bbox:
                    if candidate[0][0] <= pred_point[0] <= candidate[1][0] and candidate[0][1] <= pred_point[1] <= \
                            candidate[1][1]:
                        return True
            else:
                print(f'bbox_depth: {bbox_depth} not match.')

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
    bool, bool]:
    if gt_action != pred_action:
        action_match_score, param_match_score = False, False
    else:
        action_match_score = True
        if pred_action in ['press_home', 'press_back', 'press_enter', 'wait', 'finished', 'press_appselect', 'error']:
            param_match_score = True
        elif pred_action in ["click", "long_press", 'rightclick', 'moveto', 'doubleclick']:
            param_match_score = click_matching(gt_params, pred_params, extra_info)
        elif pred_action in ["open_app"]:
            param_match_score = text_matching(gt_params, pred_params, pfx='app_name')
        elif pred_action in ["type"]:
            param_match_score = text_matching(gt_params, pred_params, pfx='content')
        elif pred_action in ["scroll"]:
            param_match_score = scroll_matching(gt_params, pred_params)
        else:
            raise ValueError('unexpected action.')

    return action_match_score, param_match_score


def _eval_one(args):
    pred, extra_info = args
    assert len(pred) == 1

    if pred[0].find('<answer>') == -1:
        pred[0] = f'<answer>{pred[0]}</answer>'

    pred_action, pred_params = extract_answer_action(pred[0])
    gt_action, gt_params = extract_answer_action(f"<answer>{extra_info['answer']}</answer>")

    return compute_match_score(pred_action, gt_action, pred_params, gt_params, extra_info)


def evaluate(file_path, save_path):
    #dataset = pl.read_parquet(file_path).to_pandas()
    dataset = pd.read_parquet(file_path, columns=['prompt', 'reward_model', 'extra_info', 'responses'])
    # dataset = pd.read_parquet(file_path) #, engine="pyarrow")

    preds = dataset["responses"].tolist()
    extra_infos = dataset["extra_info"].tolist()

    tasks = []
    for idx, pred in enumerate(preds):
        tasks.append((pred, extra_infos[idx]))

    # import pdb; pdb.set_trace()
    with ThreadPoolExecutor(max_workers=max(64, os.cpu_count())) as exe:
        eval_results = list(exe.map(_eval_one, tasks))

    type_match, full_match = map(list, zip(*eval_results))

    dataset["type_match"], dataset["full_match"] = type_match, full_match
    SR = dataset["full_match"].mean()
    TYPE = dataset["type_match"].mean()

    mask_action = dataset["extra_info"].apply(lambda x: x["answer"].split("(")[0]).isin(["click", "long_press", 'rightclick', 'moveto', 'doubleclick'])
    mask = mask_action & dataset["type_match"]
    num_gr = dataset.loc[mask, "full_match"].sum()

    denom = mask.sum()
    GR = dataset.loc[mask, "full_match"].mean() if denom > 0 else 0.0

    TYPE_score = f'{sum(dataset["type_match"])} / {len(dataset)} = {TYPE:.2%}'
    GR_score = f'{num_gr} / {denom} = {GR:.2%}'
    SR_score = f'{sum(dataset["full_match"])} / {len(dataset)} = {SR:.2%}'

    print(f"TYPE: {TYPE_score}")
    print(f"GR: {GR_score}")
    print(f"SR: {SR_score}")

    # dataset = dataset.drop(columns=["images", "data_source", "ability"])
    
    def convert_type(x):
        x.pop('tools_kwargs', None)
        bbox = x['bbox']
        bbox = [b.tolist() for b in bbox]
        x['bbox'] = bbox
        return x
    
    
    dataset['extra_info'] = dataset['extra_info'].apply(convert_type)
    dataset['prompt'] = dataset['prompt'].apply(lambda x: x.tolist())
    dataset['responses'] = dataset['responses'].apply(lambda x: x.tolist())
    
    
    records = dataset.to_dict(orient="records")
    output = {
        "detail": {
            "TYPE": TYPE_score,
            "GR": GR_score,
            "SR": SR_score,
        },
        "data": records
    }
    json.dump(output, open(save_path, "w"), ensure_ascii=False, indent=4)
    return dataset, SR, TYPE, GR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fp", type=str)
    p.add_argument("--out", type=str)
    
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    fp = args.fp
    save_path = args.out
    
    dataset, AMS_strict, TMS_strict, GR = evaluate(fp, save_path)
    print(f'inference result: {fp}\nsave to: {save_path}')
