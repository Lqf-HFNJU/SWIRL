import argparse
import polars as pl
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

import os, json
import re
import ast
import numpy as np


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig



def _eval_one(args):
    solution_str, ground_truth = args
    
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            ground_truth_boxed = "\\boxed{" + ground_truth + "}"
            
            verify_func = math_metric(
                gold_extraction_target=(LatexExtractionConfig(),),
                pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
            )
            
            res, _ = verify_func([ground_truth_boxed], [solution_str])
            return res == 1
            #return is_equiv(answer, ground_truth)
    except Exception as e:
        print('error:', ground_truth)
        return False



def evaluate(file_path, save_path):
    #dataset = pl.read_parquet(file_path).to_pandas()
    dataset = pd.read_parquet(file_path, columns=['prompt', 'reward_model', 'extra_info', 'responses'])

    preds = dataset["responses"].tolist()
    reward_model = dataset["reward_model"].tolist()
    extra_infos = dataset["extra_info"].tolist()

    tasks = []
    for idx, pred in enumerate(preds):
        tasks.append((pred[0], reward_model[idx]['ground_truth']   ))

    # import pdb; pdb.set_trace()
    
    eval_results = []
    for t in tasks:
        eval_results.append(_eval_one(t))
    # with ThreadPoolExecutor(max_workers=max(64, os.cpu_count())) as exe:
    #     eval_results = list(exe.map(_eval_one, tasks))

    dataset["result"] = eval_results
    accuracy = dataset["result"].eq(True).sum() / len(dataset["result"])

    print(f"accuracy: {accuracy}")
    
    # import pdb; pdb.set_trace()
    
    
    def convert_type(x):
        student_prompt = x['student_prompt'].tolist()
        x['student_prompt'] = student_prompt
        return x
    
    
    dataset['extra_info'] = dataset['extra_info'].apply(convert_type)
    dataset['prompt'] = dataset['prompt'].apply(lambda x: x.tolist())
    dataset['responses'] = dataset['responses'].apply(lambda x: x.tolist())
    
    records = dataset.to_dict(orient="records")
    output = {
        "detail": {
            "accuracy": accuracy,
        },
        "data": records
    }
    json.dump(output, open(save_path, "w"), ensure_ascii=False, indent=4)
    return dataset, accuracy

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fp", type=str, default='/mnt/petrelfs/luquanfeng/SWIRL/data/eval_result/math7b_v2/olympiad_round_54.parquet')
    p.add_argument("--out", type=str,  default='/mnt/petrelfs/luquanfeng/SWIRL/data/eval_result/math7b_v2/olympiad_round_54.json')
    p.add_argument("--bp", type=str, default='/mnt/petrelfs/luquanfeng/SWIRL/data/eval_result/math7b_v2')
    p.add_argument("--rx", type=int,  default=6)
    
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.rx == -1:
        fp = args.fp
        save_path = args.out
        
        dataset, accuracy = evaluate(fp, save_path)
        print(f'inference result: {fp}\nsave to: {save_path}')
    else:
        bp = args.bp
        for f in os.listdir(bp):
            if f.find(f'round_{args.rx}.parquet') != -1 and f.find('all') == -1:
                fp = os.path.join(bp, f)
                save_path = fp.replace('.parquet', '.json')
                dataset, accuracy = evaluate(fp, save_path)
                print(f'inference result: {fp}\nsave to: {save_path}', flush=True)