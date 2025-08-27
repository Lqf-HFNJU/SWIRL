import os
import sys
import json
import signal
import subprocess
from time import sleep
import socket
import argparse


def get_local_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def main(args):
    MODEL = args.model
    PORTS = args.ports
    GROUNDING_CONFIG_DIR = args.grounding_config_dir

    print(MODEL, PORTS, GROUNDING_CONFIG_DIR, flush=True)


    this_ip = get_local_lan_ip()
    urls = [f"http://{this_ip}:{port}/v1" for port in PORTS]

    save_info = {
        "executor_agent_urls": urls,
        "executor_agent_path": MODEL,
    }

    json.dump(save_info, open(GROUNDING_CONFIG_DIR, 'w'), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLLM grounding agent.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--ports", nargs="+", type=int, required=True)
    parser.add_argument("--grounding_config_dir", type=str, default="./examples/my/tool_config/executor_agent_config.json")
    args = parser.parse_args()
    main(args)