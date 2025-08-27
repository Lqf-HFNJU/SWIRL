import os
import argparse

def find_latest_ckp(ckp_dir):
    info_txt = os.path.join(ckp_dir, 'latest_checkpointed_iteration.txt')
    with open(info_txt, 'r') as f:
        latest_version = f.read().strip()
    latest_ckp = os.path.join(ckp_dir, f'global_step_{latest_version}', 'actor')
    assert os.path.exists(latest_ckp), f"Checkpoint {latest_ckp} does not exist."
    return latest_ckp


def main(args):
    latest_ckp_path = find_latest_ckp(args.local_dir)
    os.system(f"python -m verl.model_merger merge --backend fsdp --local_dir {latest_ckp_path} --target_dir {args.target_dir}")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Find the latest checkpoint.")
    args.add_argument('--local_dir', type=str, required=True, help='Directory containing checkpoints')
    args.add_argument('--target_dir', type=str, required=True, help='Path to save the latest checkpoint')
    args = args.parse_args()
    main(args)