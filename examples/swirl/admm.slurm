#!/bin/bash
#SBATCH --job-name=swirl_gui
#SBATCH -p your_partition
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=120
#SBATCH --quotatype reserved
#SBATCH -o log/swirl.out
#SBATCH -e log/swirl.err

set -x
set -euo pipefail

PROJECT_DIR="$(pwd)"

mapfile -t NODES < <(scontrol show hostnames $SLURM_NODELIST)
SERVER_NODE=${NODES[0]}
CLIENT_NODE=${NODES[1]}

echo $SERVER_NODE
echo $CLIENT_NODE

EXPNAME=SWIRL_GUI

DATA=$PROJECT_DIR/data/SWIRL_GUI_data/train/stage2_interleaved2000.parquet # path to training data


ADMM_MAX_ROUND=20
BASE_INTERACTOR_MODEL=Qwen/Qwen2.5-VL-3B-Instruct # path to the interactor model trained in Stage 1 (warm-up)
BASE_NAVIGATOR_MODEL=Qwen/Qwen2.5-VL-3B-Instruct # path to the navigator model trained in Stage 1 (warm-up)


tool_config_path=$PROJECT_DIR/examples/swirl/tool_config/gui_executor_tool_config.yaml
grounding_config_path=$PROJECT_DIR/examples/swirl/tool_config/executor_agent_config.json
grounding_server_ports=(10030 10031 10032 10033 10034 10035 10036 10037)
PORTS_CSV=$(IFS=,; echo "${grounding_server_ports[*]}")



RAY_TMPDIR=$PROJECT_DIR/ray
export RAY_TEMP_DIR="$RAY_TMPDIR"
export RAY_TMPDIR="$RAY_TMPDIR"



for (( i=1; i<=ADMM_MAX_ROUND; i++ )); do
  echo "ADMM Iteration: $i"
  TRAINER_NAME="round_${i}"

  interactor_save_path="./checkpoints/hf/$EXPNAME/interactor_$TRAINER_NAME"
  navigator_save_path="./checkpoints/hf/$EXPNAME/navigator_$TRAINER_NAME"

  grounder_input_data_path="./data/SWIRL_GUI_data/train/stage2_interleaved2000_interactor_$TRAINER_NAME.parquet"

  # step1: launch vllm interactor agent serve
  pkill -f "vllm serve" || true

  sleep 10

  srun --nodes=1 --ntasks=1 --nodelist=$SERVER_NODE --overlap python multi_agent/launch/server_config.py --model $BASE_INTERACTOR_MODEL --ports "${grounding_server_ports[@]}" --grounding_config_dir $grounding_config_path
  srun --nodes=1 --ntasks=1 --nodelist=$SERVER_NODE -o log/vllm_start.out --overlap bash examples/swirl/start_vllm.sh $BASE_INTERACTOR_MODEL $PORTS_CSV log &

  sleep 180

  # step2: train navigator
  srun --nodes=1 --ntasks=1 --nodelist=$CLIENT_NODE bash examples/swirl/train_navigator.sh $BASE_NAVIGATOR_MODEL $navigator_save_path $TRAINER_NAME $DATA $tool_config_path $grounding_config_path
  BASE_NAVIGATOR_MODEL=$navigator_save_path

  sleep 30

  # step3: generation training data for interactor
  srun --nodes=1 --ntasks=1 --nodelist=$CLIENT_NODE bash examples/swirl/generation_for_interactor_training.sh $DATA $grounder_input_data_path $BASE_NAVIGATOR_MODEL $tool_config_path

  sleep 30

  # step4: train navigator
  srun --nodes=1 --ntasks=1 --nodelist=$CLIENT_NODE bash examples/swirl/train_interactor.sh $BASE_INTERACTOR_MODEL $interactor_save_path $TRAINER_NAME $grounder_input_data_path
  BASE_INTERACTOR_MODEL=$interactor_save_path

  sleep 30

done

echo "All done."

