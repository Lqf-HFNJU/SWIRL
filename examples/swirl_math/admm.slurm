#!/bin/bash
#SBATCH --job-name=swirl_math
#SBATCH -p your_partition
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=120
#SBATCH --quotatype reserved
#SBATCH -o log/swirl_math.out
#SBATCH -e log/swirl_math.err

set -x
set -euo pipefail

PROJECT_DIR="$(pwd)"

mapfile -t NODES < <(scontrol show hostnames $SLURM_NODELIST)
SERVER_NODE=${NODES[0]}
CLIENT_NODE=${NODES[1]}

echo $SERVER_NODE
echo $CLIENT_NODE

EXPNAME=SWIRL_MATH

DATA=$PROJECT_DIR/data/SWIRL_MATH_data/train/MATH_train.parquet # path to training data


ADMM_MAX_ROUND=10
BASE_STUDENT_MODEL=Qwen/Qwen2.5-Coder-3B-Instruct # path to base model
BASE_TEACHER_MODEL=Qwen/Qwen2.5-Coder-3B-Instruct # path to base model

student_config_path=$PROJECT_DIR/examples/swirl_math/tool_config/student_config.json
student_server_ports=(10040 10041 10042 10043 10044 10045 10046 10047)
PORTS_CSV=$(IFS=,; echo "${student_server_ports[*]}")



RAY_TMPDIR=$PROJECT_DIR/ray
export RAY_TEMP_DIR="$RAY_TMPDIR"
export RAY_TMPDIR="$RAY_TMPDIR"



for (( i=1; i<=ADMM_MAX_ROUND; i++ )); do
  echo "ADMM Iteration: $i"
  TRAINER_NAME="round_${i}"

  teacher_save_path="./checkpoints/hf/$EXPNAME/teacher_$TRAINER_NAME"
  student_save_path="./checkpoints/hf/$EXPNAME/student_$TRAINER_NAME"

  student_input_data_path="./data/SWIRL_MATH_data/train/train_student_$TRAINER_NAME.parquet"

  # step1: launch vllm student agent serve
  pkill -f "vllm serve" || true

  sleep 10

  srun --nodes=1 --ntasks=1 --nodelist=$SERVER_NODE --overlap python multi_agent/launch/server_config.py --model $BASE_STUDENT_MODEL --ports "${student_server_ports[@]}" --grounding_config_dir $student_config_path
  srun --nodes=1 --ntasks=1 --nodelist=$SERVER_NODE -o log/vllm_start.out --overlap bash examples/swirl_math/start_vllm.sh $BASE_STUDENT_MODEL $PORTS_CSV log &

  sleep 200

  # step2: train teacher
  srun --nodes=1 --ntasks=1 --nodelist=$CLIENT_NODE bash examples/swirl_math/train_teacher.sh $BASE_TEACHER_MODEL $teacher_save_path $TRAINER_NAME $DATA $student_config_path
  BASE_TEACHER_MODEL=$teacher_save_path

  sleep 30

  # step3: generation training data for student
  srun --nodes=1 --ntasks=1 --nodelist=$CLIENT_NODE bash examples/swirl_math/generation_for_student_training.sh $DATA $student_input_data_path $BASE_TEACHER_MODEL

  sleep 30

  # step4: train student
  srun --nodes=1 --ntasks=1 --nodelist=$CLIENT_NODE bash examples/swirl_math/train_student.sh $BASE_STUDENT_MODEL $student_save_path $TRAINER_NAME $student_input_data_path
  BASE_STUDENT_MODEL=$student_save_path

  sleep 30

done

echo "All done."

