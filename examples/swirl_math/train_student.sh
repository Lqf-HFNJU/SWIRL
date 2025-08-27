
PROJECT_DIR="$(pwd)"

MODEL_PATH=/path/to/base/student
MODEL_SAVE_PATH=/path/to/save/student
TRAINER_NAME=round_x
TRAIN_FILE=/path/to/your/training/data


MODEL_PATH="${1:-$MODEL_PATH}"
MODEL_SAVE_PATH="${2:-$MODEL_SAVE_PATH}"
TRAINER_NAME="${3:-$TRAINER_NAME}"
TRAIN_FILE="${4:-$TRAIN_FILE}"


EXP_NAME=student

LOG_PATH=log/$EXP_NAME
mkdir -p $LOG_PATH
VAL_FILE=$TRAIN_FILE

REWARD_FUNCTION_PATH=$PROJECT_DIR/verl/utils/reward_score/math_decouple.py


python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/examples/swirl/running_config" \
    --config-name='ppo_trainer' \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=128 \
    data.max_prompt_length=1280 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    +online_reweighting.lb=0.2 \
    +online_reweighting.ub=0.8 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    custom_reward_function.path=$REWARD_FUNCTION_PATH \
    trainer.project_name=$EXP_NAME \
    trainer.experiment_name=$TRAINER_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.total_epochs=1 >> "$LOG_PATH"/"$TRAINER_NAME".log 2>&1

sleep 30

python -m multi_agent.smart_model_merger \
    --local_dir checkpoints/$EXP_NAME/$TRAINER_NAME \
    --target_dir $MODEL_SAVE_PATH