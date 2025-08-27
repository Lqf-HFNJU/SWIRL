set -x

EXP_NAME=SWIRL_GUI
PROJECT_DIR="$(pwd)"

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
MODEL_SAVE_PATH=$PROJECT_DIR/checkpoints/hf/$EXPNAME/navigator_round_0
TRAINER_NAME=warm_up
TRAIN_FILE=$PROJECT_DIR/data/SWIRL_GUI_data/train/stage1_Interactor_warmup1500.parquet # path to training data


LOG_PATH=log/$EXP_NAME
mkdir -p $LOG_PATH

VAL_FILE=$TRAIN_FILE

REWARD_FUNCTION_PATH=$PROJECT_DIR/verl/utils/reward_score/gui.py


RAY_TMPDIR=$PROJECT_DIR/ray
export RAY_TEMP_DIR="$RAY_TMPDIR"
export RAY_TMPDIR="$RAY_TMPDIR"


python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/examples/swirl/running_config" \
    --config-name='ppo_trainer' \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=32 \
    data.max_prompt_length=2500 \
    data.max_response_length=128 \
    +online_reweighting.lb=0.1 \
    +online_reweighting.ub=1.0 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    custom_reward_function.path=$REWARD_FUNCTION_PATH \
    trainer.project_name=$EXP_NAME \
    trainer.experiment_name=$TRAINER_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.total_epochs=5 >> "$LOG_PATH"/"$TRAINER_NAME".log 2>&1

sleep 30

python -m multi_agent.smart_model_merger \
    --local_dir checkpoints/$EXP_NAME/$TRAINER_NAME \
    --target_dir $MODEL_SAVE_PATH
