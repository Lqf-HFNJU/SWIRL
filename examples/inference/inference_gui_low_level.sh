set -x

PROJECT_DIR="$(pwd)"


data_path=$PROJECT_DIR/data/SWIRL_GUI_data/test/low_level/AndroidControl_test.parquet # /path/to/eval/data
save_path=$PROJECT_DIR/eval/SWIRL_GUI_data/test/low_level/AndroidControl_test.parquet # /patg/to/result

interactor_path=/path/to/interactor/model



RAY_TMPDIR=$PROJECT_DIR/ray
export RAY_TEMP_DIR="$RAY_TMPDIR"
export RAY_TMPDIR="$RAY_TMPDIR"
export WANDB_MODE=offline




python3 -m verl.trainer.main_generation \
    --config-path="$PROJECT_DIR/examples/my/running_config" \
    --config-name='planner_generation_default' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$data_path \
    data.n_samples=1 \
    data.output_path=$save_path \
    data.batch_size=64 \
    data.max_prompt_length=4000 \
    data.max_response_length=256 \
    model.path=$interactor_path \
    rollout.name=vllm \
    rollout.val_kwargs.temperature=0.0 \
    rollout.val_kwargs.do_sample=False
