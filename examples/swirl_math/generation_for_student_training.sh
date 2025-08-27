PROJECT_DIR="$(pwd)"


data_path=/path/to/data
save_path=/path/to/saved/data
model_path=/path/to/model


data_path="${1:-$data_path}"
save_path="${2:-$save_path}"
model_path="${3:-$model_path}"


LOG_PATH=log/generation4student.log


python3 -m multi_agent.data_preporcess.teacher_rollout \
    --config-path="$PROJECT_DIR/examples/swirl_math/running_config" \
    --config-name='planner_generation_default' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$data_path \
    data.batch_size=128 \
    data.output_path=$save_path \
    data.max_prompt_length=5120 \
    data.max_response_length=200 \
    data.filter_overlong_prompts=True \
    model.path=$model_path >> $LOG_PATH 2>&1
