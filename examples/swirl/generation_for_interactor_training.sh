PROJECT_DIR="$(pwd)"


model_path=/path/to/navigator
data_path=$PROJECT_DIR/data/SWIRL_GUI_data/train/stage2_interleaved2000.parquet
save_path=$PROJECT_DIR/data/SWIRL_GUI_data/train/stage2_interleaved2000_Interactor_r1.parquet
tool_config_path=$PROJECT_DIR/examples/swirl/tool_config/gui_executor_tool_config.yaml


data_path="${1:-$data_path}"
save_path="${2:-$save_path}"
model_path="${3:-$model_path}"
tool_config_path="${4:-$tool_config_path}"


LOG_PATH=log/generation4interactor.log


python3 -m multi_agent.data_preporcess.planner_rollout \
    --config-path="$PROJECT_DIR/examples/swirl/running_config" \
    --config-name='planner_generation_default' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$data_path \
    data.batch_size=64 \
    data.output_path=$save_path \
    data.max_prompt_length=4000 \
    model.path=$model_path \
    rollout.multi_turn.tool_config_path=$tool_config_path >> $LOG_PATH 2>&1
