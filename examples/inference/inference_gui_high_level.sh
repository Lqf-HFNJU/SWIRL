set -x

PROJECT_DIR="$(pwd)"


data_path=$PROJECT_DIR/data/SWIRL_GUI_data/test/high_level/AndroidControl_test.parquet # /path/to/eval/data
save_path=$PROJECT_DIR/eval/SWIRL_GUI_data/test/high_level/AndroidControl_test.parquet # /patg/to/result


navigator_path=/path/to/navigator/model
interactor_path=/path/to/interactor/model



tools_config_file=$PROJECT_DIR/examples/swirl/tool_config/gui_executor_tool_config.yaml

RAY_TMPDIR=$PROJECT_DIR/ray
export RAY_TEMP_DIR="$RAY_TMPDIR"
export RAY_TMPDIR="$RAY_TMPDIR"
export WANDB_MODE=offline


# navigator inference
python3 -m verl.trainer.main_generation \
    --config-path="$PROJECT_DIR/examples/swirl/running_config" \
    --config-name='planner_generation_default' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.batch_size=64 \
    data.path=$data_path \
    data.output_path=$save_path \
    model.path=$navigator_path

sleep 10


# convert navigator output to interactor input
python -m multi_agent.data_preporcess.plan2exec \
    --filepath $save_path \
    --savepath $save_path \
    --tools_config_file $tools_config_file

sleep 10


# interactor inference
python3 -m verl.trainer.main_generation \
    --config-path="$PROJECT_DIR/examples/swirl/running_config" \
    --config-name='planner_generation_default' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.batch_size=64 \
    data.path=$save_path \
    data.output_path=$save_path \
    model.path=$interactor_path