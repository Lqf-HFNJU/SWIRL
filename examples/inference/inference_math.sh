set -x

PROJECT_DIR="$(pwd)"


data_path=$PROJECT_DIR/data/SWIRL_MATH_data/test/MATH500_test.parquet # /path/to/eval/data
save_path=$PROJECT_DIR/eval/SWIRL_MATH_data/test/MATH500_test.parquet # /patg/to/result



teacher_path=/path/to/teacher/model
student_path=/path/to/student/model


RAY_TMPDIR=$PROJECT_DIR/ray
export RAY_TEMP_DIR="$RAY_TMPDIR"
export RAY_TMPDIR="$RAY_TMPDIR"
export WANDB_MODE=offline


# teacher inference
python3 -m verl.trainer.main_generation \
    --config-path="$PROJECT_DIR/examples/swirl_math/running_config" \
    --config-name='planner_generation_default' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.batch_size=128 \
    data.max_prompt_length=2000 \
    data.max_response_length=2000 \
    data.path=$data_path \
    data.output_path=$save_path \
    model.path=$teacher_path

sleep 10


# convert teacher output to student input
python -m multi_agent.data_preporcess.plan2exec_math \
    --filepath $save_path \
    --savepath $save_path \

sleep 10


# student inference
python3 -m verl.trainer.main_generation \
    --config-path="$PROJECT_DIR/examples/swirl_math/running_config" \
    --config-name='planner_generation_default' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.batch_size=128 \
    data.max_prompt_length=4000 \
    data.max_response_length=4000 \
    data.path=$save_path \
    data.output_path=$save_path \
    model.path=$student_path