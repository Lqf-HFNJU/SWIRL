GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM % 40001 + 10000))
GPUS=$((GPUS_PER_NODE * NNODES))
CPUS=$((GPUS_PER_NODE * 16))


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

DEEPSPEED=./scripts/zero1.json

llm=Qwen/Qwen2.5-VL-3B-Instruct

lr=1e-6
batch_size=2
grad_accum_steps=2

# Specify your dataset path in SWIRL/qwen-vl-finetune/qwenvl/data/__init__.py at lines 32–33
datasets=navigator_s1_sft%100

run_name="navigator_s1_sft"
output_dir=./output/$run_name

mkdir -p log/"$run_name"

args="
    --deepspeed ${DEEPSPEED} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 1605632 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to wandb"


python -m torch.distributed.run $DISTRIBUTED_ARGS qwenvl/train/train_qwen.py ${args}