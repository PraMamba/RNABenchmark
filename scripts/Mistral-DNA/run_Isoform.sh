#!/bin/bash

set -eu
export TOKENIZERS_PARALLELISM=false

task='Isoform'
token_type='single'
model_type='Mistral-DNA'
model_name_or_path="/pri_exthome/Mamba/Project/GRE_EMB/Evaluate/BEACON/Pretrain/Mistral-DNA/Mistral-DNA-v1-422M-hg38"
model_max_length=512
dataset_dir="/pri_exthome/Mamba/Project/GRE_EMB/Evaluate/BEACON/Data/${task}"
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
output_dir="/pri_exthome/Mamba/Project/GRE_EMB/Evaluate/BEACON/FineTurn/${task}/${model_type}/Mistral-DNA-v1-422M-hg38"
cache_dir="/pri_exthome/Mamba/HuggingFace_Cache/cache"
batch_size=32
attn_implementation="flash_attention_2"
seed=42

mkdir -p "${output_dir}"
log_file="${output_dir}/model_train.log"
if [ -f "$log_file" ]; then
    echo "Overwrite Log: $log_file"
    > "$log_file"
else
    echo "Create Log: $log_file"
    touch "$log_file"
fi

echo "=============================================="
echo "Real-Time Training Log Monitoring"
echo "tail -f ${log_file}"
echo "=============================================="

common_args=\
"
    --output_dir ${output_dir} \
    --model_name_or_path ${model_name_or_path} \
    --data_path ${dataset_dir} \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test} \
    --model_max_length ${model_max_length} \
    --num_train_epochs 50 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --learning_rate 2e-5 \
    --optim adamw_torch_fused \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.3 \
    --logging_strategy steps \
    --logging_steps 2 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 2 \
    --save_safetensors True \
    --save_only_model False \
    --load_best_model_at_end True \
    --metric_for_best_model eval_R^2 \
    --greater_is_better True \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 4 \
    --dataloader_drop_last False \
    --overwrite_output_dir True \
    --ddp_timeout 30000 \
    --log_on_each_node False \
    --logging_first_step True \
    --bf16 True \
    --fp16 False \
    --attn_implementation ${attn_implementation} \
    --report_to wandb \
    --ddp_find_unused_parameters True \
    --gradient_checkpointing False \
    --do_train True \
    --do_eval True \
    --trust_remote_code True \
    --seed ${seed} \
    --data_seed ${seed} \
    --cache_dir ${cache_dir} \
    --token_type ${token_type} \
    --model_type ${model_type} \
    --run_name ${task}_${model_type} \
    --batch_eval_metrics True
"


train_type='torchrun'

if [[ "$train_type" == "torchrun" ]]; then
    echo "Using TorchRun"
    GPU_DEVICES="0,1,2,3,4,5,6,7"
    NUM_PROC=8
    CUDA_VISIBLE_DEVICES=${GPU_DEVICES} torchrun --nproc_per_node=${NUM_PROC} downstream/train_isoform.py $common_args >> "${log_file}" 2>&1
else
    echo "Using DeepSpeed"
    GPU_DEVICES="localhost:0,1,2,3,4,5,6,7"
    MASTER_PORT=$(shuf -i 10000-45000 -n 1)
    DEEPSPEED_CONFIG=~/DeepSpeed_Zero_Config/ds_zero2.json
    deepspeed --include ${GPU_DEVICES} --master_port ${MASTER_PORT} downstream/train_isoform.py --deepspeed ${DEEPSPEED_CONFIG} $common_args >> "${log_file}" 2>&1
fi

