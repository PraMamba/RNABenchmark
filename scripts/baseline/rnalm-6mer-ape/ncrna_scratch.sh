#!/bin/bash

# This is your argument
export kmer=6
export MODEL_PATH=/mnt/data/ai4bio/renyuchen/DNABERT/examples/output/rna/base/noncoding_rna_human_6mer_1024/checkpoint-80000
export DATA_PATH=/mnt/data/oss_beijing/multi-omics/RNA/downstream/NoncodingRNAFamily
export OUTPUT_PATH=./outputs/ft/rna-all/ncrna/rna/baseline/rnalm-6mer-ape/scratch
export STAGE=None
export MODEL_TYPE=rnalm
export gpu_device="0"
export master_port=47707
export nproc_per_node=1
export batch_size=8
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
export token='6mer'
export data=''
for seed in 42 666 3407
do

    

    CUDA_VISIBLE_DEVICES=$gpu_device torchrun \
        --nproc_per_node $nproc_per_node \
        --master_port $master_port \
        downstream/train_ncrna.py \
            --model_name_or_path $MODEL_PATH \
            --data_path  $DATA_PATH/$data \
            --kmer ${kmer} \
            --run_name ${MODEL_TYPE}_${data}_seed${seed} \
            --model_max_length 1026 \
            --per_device_train_batch_size $batch_size \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 4 \
            --learning_rate 3e-5 \
            --num_train_epochs 30 \
            --fp16 \
            --save_steps 400 \
            --output_dir ${OUTPUT_PATH}/${data} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 200 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False \
            --stage $STAGE \
            --token_type ${token} \
            --model_type ${MODEL_TYPE} \
            --train_from_scratch True
    
done
