#!/bin/bash

gpu_device="2"
master_port=41611
nproc_per_node=1
partition='ai4bio'
USE_SLURM='0'

MODEL_TYPE='rnalm'

task='DistanceMap'

#  USE_SLURM or not
if [ "$USE_SLURM" == "1" ]; then
    EXEC_PREFIX="srun --job-name=${MODEL_TYPE}_${data} --gres=gpu:$nproc_per_node --cpus-per-task=$(($nproc_per_node * 5)) --mem=50G"
elif [ "$USE_SLURM" == "2" ]; then
    uotatype='vip_gpu_ailab'
    module load anaconda/2021.11
    module load cuda/11.7.0
    module load cudnn/8.6.0.163_cuda11.x
    module load compilers/gcc/9.3.0
    module load llvm/triton-clang_llvm-11.0.1
    source activate dnalm_v2
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/peng_tmp_test/miniconda3/lib
    export CPATH=/usr/include:$CPATH
    export PYTHONUNBUFFERED=1
    export LD_PRELOAD=/home/bingxing2/apps/compilers/gcc/12.2.0/lib64/libstdc++.so.6
    data_root=/home/bingxing2/ailab/group/ai4bio/public/
    EXEC_PREFIX="srun --nodes=1 -p $quotatype -A $partition --job-name=${MODEL_TYPE}_${task} --gres=gpu:$nproc_per_node --cpus-per-task=32 torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"
else
    data_root=/mnt/data/oss_beijing/   
    EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device accelerate launch --num_processes=$nproc_per_node --main_process_port=$master_port"
fi


DATA_PATH=${data_root}multi-omics/RNA/downstream/${task}
batch_size=1
data=''
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test,RFAM19,DIRECT

for token in 'single' 'bpe' 'non-overlap' '6mer' 
do
    for pos in 'ape' 'alibi' 'rope'
    do 

        MODEL_PATH=/mnt/data/ai4bio/renyuchen/RNABenchmark/model/rnalm/config/${MODEL_TYPE}-${token}-${pos}
        OUTPUT_PATH=./outputs/ft/rna-all/${task}/rna/baseline/${MODEL_TYPE}-${token}-${pos}-scratch

        for seed in 42 666 3407
        do

            echo ${MODEL_PATH}

            ${EXEC_PREFIX} \
            downstream/train_distance_map.py \
            --mode bprna \
            --data_path  ${DATA_PATH}/${data} \
            --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
            --model_name_or_path ${MODEL_PATH} \
            --output_dir ${OUTPUT_PATH} \
            --run_name ${MODEL_TYPE}_${data}_seed${seed} \
            --num_epochs 100 \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 2 \
            --lr 3e-5 \
            --num_workers 1 \
            --token_type ${token} \
            --model_type ${MODEL_TYPE} \
            --model_max_length 1026 \
            --train_from_scratch True \

        done
    done
done