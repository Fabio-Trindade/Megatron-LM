#!/bin/bash

rm forward_info.csv
rm train_time_log.csv

# Script to run the "175B" parameter model

# Environment variable to limit CUDA connections
export CUDA_DEVICE_MAX_CONNECTIONS=1

# GPU and node configuration
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))

# Paths configuration
ROOT_PATH=/workspace/Megatron-LM/
CHECKPOINT_PATH=${ROOT_PATH}m-Llama-3.1-8B-2t-4p
TENSORBOARD_LOGS_PATH=${ROOT_PATH}tensorboard_logs/
VOCAB_FILE=${ROOT_PATH}llama_vocab_3_1_8B.json
DATA_PATH=./llama_text_sentence

# Distributed training arguments
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

# GPT model configuration
GPT_MODEL_ARGS=(
    --num-layers 32 
    --hidden-size 4096 
    --num-attention-heads 32 
    --seq-length 8192 
    --ffn-hidden-size 14336 
    --max-position-embeddings 131072 
)

# Training arguments
TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 16  
    --train-iters 43 
    --lr-decay-iters 43000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .1 
    --exit-on-missing-checkpoint 
    --use-checkpoint-args 
    --untie-embeddings-and-output-weights 
    --normalization RMSNorm 
    --position-embedding-type rope  
    --use-rope-scaling 
    --swiglu 
    --no-masked-softmax-fusion 
    --no-load-optim 
    --no-load-rng 
    --attention-softmax-in-fp32 
    --tokenizer-type HuggingFaceTokenizer 
    --tokenizer-model ./Llama-3.1-8B 
    --num-query-groups 8
    --group-query-attention 
    --disable-bias-linear 
    --transformer-impl transformer_engine 
    --attention-dropout 0.0 
    --hidden-dropout 0.0 
    --rotary-base 500000 
    --rotary-percent 1.0 
)

# Model parallelization configuration
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2 
    --pipeline-model-parallel-size 4  
)

# Data arguments
DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --split 100,0,0
)

# Evaluation and logging arguments
EVAL_AND_LOGGING_ARGS=(
    --log-interval 50
    --save-interval 50
    --eval-interval 50 
    --load $CHECKPOINT_PATH 
    --eval-iters 0
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# Launch training with torchrun
torchrun "${DISTRIBUTED_ARGS[@]}" pretrain_gpt.py \
    "${GPT_MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}"
