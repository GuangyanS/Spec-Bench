#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu&hbm80g
#SBATCH -G 1
#SBATCH -q regular
#SBATCH --ntasks-per-node=1
#SBATCH -J gpt2-xl-cnn
#SBATCH --mail-user=g.sun@rochester.edu
#SBATCH --mail-type=ALL
#SBATCH -t 01:00:00
#SBATCH -A m4705
 
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export GPT2_PATH=openai-community/gpt2-xl
export DRAFTER_PATH=openai-community/gpt2
export MODEL_NAME=gpt2-xl
export bench_NAME=cnn_dailymail
export torch_dtype=float32
export TEMP=0.0
export CUDA_VISIBLE_DEVICES=0,1
module load conda
conda activate spec
 
python -m evaluation.inference_baseline --model-path $GPT2_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
 
python -m evaluation.inference_sps --model-path $GPT2_PATH --drafter-path $DRAFTER_PATH --model-id ${MODEL_NAME}-sps-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype

 
 