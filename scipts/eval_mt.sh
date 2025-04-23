#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu&hbm80g
#SBATCH -G 1
#SBATCH -q regular
#SBATCH --ntasks-per-node=1
#SBATCH -J gpt2-xl-mt
#SBATCH --mail-user=g.sun@rochester.edu
#SBATCH --mail-type=ALL
#SBATCH -t 04:00:00
#SBATCH -A m4705
 
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export VICUNA_PATH=lmsys/vicuna-33b-v1.3
export DRAFTER_PATH=double7/vicuna-68m
export MODEL_NAME=vicuna-33b-v1.3
export bench_NAME=mt_bench
export torch_dtype=float32
export TEMP=0.0

source activate spec
 
python -m evaluation.inference_baseline --model-path $VICUNA_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype