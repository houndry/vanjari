#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=bh-virus
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 5:00:00
# SBATCH -t 8-0:00:00
 
#SBATCH --mem-per-cpu=12G
 
# SBATCH -p cascade
#SBATCH -p gpu-h100,gpu-a100,gpu-a100-short
# SBATCH -p gpu-a100-short,gpu-a100,gpu-a100-preempt
# SBATCH -p gpu-a100-short,gpu-a100
# SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --account=punim2199
 
# Specify your email address to be notified of progress.
#SBATCH --mail-user=robert.turnbull@unimelb.edu.au
#SBATCH --mail-type=ALL
 
# Load the environment variables
module purge
module load GCCcore/11.3.0
module load Python/3.10.4

source /data/gpfs/projects/punim0111/CORGI_TERRIER/rob/vanjarint/.venv/bin/activate

export PATH=/data/gpfs/projects/punim2199/poetry-env/bin:$PATH
export HF_HOME=/home/rturnbull/wytamma/huggingface

GROWTH=2
LAYERS=2
EMBEDDING=0
LR=0.0001
EPOCHS=20
BATCH=16
FEATURES=768
FEATURES=1024
#FEATURES=256
ESM_LAYERS=6
DROPOUT=0.4
#FEATURES=2048


#MODEL_NAME=NT-500m
#MODEL_NAME=NT
#MODEL_NAME=checkpoint-66000

LENGTH=1000
MODEL_NAME=checkpoint121000l${LENGTH}
#MODEL_NAME=checkpoint66000l${LENGTH}


PREPROCESSED_DIR=/data/gpfs/projects/punim0111/CORGI_TERRIER/rob/${MODEL_NAME}
MEMMAP=${PREPROCESSED_DIR}/${MODEL_NAME}.npy
MEMMAP_INDEX=${PREPROCESSED_DIR}/${MODEL_NAME}.txt

MEMMAP=/tmp/NT.npy
if [[ ! -s "$MEMMAP" ]]; then
        cp ${PREPROCESSED_DIR}/${MODEL_NAME}.npy $MEMMAP
fi

SEQTREE=${PREPROCESSED_DIR}/${MODEL_NAME}.st


vanjarint-tools train  \
        --memmap  $MEMMAP \
        --memmap-index  ${MEMMAP_INDEX} \
        --seqtree  $SEQTREE \
        --features $FEATURES \
        --intermediate-layers $LAYERS \
        --growth-factor $GROWTH \
        --dropout $DROPOUT \
        --family-embedding-size $EMBEDDING \
        --max-epochs $EPOCHS \
        --max-learning-rate $LR \
        --embedding-model ESM${ESM_LAYERS} \
        --num-workers 2 \
        --run-name VanjariNT-b${BATCH}f${FEATURES}l${LAYERS}g${GROWTH}e${EMBEDDING}lr${LR}ep${EPOCHS}d${DROPOUT}-${MODEL_NAME} \
        --project-name VanjariNT \
        --wandb-entity   "mariadelmarq-The University of Melbourne" \
        --wandb 

