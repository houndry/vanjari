#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=vanjara
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-20%1
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
# SBATCH -t 7-0:00:00
#SBATCH -t 2-0:00:00
 
#SBATCH --mem-per-cpu=12G
 
# SBATCH -p cascade
#SBATCH -p gpu-a100,gpu-h100
# SBATCH -p gpu-a100-short,gpu-a100,gpu-a100-preempt
# SBATCH -p gpu-a100-short,gpu-a100
# SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --account=punim0111
 
# Specify your email address to be notified of code progress.
#SBATCH --mail-user=robert.turnbull@unimelb.edu.au
#SBATCH --mail-type=ALL
 
# Load the environment variables
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
PATH=/data/gpfs/projects/punim0111/CORGI_TERRIER/poetry/bin:$PATH


source .venv/bin/activate

SEQBANK=/tmp/ictv.sb
cp -r /home/rturnbull/wytamma/rob/virus/ictv.sb $SEQBANK
TREEDICT=/home/rturnbull/wytamma/rob/virus/ictv.td

MACC=100000000000
EPOCHS=100
MAX_LENGTH=5000
BATCH_SIZE=32
VAL_LENGTH=2000
RUN_NAME=vanjari-tune-m${MACC}e${EPOCHS}max${MAX_LENGTH}val${VAL_LENGTH}b${BATCH_SIZE}

vanjari-tools tune         \
    --engine optuna \
    --id $RUN_NAME \
    --treedict $TREEDICT         \
    --seqbank $SEQBANK         \
    --macc $MACC         \
    --max-epochs $EPOCHS \
    --test-partition -1 \
    --batch-size $BATCH_SIZE \
    --validation-partition 0 \
    --project-name ICTV \
    --validation-length $VAL_LENGTH \
    --maximum-length $MAX_LENGTH \
    --wandb-entity "mariadelmarq-The University of Melbourne" \
    --wandb

