#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=vanjara
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
# SBATCH -t 7-0:00:00
#SBATCH -t 3-0:00:00
 
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

SEQBANK=/home/rturnbull/wytamma/rob/virus/ictv.sb
SEQTREE=/home/rturnbull/wytamma/rob/virus/ictv.st

LEARNING_RATE=0.001
MACC=100000000000
EPOCHS=100
RUN_NAME=vanjari-lr${LEARNING_RATE}m${MACC}e${EPOCHS}
#OUTPUT_DIR=outputs/$RUN_NAME
vanjara-tools train         \
    --seqtree $SEQTREE         \
    --seqbank $SEQBANK         \
    --max-learning-rate $LEARNING_RATE         \
    --macc $MACC         \
    --max-epochs $EPOCHS \
    --run-name $RUN_NAME \
    --test-partition 0 \
    --validation-partition 1 \
    --project-name ICTV \
    --wandb-entity "mariadelmarq-The University of Melbourne" \
    --wandb

