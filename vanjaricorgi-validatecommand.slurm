#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=vanjaricorgi
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
# SBATCH -t 7-0:00:00
#SBATCH -t 1-0:00:00
 
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
SEQTREE=/home/rturnbull/wytamma/rob/virus/ictv-validation.st

LEARNING_RATE=0.001
MACC=100000000000
EPOCHS=100
MAX_LENGTH=5000
BATCH_SIZE=32
VAL_LENGTH=2000

RUN_NAME=vanjari-lr${LEARNING_RATE}m${MACC}e${EPOCHS}max${MAX_LENGTH}val${VAL_LENGTH}b${BATCH_SIZE}
CHECKPOINT=/data/gpfs/projects/punim0111/CORGI_TERRIER/rob/vanjarint/logs/vanjari-lr0.001m100000000000e100max5000val2000b32/version_1/checkpoints/checkpoint-epoch\=90-Genus\=0.51.ckpt


vanjari-tools validate         \
    --treedict $SEQTREE         \
    --seqbank $SEQBANK         \
    --phi 0.9869985869163924 \
    --checkpoint $CHECKPOINT \
    --max-epochs $EPOCHS \
    --run-name $RUN_NAME \
    --test-partition -1 \
    --max-items 20 \
    --batch-size $BATCH_SIZE \
    --validation-partition 0 \
    --project-name ICTV-val \
    --validation-length $VAL_LENGTH \
    --maximum-length $MAX_LENGTH 
