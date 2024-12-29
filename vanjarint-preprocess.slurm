#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=vanjari-preprocess500
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
# SBATCH -t 7-0:00:00
#SBATCH -t 4-0:00:00
 
#SBATCH --mem-per-cpu=12G
 
# SBATCH -p cascade
# SBATCH -p gpu-h100,gpu-a100
#SBATCH -p gpu-h100
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

LENGTH=500
#LENGTH=1000

MODEL_NAME=checkpoint182000l${LENGTH}

SEQTREE=../$MODEL_NAME/${MODEL_NAME}.st

#CHECKPOINT=/data/gpfs/projects/punim0111/CORGI_TERRIER/nt/
CHECKPOINT=/data/gpfs/projects/punim0111/CORGI_TERRIER/checkpoint-182000/

vanjarint-tools preprocess         \
    --seqtree $SEQTREE         \
    --output-dir ../$MODEL_NAME \
    --fasta-dir ../all_fastas/ \
    --length $LENGTH \
    --model-name $CHECKPOINT

