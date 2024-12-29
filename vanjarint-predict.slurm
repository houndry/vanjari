#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=vanjari-predict
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
# SBATCH -t 5:00:00
#SBATCH -t 3-0:00:00
 
#SBATCH --mem-per-cpu=12G
 
# SBATCH -p cascade
#SBATCH -p gpu-h100
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
module load Graphviz/5.0.0

source /data/gpfs/projects/punim0111/CORGI_TERRIER/rob/vanjarint/.venv/bin/activate

export PATH=/data/gpfs/projects/punim2199/poetry-env/bin:$PATH
export HF_HOME=/home/rturnbull/wytamma/huggingface


EMBEDDING_MODEL=/data/gpfs/projects/punim0111/CORGI_TERRIER/checkpoint-182000/

CHECKPOINT=/data/gpfs/projects/punim0111/CORGI_TERRIER/rob/vanjarint/logs/VanjariNT-b16f1024l2g2e0lr0.0001ep20d0.4-checkpoint121000l1000/version_0/checkpoints/checkpoint-epoch\=17-species_accuracy\=0.21.ckpt


FASTA_DIR=/data/gpfs/projects/punim0111/CORGI_TERRIER/rob/ICTV-TaxonomyChallenge/dataset/dataset_challenge
FASTA_DIR=/data/gpfs/projects/punim0111/CORGI_TERRIER/rob/ICTV-TaxonomyChallenge/dataset/dataset_challenge_tiny
LENGTH=1000
#LENGTH=500
vanjarint-tools predict \
    --input ${FASTA_DIR} \
    --model-name $EMBEDDING_MODEL \
    --checkpoint $CHECKPOINT \
    --memmap-array-path ${FASTA_DIR}/embeddings/checkpoint-182000-l${LENGTH}.npy \
    --memmap-index ${FASTA_DIR}/embeddings/checkpoint-182000-l${LENGTH}.txt \
    --length $LENGTH \
    --output-csv ${FASTA_DIR}/predictions/VanjariNT-b16f1024l2g2e0lr0.0001ep20d0.4-checkpoint121000l1000.csv \
    --probability-csv ${FASTA_DIR}/predictions/VanjariNT-b16f1024l2g2e0lr0.0001ep20d0.4-checkpoint121000l1000.probability.csv \
    --image-dir ${FASTA_DIR}/predictions/VanjariNT-b16f1024l2g2e0lr0.0001ep20d0.4-checkpoint121000l1000-images

