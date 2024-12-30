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


CHECKPOINT=/data/gpfs/projects/punim0111/CORGI_TERRIER/rob/vanjarint/logs/vanjari-lr0.001m100000000000e100max5000val2000b32/version_1/checkpoints/checkpoint-epoch\=90-Genus\=0.51.ckpt

FASTA_DIR=/data/gpfs/projects/punim0111/CORGI_TERRIER/rob/ICTV-TaxonomyChallenge/dataset/dataset_challenge_tiny
FASTA_DIR=/data/gpfs/projects/punim0111/CORGI_TERRIER/rob/validation_fastas

vanjaricorgi-tools predict \
    --file ${FASTA_DIR} \
    --checkpoint $CHECKPOINT \
    --output-csv ${FASTA_DIR}/predictions/vanjari-lr0.001m100000000000e100max5000val2000b3.csv \
    --prediction-threshold 0.0


#    --image-dir ${FASTA_DIR}/predictions/vanjari-lr0.001m100000000000e100max5000val2000b3-images    
#    --probability-csv ${FASTA_DIR}/predictions/vanjari-lr0.001m100000000000e100max5000val2000b3.probability.csv \


