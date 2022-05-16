#!/bin/bash
#SBATCH --job-name=roberta_
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yz5880/301/%j_%x.out
#SBATCH --error=/scratch/yz5880/301/%j_%x.err
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --mem=20G
#SBATCH -c 4
#SBATCH --mail-user=<yz5880@nyu.edu>
#SBATCH --mail-type=ALL

## TODO: Run your hyperparameter search in a singularity container. Remember to 
## activate your Conda environment in the container before running your Python script.
singularity exec --nv --bind /scratch/yz5880 --overlay /scratch/yz5880/overlay-25GB-500K.ext3:ro /scratch/yz5880/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python /scratch/yz5880/301/bert_bayes.py BoolQ/
"
