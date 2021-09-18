#!/bin/bash
#SBATCH --account=rpp-kshook
#SBATCH --time=01:00:00
#SBATCH --mem=250MB
#SBATCH --job-name=xxxxxx
#SBATCH --output=%x-%j.out

mkdir -p /home/h294liu/scratch/temp
export TMPDIR=/home/h294liu/scratch/temp
export MPI_SHEPHERD=true

#### Note: This bash file must be put in the same directory as Ishigami test.

nSample=NSAMPLE
python Ishigami_test.py $nSample 

