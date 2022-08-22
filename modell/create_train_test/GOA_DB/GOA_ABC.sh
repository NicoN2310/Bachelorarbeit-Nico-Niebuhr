#!/bin/bash
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=1:mem=1GB
#PBS -A kcat-prediction
#PBS -N GOA_ABC
#PBS -j oe
#PBS -o "GOA_ABC.log"
#PBS -r y
#PBS -m ae
#PBS -M alkro105@hhu.de
#PBS -J 0-1

#cd $PBS_O_WORKDIR

module load Python/3.8.3

python3 GOA_ABC.py $PBS_ARRAY_INDEX