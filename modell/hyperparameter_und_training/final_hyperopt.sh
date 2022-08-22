#!/bin/bash
#PBS -l walltime=46:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=10GB:accelerator_model=gtx1080ti
#PBS -A "TransportPredict"
#PBS -N final_hyperopt
#PBS -j oe
#PBS -o "hyperopt_output.log"
#PBS -r y
#PBS -m ae
#PBS -M ninie104@hhu.de

module load Python/3.8.3

python final_hyperopt.py