#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=16GB:accelerator_model=rtx8000
#PBS -A kcat-prediction
#PBS -N extract_ESM_enzyme_rep_KM
#PBS -j oe
#PBS -o "extract_ESM_enzyme_rep_KM.log"
#PBS -r y
#PBS -m ae
#PBS -M alkro105@hhu.de


#cd /home/alkro105/enzym_rep/ESM

module load Python/3.6.5
module load CUDA/11.0.2
module load cuDNN/8.0.4

python3 /home/alkro105/enzym_rep/ESM/extract_esm1b.py esm1b_t33_650M_UR50S /gpfs/project/alkro105/all_transporter_sequences.fasta /gpfs/project/alkro105/all_transporter_sequences --repr_layers 33 --include mean
python3 /home/alkro105/enzym_rep/ESM/merge_pt_files.py