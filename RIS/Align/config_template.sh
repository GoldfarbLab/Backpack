#!/bin/bash

# parameters automatically filled in
DATA_SET_NAME=
DATA_PATH=
SAGE_CONFIG_PATH=

# setup paths
PROJECT_PATH=/storage1/fs1/d.goldfarb/Active/Projects/Backpack/
REF_PATH=/storage1/fs1/d.goldfarb/Active/Backpack/
FASTA_PATH=${REF_PATH}/fasta
SRC_PATH=${PROJECT_PATH}
SCRATCH_PATH=/scratch1/fs1/d.goldfarb/Backpack/

# output paths
DATA_SET_OUT_PATH=${SCRATCH_PATH}/Align/${DATA_SET_NAME}
OUT_PATH=${DATA_SET_OUT_PATH}/${DATA_NAME}
RESULTS_PATH=${OUT_PATH}/results
LOG_PATH=${OUT_PATH}/logs
SCRIPT_PATH=${OUT_PATH}/scripts

# reference files
HUMAN_FASTA=${FASTA_PATH}/human-2024-06-04.fas
MOUSE_FASTA=${FASTA_PATH}/mouse-2024-06-04.fas
YEAST_FASTA=${FASTA_PATH}/yeast-2024-06-04.fas

export PYTHONPATH=${SRC_PATH}/python/