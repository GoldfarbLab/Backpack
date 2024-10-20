#!/bin/bash

# Read command line arguments
DATA_SET=$1
CONFIG_PATH=$2
DATA_FOLDER=$3
PIPELINE=$4

# Find .raw files and execute pipeline.sh
for f in "$DATA_FOLDER/"*
do
    if [[ ${f,,} == *.raw ]]; then
        base_name=$(basename "${f}")
        if [[ $base_name != *"ETD"* ]]; then
            echo "Executing $PIPELINE on $base_name"
            ./${PIPELINE}.sh $DATA_SET $CONFIG_PATH $f 
            exit
        fi
    fi
done