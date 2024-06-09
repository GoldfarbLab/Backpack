#!/bin/bash

# Read command line arguments
DATA_FOLDER=$1
CONFIG_PATH=$2

# Find .raw files and execute pipeline.sh
for f in "$DATA_FOLDER/"*
do
    if [[ ${f,,} == *.raw ]]; then
        base_name=$(basename "${f}")
        echo "Executing pipeline on $base_name"

        ./prepare.sh $DATA_FOLDER $CONFIG_PATH
    fi
done