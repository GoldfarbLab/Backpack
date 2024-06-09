#!/bin/bash

# Read command line arguments
DATA_SET=$1
CONFIG_PATH=$2
DATA_FOLDER=$3


# Find .raw files and execute pipeline.sh
for f in "$DATA_FOLDER/"*
do
    if [[ ${f,,} == *.raw ]]; then
        base_name=$(basename "${f}")
        echo "Executing pipeline on $base_name"

        ./pipeline.sh $DATA_SET $CONFIG_PATH $f 
    fi
done