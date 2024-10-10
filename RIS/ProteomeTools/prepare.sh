#!/bin/bash

# Read command line arguments
DATA_SET=$1
SAGE_CONFIG_PATH=$2
DATA_PATH=$3
DATA_NAME=$4
POOL_NAME=$5
CONFIG_FILE=${DATA_NAME}_config.sh

# Create config file from template and command line arguments
sed "s+DATA_SET_NAME=+DATA_SET_NAME=$DATA_SET+g" < config_template.sh > $CONFIG_FILE
sed -i "s+SAGE_CONFIG_PATH=+SAGE_CONFIG_PATH=$SAGE_CONFIG_PATH+g" $CONFIG_FILE
sed -i "s+DATA_PATH=+DATA_PATH=$DATA_PATH+g" $CONFIG_FILE
sed -i "s+DATA_NAME=+DATA_NAME=$DATA_NAME+g" $CONFIG_FILE
sed -i "s+POOL_NAME=+POOL_NAME=$POOL_NAME+g" $CONFIG_FILE

# Read config
source ./$CONFIG_FILE

# Create output folders
mkdir -p $OUT_PATH
mkdir -p $RESULTS_PATH
mkdir -p $LOG_PATH
mkdir -p $SCRIPT_PATH

# Move config to output scripts folder
mv ./$CONFIG_FILE $SCRIPT_PATH/config.sh

# Update scripts with dataset specific parameters
declare -a scripts=("convert_raw.bsub" "sage.bsub" "annotate.bsub" "deisotope.bsub" "NCE_align.bsub" "consolidate_NCEs.bsub" "pipeline2.sh" "update_NCEs.bsub" "chronologer.bsub" "split_indiv.bsub" "merge_ion_dict.bsub" "filter.bsub" "merge_xv.bsub" "create_aidata.bsub" "train.bsub")

for val in ${scripts[@]}; do
    sed "s+LSF_SCRIPT_PATH+$SCRIPT_PATH/+g" < $val > $SCRIPT_PATH/$val
done

#chmod 775 $SCRIPT_PATH/pipeline2.sh

# Copy python scripts
#cp *.py $SCRIPT_PATH/