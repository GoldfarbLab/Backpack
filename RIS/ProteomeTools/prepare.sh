#!/bin/bash

# Read command line arguments
DATA_SET=$1
SAGE_CONFIG_PATH=$2
DATA_PATH=$3
DATA_NAME=$4
POOL_NAME=$5

# Create config file from template and command line arguments
sed "s+DATA_SET_NAME=+DATA_SET_NAME=$DATA_SET+g" < config_template.sh > ${DATA_NAME}_config.sh
sed -i "s+SAGE_CONFIG_PATH=+SAGE_CONFIG_PATH=$SAGE_CONFIG_PATH/+g" ${DATA_NAME}_config.sh
sed -i "s+DATA_PATH=+DATA_PATH=$DATA_PATH/+g" ${DATA_NAME}_config.sh
sed -i "s+DATA_NAME=+DATA_NAME=$DATA_NAME/+g" ${DATA_NAME}_config.sh
sed -i "s+POOL_NAME=+POOL_NAME=$POOL_NAME/+g" ${DATA_NAME}_config.sh

# Read config
source ./${DATA_NAME}_config.sh

# Create output folders
mkdir -p $OUT_PATH
mkdir -p $RESULTS_PATH
mkdir -p $LOG_PATH
mkdir -p $SCRIPT_PATH

# Move config to output scripts folder
mv ./${DATA_NAME}_config.sh $SCRIPT_PATH/config.sh

# Copy python scripts
#cp *.py $SCRIPT_PATH/