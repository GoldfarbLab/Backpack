#!/bin/bash

# Function to extract job ID from console output
function get_jobid {
    output=$($*)
    echo $output | tail -n 1 | cut -d'<' -f2 | cut -d'>' -f1
}

function get_pool {
    POOL_FULL=`grep -o -P '\-.*?\-' <<< $1 | head -1`
    POOL=${POOL_FULL:1:-7}
    echo $POOL
}

# Read config template to get initial paths
source config_template.sh

# Read command line arguments
DATA_SET=$1
SAGE_CONFIG_PATH=$2
DATA_PATH=$3
DATA_NAME=$(basename -- "$DATA_PATH")
DATA_NAME="${DATA_NAME%.*}"
POOL_NAME=$(get_pool ${DATA_NAME})


# Execute scripts with command line arguments
./prepare.sh $DATA_SET $SAGE_CONFIG_PATH $DATA_PATH $DATA_NAME $POOL_NAME

# Read final config
source ${SCRATCH_PATH}/ProteomeTools/${DATA_SET}/${DATA_NAME}/scripts/config.sh

# Submit LSF jobs
jid1=$(get_jobid bsub < $SCRIPT_PATH/convert_raw.bsub)
echo Submitted convert job for $DATA_NAME with ID: $jid1

#jid2=$(get_jobid bsub -w "ended($jid1)" < $SCRIPT_PATH/split.bsub)
#echo Submitted split job for $DATA_SET_NAME with ID: $jid2