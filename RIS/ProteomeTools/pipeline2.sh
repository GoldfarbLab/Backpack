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
DATA_PATH=$3
DATA_NAME=$(basename -- "$DATA_PATH")
DATA_NAME="${DATA_NAME%.*}"

# Read final config
source ${SCRATCH_PATH}/ProteomeTools/${DATA_SET}/${DATA_NAME}/scripts/config.sh

#jid4=$(get_jobid bsub -w "ended($jid3)" < $SCRIPT_PATH/filter.bsub)
jid4=$(get_jobid bsub < $SCRIPT_PATH/filter.bsub)
echo Submitted filter job for $DATA_NAME with ID: $jid4

#jid5=$(get_jobid bsub -w "ended($jid4)" < $SCRIPT_PATH/deisotope.bsub)
#jid5=$(get_jobid bsub < $SCRIPT_PATH/deisotope.bsub)
#echo Submitted deisotope job for $DATA_NAME with ID: $jid5

#jid6=$(get_jobid bsub -w "ended($jid5)" < $SCRIPT_PATH/NCE_align.bsub)
#jid6=$(get_jobid bsub < $SCRIPT_PATH/NCE_align.bsub)
#echo Submitted NCE alignment job for $DATA_NAME with ID: $jid6
