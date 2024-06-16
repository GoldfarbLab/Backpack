#!/bin/bash

# Function to extract job ID from console output
function get_jobid {
    output=$($*)
    echo $output | tail -n 1 | cut -d'<' -f2 | cut -d'>' -f1
}

# Read config template to get initial paths
source config_template.sh

# Read command line arguments
DATA_SET=$1
DATA_PATH=$2
DATA_NAME=$(basename -- "$DATA_PATH")
DATA_NAME="${DATA_NAME%.*}"


# Execute scripts with command line arguments
./prepare.sh $DATA_SET $DATA_PATH $DATA_NAME

# Read final config
source ${SCRATCH_PATH}/ProteomeTools/${DATA_SET}/${DATA_NAME}/scripts/config.sh

# Submit LSF jobs
#jid1=$(get_jobid bsub < $SCRIPT_PATH/convert_raw.bsub)
#echo Submitted convert job for $DATA_NAME with ID: $jid1

#jid2=$(get_jobid bsub -w "ended($jid1)" < $SCRIPT_PATH/sage.bsub)
#echo Submitted sage job for $DATA_NAME with ID: $jid2

#jid3=$(get_jobid bsub -w "ended($jid2)" < $SCRIPT_PATH/filter.bsub)
jid3=$(get_jobid bsub < $SCRIPT_PATH/filter.bsub)
echo Submitted filter job for $DATA_NAME with ID: $jid3