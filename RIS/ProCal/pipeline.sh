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

#jid3=$(get_jobid bsub < $SCRIPT_PATH/annotate.bsub)
#echo Submitted annotate job for $DATA_NAME with ID: $jid3


#jid4=$(get_jobid bsub -w "ended($jid3)" < $SCRIPT_PATH/filter.bsub)
#jid4=$(get_jobid bsub < $SCRIPT_PATH/filter.bsub)
#echo Submitted filter job for $DATA_NAME with ID: $jid4

#jid=$(get_jobid bsub -w "ended($jid3)" < $SCRIPT_PATH/quad_fit.bsub)
#jid=$(get_jobid bsub < $SCRIPT_PATH/quad_fit.bsub)
#echo Submitted quad_fit job for $DATA_NAME with ID: $jid


#jid5=$(get_jobid bsub -w "ended($jid4)" < $SCRIPT_PATH/deisotope.bsub)
#jid5=$(get_jobid bsub < $SCRIPT_PATH/deisotope.bsub)
#echo Submitted deisotope job for $DATA_NAME with ID: $jid5
