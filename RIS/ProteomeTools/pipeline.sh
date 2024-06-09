#!/bin/bash

# Function to extract job ID from console output
function get_jobid {
    output=$($*)
    echo $output | tail -n 1 | cut -d'<' -f2 | cut -d'>' -f1
}

# Read config template to get initial paths
source config_template.sh

# Read command line arguments
DATA_PATH=$1


# Execute scripts with command line arguments
./prepare.sh $DATA_SET_NAME $DATA_PATH

# Read final config
source ${SCRATCH_PATH}/${DATA_SET_NAME}/total_RNA/scripts/config.sh

# Submit LSF jobs
jid1=$(get_jobid bsub < $SCRIPT_PATH/align.bsub)
echo Submitted align job for $DATA_SET_NAME with ID: $jid1

jid2=$(get_jobid bsub -w "ended($jid1)" < $SCRIPT_PATH/split.bsub)
echo Submitted split job for $DATA_SET_NAME with ID: $jid2

jid3=$(get_jobid bsub -w "ended($jid2)" < $SCRIPT_PATH/collapse.bsub)
echo Submitted collapse job for $DATA_SET_NAME with ID: $jid3

jid4=$(get_jobid bsub -w "ended($jid3)" < $SCRIPT_PATH/index.bsub)
echo Submitted index job for $DATA_SET_NAME with ID: $jid4

jid5=$(get_jobid bsub -w "ended($jid4)" < $SCRIPT_PATH/translate.bsub)
echo Submitted translate job for $DATA_SET_NAME with ID: $jid5

jid6=$(get_jobid bsub -w "ended($jid5)" < $SCRIPT_PATH/trim.bsub)
echo Submitted trim job for $DATA_SET_NAME with ID: $jid6

jid7=$(get_jobid bsub -w "ended($jid6)" < $SCRIPT_PATH/domains.bsub)
echo Submitted domains job for $DATA_SET_NAME with ID: $jid7