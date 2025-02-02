#!/bin/bash

# Function to extract job ID from console output
function get_jobid {
    output=$($*)
    echo $output | tail -n 1 | cut -d'<' -f2 | cut -d'>' -f1
}

# submit pipeline_eval on each dataset (split)
#./processFolder.sh Part1 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part1/ pipeline_eval
#./processFolder.sh Part2 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part2/ pipeline_eval
#./processFolder.sh Part3 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config_aspN.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part3/aspn/ pipeline_eval
#./processFolder.sh Part3 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config_lysN.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part3/lysn/ pipeline_eval
#./processFolder.sh Part3 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part3/HLA/ pipeline_eval

# submit merge job (train, val, test)
jid_merge=$(get_jobid bsub < /scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/Part1/01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1/scripts/merge_xv_raw.bsub)
echo Submitted merge_xv_raw job with ID: $jid_merge

