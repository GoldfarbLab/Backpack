#!/bin/bash

# Function to extract job ID from console output
function get_jobid {
    output=$($*)
    echo $output | tail -n 1 | cut -d'<' -f2 | cut -d'>' -f1
}

# submit pipeline1 on each dataset (convert, sage, chronologer, annotate)
./processFolder.sh Part1 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part1/ pipeline
#./processFolder.sh Part2 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part2/ pipeline
#./processFolder.sh Part3 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config_aspN.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part3/aspn/ pipeline
#./processFolder.sh Part3 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config_lysN.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part3/lysn/ pipeline
#./processFolder.sh Part3 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part3/HLA/ pipeline


# submit ion dictionary merge job
#jid_dict=$(get_jobid bsub < /scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/Part1/01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1/scripts/merge_ion_dict.bsub)
#echo Submitted merge_ion_dict job with ID: $jid_dict

# submit pipeline2 on each dataset (filter, quad, deisotope, NCE_align)
#./processFolder.sh Part1 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part1/ pipeline2
#./processFolder.sh Part2 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part2/ pipeline2
#./processFolder.sh Part3 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config_aspN.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part3/aspn/ pipeline2
#./processFolder.sh Part3 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config_lysN.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part3/lysn/ pipeline2
#./processFolder.sh Part3 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part3/HLA/ pipeline2

# Quad fit
#jid_merge=$(get_jobid bsub < /scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/Part1/01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1/scripts/merge_quad.bsub)
#echo Submitted merge_quad job with ID: $jid_merge


# submit consolidate NCEs job
#jid_NCE=$(get_jobid bsub < /scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/Part1/01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1/scripts/consolidate_NCEs.bsub)
#echo Submitted consolidate_NCEs job for Part1 with ID: $jid_NCE
#jid_NCE=$(get_jobid bsub < /scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/Part2/01974c_BA1-TUM_missing_first_1_01_01-2xIT_2xHCD-1h-R4/scripts/consolidate_NCEs.bsub)
#echo Submitted consolidate_NCEs job for Part2 with ID: $jid_NCE
#jid_NCE=$(get_jobid bsub < /scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/Part3/02445a_BA10-TUM_HLA_10_01_01-DDA-1h-R1/scripts/consolidate_NCEs.bsub)
#echo Submitted consolidate_NCEs job for Part3 with ID: $jid_NCE


# submit pipeline3 on each dataset (update NCEs, split_indiv)
#./processFolder.sh Part1 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part1/ pipeline3
#./processFolder.sh Part2 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part2/ pipeline3
#./processFolder.sh Part3 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config_aspN.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part3/aspn/ pipeline3
#./processFolder.sh Part3 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config_lysN.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part3/lysn/ pipeline3
#./processFolder.sh Part3 /storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/closed_config.json /storage1/fs1/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part3/HLA/ pipeline3


# submit merge job (train, val, test)
#jid_merge=$(get_jobid bsub < /scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/Part1/01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1/scripts/merge_xv.bsub)
#echo Submitted merge_xv job with ID: $jid_merge

# submit create AI data job
#jid_ai=$(get_jobid bsub < /scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/Part1/01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1/scripts/create_aidata.bsub)
#echo Submitted ai data job with ID: $jid_ai

# submit model training job
jid_train=$(get_jobid bsub < /scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/Part1/01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1/scripts/train.bsub)
echo Submitted train job with ID: $jid_train

# submit model training array job
#jid_train=$(get_jobid bsub < /scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/Part1/01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1/scripts/train_array.bsub)
#echo Submitted train array job with ID: $jid_train

# submit onnx export job
#jid=$(get_jobid bsub < /scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/Part1/01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1/scripts/export2onnx.bsub)
#echo Submitted onnx job with ID: $jid
