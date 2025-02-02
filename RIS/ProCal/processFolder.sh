#!/bin/bash

# Read command line arguments
DATA_SET=$1
DATA_FOLDER=$2


# Find .raw files and execute pipeline.sh
for f in "$DATA_FOLDER/"*
do
    if [[ ${f,,} == *.raw ]]; then
        base_name=$(basename "${f}")
        echo "Executing pipeline on $base_name"

        ./pipeline.sh $DATA_SET $f 
    fi
done

#bsub < /scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/ProCal/Fig2c_170330_PROST_200fmol_4to42_LUMOS_CE_RAMP/scripts/merge_msp.bsub

#bsub < /scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/ProCal/Fig2c_170330_PROST_200fmol_4to42_LUMOS_CE_RAMP/scripts/extract_traces.bsub

#bsub < /scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/ProCal/Fig2c_170330_PROST_200fmol_4to42_LUMOS_CE_RAMP/scripts/frag_splines.bsub

#bsub < /scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/ProCal/Fig2c_170330_PROST_200fmol_4to42_LUMOS_CE_RAMP/scripts/calibrate_lumos.bsub

#bsub < /scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/ProCal/Fig2c_170330_PROST_200fmol_4to42_LUMOS_CE_RAMP/scripts/lumos2qe_poly.bsub