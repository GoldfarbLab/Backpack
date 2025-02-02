import os
import sys
import pandas as pd
import msp
import numpy as np


frames = []
NCE_align_data_path = sys.argv[1]
msp_path = sys.argv[2]
out_path = sys.argv[3]


NCE_align_data = pd.read_csv(NCE_align_data_path, sep="\t")


filename = os.path.basename(msp_path).split(".")[0]

if os.path.exists(msp_path):
    with open(os.path.join(out_path, filename + ".msp"), "w") as outfile:
        sub_table = NCE_align_data[NCE_align_data.file == filename]
        
        # prosit
        #correction_factor = sub_table.offset_aligned.iloc[0]
        for scan in msp.read_msp_file(msp_path):
            correction_factor = sub_table[sub_table.NCE == scan.metaData.NCE].offset_aligned.iloc[0]
            
            # Raw offset
            #correction_factor = sub_table[sub_table.NCE == scan.metaData.NCE].offset.iloc[0]
            #if np.isnan(correction_factor):
                #print("Correction NA")
            #    correction_factor = sub_table[sub_table.NCE == scan.metaData.NCE].offset_aligned.iloc[0]

            scan.metaData.key2val["NCE_aligned"] = scan.metaData.NCE + correction_factor
            scan.updateMSPName()
            scan.writeScan(outfile, int_prec=5)
