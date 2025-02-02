import os
import sys
import pandas as pd
import msp

NCE_align_data_path = sys.argv[1]
pep_inpath = sys.argv[2]
msp_path = sys.argv[3]
out_path = sys.argv[4]

NCE_align_data = pd.read_csv(NCE_align_data_path, sep="\t")

train_outfile = open(os.path.join(out_path, "pep.train.msp"), "w")
val_outfile = open(os.path.join(out_path, "pep.val.msp"), "w")
test_outfile = open(os.path.join(out_path, "pep.test.msp"), "w")

#train_peptides = set()
#with open(pep_inpath + "pep.train.msp", 'r') as infile:
#    for line in infile:
#        train_peptides.add(line.strip())
    
val_peptides = set()
with open(os.path.join(pep_inpath, "pep.val2.msp"), 'r') as infile:
    for line in infile:
        val_peptides.add(line.strip())

test_peptides = set()
with open(os.path.join(pep_inpath, "pep.test2.msp"), 'r') as infile:
    for line in infile:
        test_peptides.add(line.strip())

filename = os.path.basename(msp_path).split(".")[0]
sub_table = NCE_align_data[NCE_align_data.file == filename]

for i, scan in enumerate(msp.read_msp_file(msp_path)):
    if i > 0 and i % 10000 == 0: print(i)
    
    correction_factor = sub_table[sub_table.NCE == scan.metaData.NCE].offset_aligned.iloc[0]
    scan.metaData.key2val["NCE_aligned"] = scan.metaData.NCE + correction_factor
    scan.updateMSPName()
    
    pep = scan.peptide.toUnmodifiedString()
    pep = pep.replace("I","L")
    if len(pep) > 30 or len(pep) < 7:
        scan.writeScan(train_outfile, int_prec=5)
    elif pep in test_peptides:
        scan.writeScan(test_outfile, int_prec=5)
    elif pep in val_peptides:
        scan.writeScan(val_outfile, int_prec=5)
    else:
        scan.writeScan(train_outfile, int_prec=5)