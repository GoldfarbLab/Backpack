import os
import sys
import msp


pep_inpath = sys.argv[1]
msp_path = sys.argv[2]
out_path = sys.argv[3]

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


for i, scan in enumerate(msp.read_msp_file(msp_path)):
    if i > 0 and i % 10000 == 0: print(i)
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