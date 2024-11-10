import os
import sys
import msp
import random


#infile = sys.argv[1]
headers = sys.argv[1]
out_path = sys.argv[2]

train_outfile = open(os.path.join(out_path, "pep.train2.msp"), "w")
val_outfile = open(os.path.join(out_path, "pep.val2.msp"), "w")
test_outfile = open(os.path.join(out_path, "pep.test2.msp"), "w")

procal = set(["GIFGAFTDDYK","GFVIDDGLITK","GFLDYESTGAK","GDFTFFIDTFK","GASDFLSFAVK","FVGTEYDGLAK",
	"FLFTGYDTSVK","FLASSEGGFTK","FGTGTYAGGEK","FFLTGTSIFVK","ASDLLSGYYIK","ALFSSITDSEK","YSAHEEHHYDK","YFGYTSDTFGK","YALDSYSLSSK","VYAETLSGFIK",
	"VSSIFFDTFDK","VSGFSDISIYK","VGASTGYSGLK","TSIDSFIDSYK","TLIAYDDSSTK","TFTGTTDSFFK","TFGTETFDTFK","TFAHTESHISK","TASGVGGFSTK","SYASDFGSSAK",
	"SLFFIIDGFVK","SILAFLYLYFK","LYTGAGYDEVK","LYSYYSSTESK","LSSGYDGTSYK","ISLGEHEGGGK","HLTGLTFDTYK","HFALFSTDVTK","HEHISSDYAGK","HDTVFGSYLYK"])

peptides = set()
with open(headers, 'r') as header_file:
    for line in header_file:
        pep = line.split(" ")[1].split("/")[0]
        pep = pep.replace("I", "L")
        peptides.add(pep)
    

peptides = list(peptides)
random.shuffle(peptides)

train_peptides = peptides[0:int(len(peptides)*0.7)]
val_peptides = peptides[len(train_peptides):(len(train_peptides)+int(len(peptides)*0.2))] 
test_peptides = peptides[(len(val_peptides)+len(train_peptides)):]

for pep in train_peptides:
    train_outfile.write(pep + "\n")
    
for pep in val_peptides:
    val_outfile.write(pep + "\n")
    
for pep in test_peptides:
    test_outfile.write(pep + "\n")


sys.exit()
for i, scan in enumerate(msp.read_msp_file(infile)):
    if i > 0 and i % 10000 == 0: print(i)
    pep = scan.peptide.toUnmodifiedString()
    if len(pep) > 30 or pep in train_peptides or pep in procal:
        scan.writeScan(train_outfile, int_prec=5)
    elif pep in val_peptides:
        scan.writeScan(val_outfile, int_prec=5)
    else:
        scan.writeScan(test_outfile, int_prec=5)