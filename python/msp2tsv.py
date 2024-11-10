import os
import sys
import msp
import csv
import yaml
import numpy as np

with open(os.path.join(os.path.dirname(__file__), "../config/data.yaml"), 'r') as stream:
    config = yaml.safe_load(stream)


# open file for each letter
interp_path = os.path.join(config['base_path'], "interp_tsv")
os.makedirs(interp_path, exist_ok=True)

aa2csv_file = dict()
aa2csv_writer = dict()
for aa in "ACDEFGHIKLMNPQRSTVWY":
    aa2csv_file[aa] = open(os.path.join(interp_path, aa + ".tsv"), 'w', newline="", encoding="utf-8")
    aa2csv_writer[aa] = csv.writer(aa2csv_file[aa], delimiter="\t")
    aa2csv_writer[aa].writerow(["seq", "mods", "z", "weight", "NCE", "frag", "intensity"])

for scan_i, scan in enumerate(msp.read_msp_file(config['train_files'])):
    
    pep = scan.peptide.toUnmodifiedString()
    pep_start = pep[0]
    mod_string = scan.getModString()
    z = str(scan.metaData.z)
    nce = str(scan.metaData.key2val['NCE_aligned'])
    weight = str(np.sqrt(scan.metaData.purity * scan.metaData.rawOvFtT))
    scan_id = scan.metaData.scanID
    
    scan.normalizeToTotalAnnotated(doLOD = True)
    
    for i, [annot_list, peak] in enumerate(zip(scan.annotations, scan.spectrum)):
        if len(annot_list.entries) == 1:
            annot = annot_list.entries[0]
            
            if annot.getName() == "?": continue
            if scan.mask[i] != 0: continue
            
            aa2csv_writer[pep_start].writerow([pep, mod_string, z, weight, nce, annot.getName(), str(peak.getIntensity())])
            #print([pep, mod_string, z, weight, nce, annot.getName(), str(peak.getIntensity())])
     
    