import sys
from collections import defaultdict
import msp
from similarity import CS_scans
import csv

#data_path = "/Users/dennisgoldfarb/Downloads/procal.msp"
data_path = "/Users/dennisgoldfarb/Downloads/ProCal/merged/procal.msp.deisotoped"

pep2z2NCE2scans = defaultdict(dict)

# Group scans by pep, z, and NCE
for i, scan in enumerate(msp.read_msp_file(data_path)):
    if i > 0 and i % 10000 == 0: print(i)
    
    if scan.fileMetaData.model == "Orbitrap Fusion Lumos": continue
    seq = scan.peptide.toString()
    z = scan.metaData.z
    NCE = scan.metaData.NCE

    if z not in pep2z2NCE2scans[seq]:
        pep2z2NCE2scans[seq][z] = dict()
    if NCE not in pep2z2NCE2scans[seq][z]:
        pep2z2NCE2scans[seq][z][NCE] = []
    
    pep2z2NCE2scans[seq][z][NCE].append(scan)
    

def compare_same_NCE():
    # Compute pairwise similarity
    with open("/Users/dennisgoldfarb/Downloads/procal_cs_deisotoped.tsv", 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(["pep", "z", "NCE", "model_comp", "cs", "cs_annot", "scan1", "scan2", "rt1", "rt2"])
        for pep in pep2z2NCE2scans:
            #if pep != "TSIDSFIDSYK": continue
            for z in pep2z2NCE2scans[pep]:
                print(pep, z)
                for NCE in pep2z2NCE2scans[pep][z]:
                    for i, scan1 in enumerate(pep2z2NCE2scans[pep][z][NCE]):
                        for j, scan2 in enumerate(pep2z2NCE2scans[pep][z][NCE]):
                            if i <= j: continue
                            #if not (scan1.metaData.scanID == 17198 and scan2.metaData.scanID == 17240 and scan1.metaData.NCE == 40): continue
                            # Compute CS
                            cs = CS_scans(scan1, scan2, unannotated=True, ambiguous=True, terminal=True, precursor=True, immonium=True, neutral_losses=True, internal=True)
                            cs_annot = CS_scans(scan1, scan2, unannotated=False, ambiguous=False, terminal=True, precursor=True, immonium=True, neutral_losses=True, internal=True)

                            model_comp = "-".join(sorted(set([scan1.fileMetaData.model, scan2.fileMetaData.model])))
                            # Write result
                            writer.writerow([pep, z, NCE, model_comp, cs, cs_annot, scan1.metaData.scanID, scan2.metaData.scanID, scan1.metaData.RT, scan2.metaData.RT])


def compare_all_NCE():
    # Compute pairwise similarity
    with open("/Users/dennisgoldfarb/Downloads/procal_cs_deisotoped_all_NCE_raw.tsv", 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(["pep", "z", "NCE1", "NCE2", "model_comp", "cs", "cs_annot", "scan1", "scan2", "rt1", "rt2"])
        for pep in pep2z2NCE2scans:
            #if pep != "ALFSSITDSEK": continue
            for z in pep2z2NCE2scans[pep]:
                for NCE1 in pep2z2NCE2scans[pep][z]:
                    for NCE2 in pep2z2NCE2scans[pep][z]:
                        if NCE1 < NCE2: continue
                        print(pep, z, NCE1, NCE2)
                        for i, scan1 in enumerate(pep2z2NCE2scans[pep][z][NCE1]):
                            for j, scan2 in enumerate(pep2z2NCE2scans[pep][z][NCE2]):
                                if i <= j: continue
                                #if not (scan1.metaData.scanID == 17198 and scan2.metaData.scanID == 17240 and scan1.metaData.NCE == 40): continue
                                # Compute CS
                                cs = CS_scans(scan1, scan2, unannotated=True, ambiguous=True, terminal=True, precursor=True, immonium=True, neutral_losses=True, internal=True)
                                cs_annot = CS_scans(scan1, scan2, unannotated=False, ambiguous=False, terminal=True, precursor=True, immonium=True, neutral_losses=True, internal=True)

                                model_comp = "-".join(sorted(set([scan1.fileMetaData.model, scan2.fileMetaData.model])))
                                # Write result
                                writer.writerow([pep, z, NCE1, NCE2, model_comp, cs, cs_annot, scan1.metaData.scanID, scan2.metaData.scanID, scan1.metaData.RT, scan2.metaData.RT])
        
compare_all_NCE()     