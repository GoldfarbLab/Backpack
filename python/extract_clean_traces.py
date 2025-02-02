import sys
from collections import defaultdict
import msp
from similarity import CS_scans
import csv

#data_path = "/Users/dennisgoldfarb/Downloads/procal.msp"
#data_path = "/Users/dennisgoldfarb/Downloads/ProCal/merged/procal.msp.deisotoped"
data_path = sys.argv[1] #"/Users/dennisgoldfarb/Downloads/ProCal/v2/procal.msp"
out_path = sys.argv[2]

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
    

# Compute pairwise similarity
with open(out_path, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter="\t")
    writer.writerow(["pep", "z", "NCE", "scan", "frag", "intensity", "rawOvFtT", "purity"])
    for pep in pep2z2NCE2scans:
        print(pep)
        for z in pep2z2NCE2scans[pep]:

            for NCE in pep2z2NCE2scans[pep][z]:
                for scan in pep2z2NCE2scans[pep][z][NCE]:
                    for i, mask in enumerate(scan.mask):
                        if mask == 0 or mask == 3:
                            name = scan.annotations[i].annotationName()
                            if name != "?":
                                #if all([annot.error <= 10 for annot in scan.annotations[i].entries]):
                                    writer.writerow([pep, z, NCE, scan.metaData.scanID, name, scan.spectrum[i].getIntensity(), scan.metaData.rawOvFtT, scan.metaData.purity])
                        
                    
                    # # get clean and ambig annotations
                    # clean_annotations = scan.getAnnotationsByMask(set([0]))
                    # ambig_annotations = scan.getAnnotationsByMask(set([3]))
                    
                    # #scan.normalizeToTotalAnnotated()
                    
                    # for annot_list in clean_annotations:
                    #     #for entry in annot_list.entries:
                    #     name = entry.getName()
                    #     intensity = scan.getAnnotationIntensity(name)
                    #     writer.writerow([pep, z, NCE, scan.metaData.scanID, name, intensity])
                    
                    # for annot_list in ambig_annotations:
                    #     for entry in annot_list.entries:
                    #         name = entry.getName()
                    #         writer.writerow([pep, z, NCE, scan.metaData.scanID, annot_list.annotationName(), "NA"])
            

    
    