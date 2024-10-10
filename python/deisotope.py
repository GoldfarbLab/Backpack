#!/usr/bin/env python
import yaml
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import numpy as np
import msp
import argparse
from collections import defaultdict
from scipy.optimize import nnls
import annotation

parser = argparse.ArgumentParser(
                    prog='MSP Deisotoper',
                    description='Deisotoper msp files')
parser.add_argument("msp_path")
parser.add_argument("out_path")
args = parser.parse_args()

if not os.path.exists(args.out_path): os.makedirs(args.out_path)
    
def processFile(file, write_unannotated):
    filename = file.split('/')[-1]
    print(filename)
    
    with open(os.path.join(args.out_path, filename + ".deisotoped"), 'w') as outfile:
        for i, scan in enumerate(msp.read_msp_file(file)):
            if i > 0 and i % 10000 == 0: print(i)
            
            if scan.getAnnotatedBasePeakIntensity() <= 0: continue
            
            scan = deisotope(scan)
            scan.writeScan(outfile, write_unannotated, int_prec=5)

            
            
def deisotope(scan):
    scan.normalizeToBasePeak()
    #scan.normalizeToTotalAnnotated()
    #scan.normalizeToTotal()
    # create dictionary of base_ion2isotopes_index (observed)
    mzs = []
    intensities = []
    annotations = []
    masks = []
    
    base_ion2isotope = defaultdict(list)
    name2annot = dict()
    ambig_annots = set()
    ambig_peaks = []
    ambig_annot_group = []

    for i, annot_list in enumerate(scan.annotations):
        if annot_list.annotationString() == "?":
            mzs.append(scan.spectrum[i].getMZ())
            intensities.append(scan.spectrum[i].getIntensity())
            annotations.append(annotation.annotation_list([]))
            masks.append(0)
        else:
            for annot in annot_list.entries:    
                base_ion2isotope[annot.getName()].append([annot.isotope, scan.spectrum[i].getIntensity()])
                annot.isotope=0
                if annot.getName() not in name2annot:
                    name2annot[annot.getName()] = annot
            if len(annot_list.entries) > 1:
                ambig_peaks.append([scan.spectrum[i].getMZ(), scan.spectrum[i].getIntensity()])
                
                found_ambig_group_indices = set()
                for annot in annot_list.entries:
                    annot_name = annot.getName()
                    ambig_annots.add(annot_name)
                
                    # check if already present in an ambig_annot_group
                    for index, ambig_group in enumerate(ambig_annot_group):
                        if annot_name in ambig_group:
                            found_ambig_group_indices.add(index)
                            
                
                
                # make a new group
                if len(found_ambig_group_indices) == 0:
                    ambig_annot_group.append(set([annot.getName() for annot in annot_list.entries]))
                
                # add to existing group
                elif len(found_ambig_group_indices) == 1:
                    index = list(found_ambig_group_indices)[0]
                    for annot in annot_list.entries:
                        ambig_annot_group[index].add(annot.getName())
                
                # merge multiple groups and then add
                else:
                    new_group = set()
                    for index in found_ambig_group_indices:
                        for annot_name in ambig_annot_group[index]:
                            new_group.add(annot_name)
                    ambig_annot_group = [x for i, x in enumerate(ambig_annot_group) if i not in found_ambig_group_indices]
                    ambig_annot_group.append(new_group)
                    


    for base_ion in base_ion2isotope:
        annot = name2annot[base_ion]
        if base_ion not in ambig_annots:
            if len(base_ion2isotope[base_ion]) == 1:
                mzs.append(annot.getMZ(scan.peptide))
                intensities.append(base_ion2isotope[base_ion][0][1])
                annotations.append(annotation.annotation_list([annot]))
                if base_ion2isotope[base_ion][0][1] > 0: # no deisotoping necessary
                    masks.append(0)
                else:
                    masks.append(1) # this was masked outside the scan range already
            else:
                iso_dist = annot.getTheoreticalIsotopeDistribution(scan.peptide, scan.metaData.iso2eff)
                observed_dist = np.zeros(len(iso_dist))
                theo_dist = np.zeros(len(iso_dist))
                for iso, prob in enumerate(iso_dist):
                    theo_dist[iso] = prob
                    for isotope_match in base_ion2isotope[base_ion]:
                        if isotope_match[0] == iso:
                            observed_dist[iso] = isotope_match[1]
                
                sim = (observed_dist*theo_dist).sum() / np.linalg.norm(observed_dist) / np.linalg.norm(theo_dist)
                if observed_dist.sum() == 0:
                    print(base_ion, annot.getMZ(scan.peptide), scan.getName())
                    sys.exit(1)
                
                mzs.append(annot.getMZ(scan.peptide))
                intensities.append(observed_dist.sum())
                annotations.append(annotation.annotation_list([annot]))
                    
                if sim >= 0.9:
                    masks.append(0)
                else:
                    masks.append(2)
        else:
            # if ambiguous, then sum up all the isotopes and give it a bad isotope mask so that we use it as a maximum intensity constraint
            #totat_int = sum([isotope_match[1] for isotope_match in base_ion2isotope[base_ion]])
            #mzs.append(annot.getMZ(scan.peptide))
            #intensities.append(totat_int)
            #annotations.append(annotation.annotation_list([annot]))
            #masks.append("2")
            
            
            # UNCOMMENT FOR ALTIMETER
            # ambiguous ion group, but the peak itself was clean. Add it as ambiguous
            #mzs.append(annot.getMZ(scan.peptide))
            #intensities.append(base_ion2isotope[base_ion][0][1])
            #annotations.append(annotation.annotation_list([annot]))
            #masks.append(3)
            continue
        
    # FOR NCE CALIBRATION ONLY
    for ambig_group in ambig_annot_group:
        masks.append(5)
        mzs.append(name2annot[list(ambig_group)[0]].getMZ(scan.peptide))
        annotations.append(annotation.annotation_list([name2annot[name] for name in ambig_group]))

        peak_indices = set()
        
        for i, annot_list in enumerate(scan.annotations):
            for annot in annot_list.entries:
                for annot_name in ambig_group:
                    if annot_name == annot.getName():
                        peak_indices.add(i)
                        break
                    
        intensity = sum(scan.spectrum[i].getIntensity() for i in peak_indices)
        intensities.append(intensity)
        

    #for [mz, intensity] in ambig_peaks:
    #    mzs.append(mz)
    #    intensities.append(intensity)
    #    annotations.append(annotation.annotation_list([]))
    #    masks.append(3)

    scan.spectrum.set_peaks([mzs, intensities])
    scan.annotations = annotations
    scan.mask = masks
    

    return scan
 
processFile(args.msp_path, write_unannotated=True)