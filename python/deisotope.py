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
import copy

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
    scan.normalizeToTotalAnnotated(doLOD = True)
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
                annot = copy.deepcopy(annot)
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
            if base_ion2isotope[base_ion][0][1] <= 0: # masked outside scan range
                mzs.append(annot.getMZ(scan.peptide))
                intensities.append(base_ion2isotope[base_ion][0][1])
                annotations.append(annotation.annotation_list([annot]))
                masks.append(1)
            #if len(base_ion2isotope[base_ion]) == 1:
            #    mzs.append(annot.getMZ(scan.peptide))
            #    intensities.append(base_ion2isotope[base_ion][0][1])
            #    annotations.append(annotation.annotation_list([annot]))
            #    if base_ion2isotope[base_ion][0][1] > 0: # no deisotoping necessary
            #        masks.append(0)
            #    else:
            #        masks.append(1) # this was masked outside the scan range already
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
                
                sim_threshold = 0.9 + (min(1, (observed_dist*theo_dist).sum() / 0.2) * 0.05)
                    
                if sim >= sim_threshold:
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
        
    # Deconvolve ambiguous
    for ambig_group in ambig_annot_group:
        ambig_group = list(ambig_group)
        name2iso_peak = defaultdict(list)
        
        # check if perfect overlap between any annotations (same isotope)
        for i, annot_list in enumerate(scan.annotations):
            for annot in annot_list.entries:
                for annot_name in ambig_group:
                    if annot_name == annot.getName():
                        name2iso_peak[annot_name].append([annot.isotope, i])
        
        identical = False              
        for name1 in name2iso_peak:
            for name2 in name2iso_peak:
                if name1 == name2: continue
                if name2iso_peak[name1] == name2iso_peak[name2]:
                    identical = True
                    break
            if identical: break
            
        
        if identical:
            # give ambig mask to sum of all peaks
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
        
        else:
            # deconvolve
            max_iso = scan.getMaxIsotope()
            output_size = len(ambig_group) * (max_iso+1)
            b = np.zeros(output_size)
            
            #print(ambig_group, max_iso)
            # populate the front with annotated ambig peaks
            b_i = 0
            name2iso_index = defaultdict(list)
            for i, annot_list in enumerate(scan.annotations):
                found = False
                for annot in annot_list.entries:
                    for annot_name in ambig_group:
                        if annot_name == annot.getName():
                            name2iso_index[annot_name].append([annot.isotope, b_i])
                            if not found:
                                #print(scan.metaData.scanID, annot_name, annot.isotope, b_i)
                                b[b_i] = scan.spectrum[i].getIntensity()
                                found = True
                                
                if found:
                    b_i += 1
                    
            # create matrix A
            A = np.zeros((output_size, len(ambig_group)))
            
            # populate template matrix
            next_empty_index = b_i
            # keep track of what % of the distribution was observed
            base_ion2percent = defaultdict(float)
            for ion_index, base_ion in enumerate(ambig_group):
                # compute isotope distribution
                iso_dist = name2annot[base_ion].getTheoreticalIsotopeDistribution(scan.peptide, scan.metaData.iso2eff)
                
                for iso, prob in enumerate(iso_dist):
                    isotope_match_index = -1
                    for isotope_match in name2iso_index[base_ion]:
                        if isotope_match[0] == iso:
                            isotope_match_index = isotope_match[1]
                    if isotope_match_index != -1:
                        A[isotope_match_index, ion_index] = prob
                        base_ion2percent[base_ion] += prob
                    else:
                        A[next_empty_index, ion_index] = prob
                        next_empty_index+=1
            
            # solve it
            total_signal = b.sum()
            b /= b.max()
            x = nnls(A, b, maxiter=10000, atol=1e-5)
            
            x_perc = x[0] / x[0].sum()
            x_sig = x_perc * total_signal
            
            if x[1] <= 0.2:
                # output deconvolved values
                for i, annot_name in enumerate(ambig_group):
                    annot = copy.deepcopy(name2annot[annot_name])
                    annot.isotope = 0
                    mz = annot.getMZ(scan.peptide)
                    masks.append(0)
                    mzs.append(mz)
                    annotations.append(annotation.annotation_list([annot]))
                    intensities.append(x_sig[i]) #  * base_ion2percent[annot.getName()]
                
            else:
                # output masked values
                masks.append(5)
                mzs.append(name2annot[list(ambig_group)[0]].getMZ(scan.peptide))
                annotations.append(annotation.annotation_list([name2annot[name] for name in ambig_group]))
                intensities.append(total_signal)

        


    scan.spectrum.set_peaks([mzs, intensities])
    scan.annotations = annotations
    scan.mask = masks
    

    return scan
 
processFile(args.msp_path, write_unannotated=False)