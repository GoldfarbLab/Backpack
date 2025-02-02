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
import utils
import similarity


parser = argparse.ArgumentParser(
                    prog='MSP Reisotoper',
                    description='Reisotope msp files and compare to original')
parser.add_argument("msp_path")
parser.add_argument("deisotoped_path")
parser.add_argument("out_path")
args = parser.parse_args()

if not os.path.exists(args.out_path): os.makedirs(args.out_path)


def processFile(annot_file, deisotoped_file):
    filename = annot_file.split('/')[-1]
    print(filename)
    
    with open(os.path.join(args.out_path, filename + ".tsv"), 'w') as outfile:
        
        for i, (annot_scan, deisotoped_scan) in enumerate(zip(msp.read_msp_file(annot_file), msp.read_msp_file(deisotoped_file))):
            if i > 0 and i % 10000 == 0: print(i)
            
            if annot_scan.metaData.scanID != deisotoped_scan.metaData.scanID:
                print("FAIL")
                sys.exit()
                
            if "Acetyl" in annot_scan.name: continue
            
            reisotoped_scan = reisotope(deisotoped_scan)
            annot_scan.normalizeToTotalAnnotated(doLOD = True)
            SA = similarity.CS_scans(annot_scan, reisotoped_scan, unannotated=False)
            #similarity = compareScans(annot_scan, reisotoped_scan)
            outfile.write(str(SA) + "\n")

def reisotope(scan):
    
    mzs = []
    intensities = []
    annots = []
    mask = []
    
    for i, annot_list in enumerate(scan.annotations):
        if scan.mask[i] == 1: continue
        
        for annot_mono in annot_list.entries:
        #annot_mono = annot_list.entries[0]
            iso_dist = annot_mono.getTheoreticalIsotopeDistribution(scan.peptide, scan.metaData.iso2eff)
            for iso, prob in enumerate(iso_dist):
                annot = copy.deepcopy(annot_mono)
                annot.isotope = iso
                mz = annot.getMZ(scan.peptide)
                intensity = prob * scan.spectrum[i].getIntensity()
                mzs.append(mz)
                intensities.append(intensity)
                annots.append(annot)
                mask.append(scan.mask[i])
        
    # merge
    mzs = np.array(mzs)
    mz_order = np.argsort(mzs)
    mzs = mzs[mz_order]
    intensities = np.array(intensities)[mz_order]
    annots = np.array(annots)[mz_order]
    mask = np.array(mask)[mz_order]
    
    final_mzs = []
    final_intensites = []
    final_annotations = []
    
    #if scan.metaData.scanID == 33339:
    #    for i in range(len(mzs)-1):
    #        print(mzs[i], mzs[i+1], intensities[i], mask[i], annots[i].getIsoName(), utils.getPPMAbs(mzs[i+1], mzs[i]))
    
    
    i = 0
    ppm_tol = 20
    while i < len(mzs):
        if i < len(mzs)-1:
            if utils.getPPMAbs(mzs[i+1], mzs[i]) <= ppm_tol:
                group_indicies = [i]
                while i < len(mzs)-1 and utils.getPPMAbs(mzs[i+1], mzs[i]) <= ppm_tol:
                    group_indicies.append(i+1)
                    i+=1
                    #if scan.metaData.scanID == 9475:
                    #    print(mzs[i+1], mzs[i], utils.getPPMAbs(mzs[i+1], mzs[i]))
                #if scan.metaData.scanID == 9475:
                #    for j in group_indicies:
                #        print("grouped", annots[j].getIsoName(), group_indicies, mask[group_indicies], mzs[group_indicies], intensities[group_indicies])
                i+=1
                if any(mask[group_indicies] > 0): continue
                final_mzs.append(np.mean(mzs[group_indicies]))
                final_intensites.append(np.sum(intensities[group_indicies]))
                final_annotations.append(annotation.annotation_list(annots[group_indicies]))
            else:
                if mask[i] == 0:
                    final_mzs.append(mzs[i])
                    final_intensites.append(intensities[i])
                    final_annotations.append(annotation.annotation_list([annots[i]]))
                i+=1
        elif utils.getPPMAbs(mzs[i], mzs[i-1]) > ppm_tol:
            if mask[i] == 0:
                final_mzs.append(mzs[i])
                final_intensites.append(intensities[i])
                final_annotations.append(annotation.annotation_list([annots[i]]))
            i+=1
    
    # overwrite
    scan.spectrum.set_peaks([final_mzs, final_intensites])
    scan.annotations = final_annotations
    scan.mask = [0] * len(final_annotations)
    
    return scan

def compareScans(annot_scan, reisotoped_scan):
    
    return 1

processFile(args.msp_path, args.deisotoped_path)