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
import bisect
import numpy as np
import pyopenms as oms
import clr

from System import *
from System.Collections.Generic import *

clr.AddReference(os.path.join(os.path.dirname(__file__),"../libs/ThermoFisher.CommonCore.Data.dll"))
clr.AddReference(os.path.join(os.path.dirname(__file__),"../libs/ThermoFisher.CommonCore.RawFileReader.dll"))

from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter
from ThermoFisher.CommonCore.Data.Business import Device

parser = argparse.ArgumentParser(
                    prog='Quad fit',
                    description='Estimates quad transmission efficiency')
parser.add_argument("msp_path")
parser.add_argument("raw_path")
parser.add_argument("out_path")
args = parser.parse_args()

if not os.path.exists(args.out_path): os.makedirs(args.out_path)
    
def processFile(file, raw_path):
    filename = file.split('/')[-1]
    print(filename)
    
    cal_date = extractCal(raw_path)
    
    with open(os.path.join(args.out_path, filename + ".quad"), 'w') as outfile:
        #outfile.write("\t".join(["z", "mz", "center_iso", "ratio_index", "ratio"]))
        for i, scan in enumerate(msp.read_msp_file(file)):
            if i > 0 and i % 10000 == 0: print(i)
            
            if scan.getAnnotatedBasePeakIntensity() <= 0: continue
            
            scan = fit_scan(scan, cal_date, outfile)
    
            
def extractCal(file):
     # extract calibration date
    rawFile = RawFileReaderAdapter.FileFactory(file)
                
    rawFile.SelectInstrument(Device.MS, 1)
    cal_date = "NA"
    for i, header in enumerate(rawFile.GetTuneDataHeaderInformation()):
        if header.Label == "IsolationCalibration":
            cal_date = rawFile.GetTuneDataValues(0, True).Values[i]
            break
        
    return cal_date   
            
def get_clean_frags(scan):
    ambig_frags = set()
    clean_frags = set()
    # get clean fragments
    for i, annot_list in enumerate(scan.annotations):
        if scan.mask[i] != 0: continue
        if len(annot_list.entries) == 1:
            annot = annot_list.entries[0]
            annot_name = annot.getName()
            if annot_name not in ambig_frags:
                #if annot.length >= 4 or annot.getType() == "p":
                    clean_frags.add(annot_name)
        elif len(annot_list.entries) > 1:
            for annot in annot_list.entries:
                annot_name = annot.getName()
                ambig_frags.add(annot_name)
                if annot_name in clean_frags:
                    clean_frags.remove(annot_name)
    return clean_frags

def get_observed_isotopes(scan, frags, max_iso):
    frag2isotopes = defaultdict(lambda: np.zeros(max_iso+1))
    frag2annot = dict()
    for i, annot_list in enumerate(scan.annotations):
        if len(annot_list.entries) == 1:
            annot = annot_list.entries[0]
            annot_name = annot.getName()
            if annot_name not in frags: continue
            if annot.isotope > max_iso: continue
            frag2isotopes[annot_name][annot.isotope] = scan.spectrum[i].getIntensity()
            frag2annot[annot_name] = annot
    return frag2isotopes, frag2annot

def get_precursor_dist(scan, max_iso):
    pep_formula = scan.peptide.getFormula()
    pep_iso_gen = oms.CoarseIsotopePatternGenerator(1+max_iso)
    pep_iso_dist = pep_formula.getIsotopeDistribution(pep_iso_gen)
    return np.array([peak.getIntensity() for peak in pep_iso_dist.getContainer()])

def get_center_isotope(scan, min_iso, max_iso):
    best_diff = 1000
    best_iso = min_iso
    
    for iso in range(min_iso, max_iso+1):
        diff = abs((scan.metaData.monoMz + ((oms.Constants.C13C12_MASSDIFF_U * iso) / scan.metaData.z)) - scan.metaData.isoCenter)
        if diff < best_diff:
            best_diff = diff
            best_iso = iso

    return best_iso
            
def fit_scan(scan, cal_date, outfile):
    # get isotope range
    max_iso = scan.getMaxIsotope()
    min_iso = scan.getMinIsotope()
    center_iso = get_center_isotope(scan, min_iso, max_iso)
    #if scan.metaData.z < 2 or scan.metaData.z > 3: return
    if scan.metaData.z > 2: 
        max_iso += 2
        min_iso = max(0, min_iso-2)
    else:
        max_iso += 1
        min_iso = max(0, min_iso-1)
    
    # get clean fragments
    clean_frags = get_clean_frags(scan)
    if len(clean_frags) == 0: return
    
    frag2isotopes_init, frag2annot = get_observed_isotopes(scan, clean_frags, max_iso)
    
    
    # get the top X% abundant fragments
    frag_sums_init = [(frag, sum(frag2isotopes_init[frag])) for frag in frag2isotopes_init]
    frag_sums_init = sorted(frag_sums_init, key=lambda tup: -tup[1])
    total = sum([frag[1] for frag in frag_sums_init])
    prefix_sums = [sum([frag_sums_init[j][1] for j in range(0,i+1)])/total for i in range(len(frag_sums_init))]    
    last_index = bisect.bisect_left(prefix_sums, 0.25) + 1
    
    
    frag2isotopes = dict()
    frag_sums = []
    
    for i in range(last_index):
        (frag, frag_sum) = frag_sums_init[i]
        frag2isotopes_init[frag] /= frag_sum
        
        annot = frag2annot[frag]
        if annot.length >= 4 or annot.getType() == "p":
            frag2isotopes[frag] = frag2isotopes_init[frag]
            frag_sums.append(frag_sums_init[i])
        
    last_index = len(frag2isotopes)
    if last_index == 0: return
        

    # build matrix
    num_iso = (max_iso - min_iso) + 1
    A = np.zeros((last_index * (max_iso+1), num_iso))
    b = np.zeros(last_index * (max_iso+1))

    # fill b
    k = 0
    for i in range(last_index):
        for j in range(max_iso + 1):
            b[k] = frag2isotopes[frag_sums[i][0]][j]
            k += 1

    # fill A
    for i in range(last_index):
        for iso in range(min_iso, max_iso + 1):
            # compute frag isotope distribution
            frag = frag_sums[i][0]
            annot = frag2annot[frag]
            iso2eff = {iso : 1}
            iso_dist = annot.getTheoreticalIsotopeDistribution(scan.peptide, iso2eff)
            iso_dist /= np.sum(iso_dist) #np.max(iso_dist)
            iso_dist *= frag_sums[i][1]

            for k, prob in enumerate(iso_dist):
                A[i * (max_iso+1) + k, iso-min_iso] = prob
    
   
    
    try:
        x = nnls(A, b, maxiter=1000)[0]
    except:
        return
    
    #print(A)
    #print(b)
    x_theo = get_precursor_dist(scan, max_iso)
    #print(scan.metaData.z, min_iso, max_iso)
    #print(x_theo)
    #print(x)
    
    #sys.exit()
    
    if x[center_iso-min_iso] == 0: return
    
    for i in range(0, x.shape[0]):
        ratio_theo = x_theo[i] / x_theo[center_iso-min_iso]
        if ratio_theo <= 0.25 or ratio_theo >= 4: continue
        ratio = x[i] / x[center_iso-min_iso]
        outfile.write("\t".join([str(scan.fileMetaData.created_date),
                                 str(scan.metaData.isoCenter), 
                                 str(scan.metaData.z),
                                 str(cal_date),
                                 str((scan.metaData.monoMz + (((oms.Constants.C13C12_MASSDIFF_U * i) + min_iso) / scan.metaData.z)) - scan.metaData.isoCenter),
                                 str(ratio / ratio_theo)]))
        outfile.write("\n")
    
    # if x_theo.shape[0] < 2: return
    # if x.shape[0] < 2: return
    # if x[0] == 0: return
    
    
    # ratio_theo = x_theo[1] / x_theo[0]
    # ratio = x[1] / x[0]
    # outfile.write("\t".join([str(scan.fileMetaData.created_date),
    #                          str(scan.metaData.isoCenter),
    #                          str(scan.metaData.z),
    #                          str(min_iso) + "-" + str(max_iso),
    #                          str((scan.metaData.monoMz + (oms.Constants.C13C12_MASSDIFF_U / scan.metaData.z)) - scan.metaData.isoCenter),
    #                          str(ratio / ratio_theo)]))
    # outfile.write("\n")
    
    
processFile(args.msp_path, args.raw_path)