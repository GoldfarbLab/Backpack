from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import defaultdict
from similarity import spectralAngle
import msp
import random as random
import argparse
import os
import sys
import clr
import raw_utils

from System import *
from System.Collections.Generic import *

clr.AddReference(os.path.join(os.path.dirname(__file__),"../libs/ThermoFisher.CommonCore.Data.dll"))
clr.AddReference(os.path.join(os.path.dirname(__file__),"../libs/ThermoFisher.CommonCore.RawFileReader.dll"))

from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter
from ThermoFisher.CommonCore.Data.Business import Device



parser = argparse.ArgumentParser(
                    prog='NCE alignment',
                    description='Align NCE values using ProCal')
parser.add_argument("msp_path")
parser.add_argument("raw_path")
parser.add_argument("out_path")
parser.add_argument("--lumos_model", default="/storage1/fs1/d.goldfarb/Active/Projects/Backpack/python/QE_to_lumos_poly_noRoot_mean.tsv", type=str)
parser.add_argument("--spline_model", default="/storage1/fs1/d.goldfarb/Active/Projects/Backpack/python/spline_fits_sqrt.tsv", type=str)
args = parser.parse_args()

class QE2lumos_models:
    def __init__(self):
        self.pep2z2poly = defaultdict(dict)
        self.init_poly_models()
        
    def init_poly_models(self):
        # read spline file
        with open(args.lumos_model, 'r') as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            header = next(tsvreader)
            for row in tsvreader:
                pep = row[0]
                z = int(row[1])
                c0 = float(row[2])
                c1 = float(row[3])
                c2 = float(row[4])
                c3 = float(row[5])
                
                self.pep2z2poly[pep][z] = np.poly1d([c3,c2,c1,c0])
                
    def eval(self, pep, z, NCE_QE):
        return self.pep2z2poly[pep][z](NCE_QE)


class spline_models:
    def __init__(self):
        self.pep2z2frag2fit = defaultdict(dict)
        self.init_spline_models()
        
    def init_spline_models(self):
        # read spline file
        with open(args.spline_model, 'r') as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            header = next(tsvreader)
            for row in tsvreader:
                pep = row[0]
                z = int(row[1])
                frag = row[2]
                coefs = [float(c) for c in row[3].strip(",").split(",")]
                knots = [float(c) for c in row[4].strip(",").split(",")]
                
                # create and store bspline
                if z not in self.pep2z2frag2fit[pep]:
                    self.pep2z2frag2fit[pep][z] = dict()
                self.pep2z2frag2fit[pep][z][frag] = BSpline(knots, coefs, 3, extrapolate=True, axis=0)
                
    def evalSpline(self, pep, z, frag, NCE):
        return max(0, self.pep2z2frag2fit[pep][z][frag](NCE))
    
    def predictSpectrum(self, pep, z, NCE):
        frag2intensity = dict()
        for frag in self.pep2z2frag2fit[pep][z]:
            if np.min(np.abs(self.pep2z2frag2fit[pep][z][frag].t - NCE)) <= 2.5:
                frag2intensity[frag] = float(self.evalSpline(pep, z, frag, NCE))
        total_int = sum(frag2intensity.values())
        if total_int > 0:
            for frag in frag2intensity:
                frag2intensity[frag] /= total_int
        return frag2intensity
    
    def hasModel(self, pep, z):
        return pep in self.pep2z2frag2fit and z in self.pep2z2frag2fit[pep]

def SA(s1, s2):
    intersection_set = set(s1.keys()) & set(s2.keys())
    if len(intersection_set) <= 2: return 0 #np.nan
    if sum(s2.values()) <= 0: return 0 #np.nan
    
    # observed
    v1_a = np.array([s1[f] for f in intersection_set])
    
    # predicted
    v2_a = np.array([s2[f] for f in intersection_set])
    
    return spectralAngle(v1_a, v2_a, 0)

def scan2dict(scan):
    frag2intensity = dict()
    for i, peak in enumerate(scan.spectrum):
        # check mask
        if scan.mask[i] == 0 or scan.mask[i] == 3:
            #if all([annot.error <= 10 for annot in scan.annotations[i].entries]):
                frag2intensity[scan.annotations[i].annotationName()] = peak.getIntensity()
            
    return frag2intensity

def get_SAs_for_scan(spline_mods, seq, z, scan, NCE_target, NCEs):
    frag2intensity = scan2dict(scan)
    SAs = np.zeros_like(NCEs)
    for i, NCE in enumerate(NCEs):
        SAs[i] = (SA(frag2intensity, spline_mods.predictSpectrum(seq, z, NCE)))
    return SAs

def plot_SA_offsets_for_scans(SAs, weighted_SAs, NCE):
    fig, ax = plt.subplots(2,1, figsize=(10, 2))
    mat = np.array(SAs).reshape(len(SAs), len(SAs[0]))
    ax[0].matshow(mat, aspect='auto')
    ax[0].axis("off")
    ax[1].matshow(np.array(weighted_SAs).reshape(1, len(weighted_SAs)), aspect='auto')
    ax[1].axis("off")
    fig.tight_layout()
    plt.savefig(os.path.join(args.out_path, os.path.basename(args.msp_path) + "_" + str(NCE) + "_mat.pdf"))
    plt.close()
    
def plot_index_offsets(SAs, NCE):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.matshow(np.array(SAs).reshape(1, len(SAs)), aspect='auto')
    ax.axis("off")
    fig.tight_layout()
    plt.savefig(os.path.join(args.out_path, os.path.basename(args.msp_path) + "_index_offsets_" + str(NCE) + "_mat.pdf"))
    plt.close()

def compute_SA_offsets(all_scans, pep, pep_z, NCE_target, NCEs):
    scan2SAs = []
    weights = []
    bestIndices = []
    
    plotScanSAs = []
    # Group scans by pep, z, and NCE
    for scan in all_scans:
        seq = scan.peptide.toString()
        z = scan.metaData.z
        
        if scan.metaData.NCE != NCE_target: continue
        if seq != pep: continue
        if z != pep_z: continue

        if not spline_mods.hasModel(seq, z): continue
        if len(scan2dict(scan)) < 2: continue
        
        SAs = get_SAs_for_scan(spline_mods, seq, z, scan, NCE_target, NCEs)
        scan2SAs.append(SAs)
        weights.append(np.sqrt(scan.metaData.rawOvFtT * scan.metaData.purity))
        bestIndices.append(np.argmax(SAs))
    
    if len(scan2SAs) == 0: return None, None, None, None
    
    #weighted_SAs = [np.average(np.array([s[i] for s in scan2SAs]), weights=weights) for i in range(len(scan2SAs[0]))]
    # Potentially do median instead of weighted average
    weighted_SAs = [np.median(np.array([s[i] for s in scan2SAs])) for i in range(len(scan2SAs[0]))]
    bestNCE = NCEs[int(np.ceil(np.median(bestIndices)))]
    
    if len(plotScanSAs) > 0:
        plot_SA_offsets_for_scans(plotScanSAs, weighted_SAs, NCE_target)
    
    return weighted_SAs, sum(weights), len(weights), bestNCE


def calibrate_lumos(all_scans):
    offsets = []
    valid_NCEs = []
    num_precursors = []
    num_scans = []
    NCE2pep2z2offset = defaultdict(dict)
    for NCE_target in [20, 23, 25, 28, 30, 33, 35]:
        values = []
        weights = []
        num_scans_NCE = 0
        min_NCE = max(10,NCE_target-15)
        max_NCE = min(50,NCE_target+15)
        step_size = 0.1
        steps = int(np.ceil((max_NCE - min_NCE) / step_size))+1
        NCEs = np.linspace(min_NCE, max_NCE, steps)
        
        for pep in spline_mods.pep2z2frag2fit:
            NCE2pep2z2offset[NCE_target][pep] = dict()
            for pep_z in spline_mods.pep2z2frag2fit[pep]:
                if pep_z != 2: continue
                #print(pep, pep_z)
                
                weighted_SAs, weight, scans, bestNCE = compute_SA_offsets(all_scans, pep, pep_z, NCE_target, NCEs)
                if weighted_SAs is None: continue
                
                NCE_lumos_ref = QE2lumos_mods.eval(pep, pep_z, bestNCE)
                
                num_scans_NCE += scans

                bestNCE_diff = NCE_lumos_ref - NCE_target
                values.append(round(bestNCE_diff, 3))
                weights.append(np.sqrt(weight))
                
                NCE2pep2z2offset[NCE_target][pep][pep_z] = round(bestNCE_diff, 3)
                
        if len(values) == 0: continue
        
        NCE_offset = np.median(values)

        offsets.append(NCE_offset)
        valid_NCEs.append(NCE_target)
        num_precursors.append(len(values))
        num_scans.append(num_scans_NCE)
        
    return offsets, valid_NCEs, num_precursors, num_scans, NCE2pep2z2offset


spline_mods = spline_models()
QE2lumos_mods = QE2lumos_models()

all_scans = [scan for scan in msp.read_msp_file(args.msp_path) if spline_mods.hasModel(scan.peptide.toString(), scan.metaData.z) and scan.peptide.toString() in QE2lumos_mods.pep2z2poly and scan.metaData.z in QE2lumos_mods.pep2z2poly[scan.peptide.toString()]]

offsets, valid_NCEs, num_precursors, num_scans, NCE2pep2z2offset = calibrate_lumos(all_scans)

with open(os.path.join(args.out_path, os.path.basename(args.msp_path) + ".NCE"), "w") as outfile:
    filename = os.path.basename(args.msp_path).split(".")[0]
    
    if "DDA" in filename:
        target_NCEs = [28]
    elif "2xIT" in filename:
        target_NCEs = [20,23]
    else:
        target_NCEs = [25,30,35]
    
    
    # extract calibration date
    rawFile = RawFileReaderAdapter.FileFactory(args.raw_path)
                
    rawFile.SelectInstrument(Device.MS, 1)
    cal_date = "NA"
    for i, header in enumerate(rawFile.GetTuneDataHeaderInformation()):
        if header.Label == "NormalizeCollisionEnergy":
            cal_date = rawFile.GetTuneDataValues(0, True).Values[i]
            break
        
    rawFileMetaData = raw_utils.getFileMetaData(rawFile)
    
    for NCE in target_NCEs:
        if NCE in valid_NCEs:
            outfile.write(filename + "\t" + rawFileMetaData.instrument_id + "\t" + cal_date + "\t" + str(rawFileMetaData.created_date) + "\t" + str(NCE) + "\t" + str(offsets[valid_NCEs.index(NCE)]) + "\n")
        else:
            outfile.write(filename + "\t" + rawFileMetaData.instrument_id + "\t" + cal_date + "\t" + str(rawFileMetaData.created_date) + "\t" + str(NCE) + "\t" + "NA" + "\n")
            
    # write summary stats
    #for diff, nce, npre, nscans in zip(diffs, valid_NCEs, num_precursors, num_scans):
    #    outfile.write(str(nce) + " " + str(diff) + " " + str(npre) + " " + str(nscans) + "\n")
    
    # write precursor stats
    #for NCE in NCE2pep2z2offset:
    #    for pep in NCE2pep2z2offset[NCE]:
    #        for z in NCE2pep2z2offset[NCE][pep]:
    #            outfile.write("precursor " + str(NCE) + " " + pep + " " + str(z) + " " + str(oms.AASequence.fromString(pep).getMonoWeight(oms.Residue.ResidueType.Full, z)) + " " + str(NCE2pep2z2offset[NCE][pep][z]) + "\n") 
    
    
    #outfile.write("num scans: " + str(len(all_scans)) + "\n")
    #outfile.write("num diffs: " + str(len(diffs)) + "\n")
    
    
    
    #print("num scans:", str(len(all_scans)))
    #print("num diffs: ", str(len(diffs)))
    #if len(diffs) > 0:
    #    outfile.write("mean: " + str(np.mean(diffs)) + "\n")


#sys.exit()
#correction_factor = np.mean(diffs) if len(diffs) > 0 else 0

#with open(os.path.join(args.out_path, os.path.basename(args.msp_path) + ".NCE"), "w") as outfile:
#    for scan in msp.read_msp_file(args.msp_path):
#        scan.metaData.key2val["NCE_aligned"] = scan.metaData.NCE + correction_factor
#        scan.updateMSPName()
#        scan.writeScan(outfile)

