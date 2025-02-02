from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import defaultdict
from similarity import spectralAngle, scribe
import msp
import sys
import random as random


frag_spline_path = sys.argv[1] #spline_fits.tsv
data_path = sys.argv[2] #"/Users/dennisgoldfarb/Downloads/ProCal/v2/procal.msp"
out_path = sys.argv[3]

class spline_models:
    def __init__(self):
        self.pep2z2frag2fit = defaultdict(dict)
        self.init_spline_models()
        
    def init_spline_models(self):
        # read spline file
        with open(frag_spline_path, 'r') as tsvfile:
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
            #if (NCE <= max(self.pep2z2frag2fit[pep][z][frag].t) or max(self.pep2z2frag2fit[pep][z][frag].t) == 50) and (NCE >= min(self.pep2z2frag2fit[pep][z][frag].t) or min(self.pep2z2frag2fit[pep][z][frag].t) == 10):
            #if (NCE >= min(self.pep2z2frag2fit[pep][z][frag].t) or min(self.pep2z2frag2fit[pep][z][frag].t) == 10):
                if np.min(np.abs(self.pep2z2frag2fit[pep][z][frag].t - NCE)) <= 2.5:
                    frag2intensity[frag] = float(self.evalSpline(pep, z, frag, NCE))
        total_int = sum(frag2intensity.values())
        if total_int > 0:
            for frag in frag2intensity:
                frag2intensity[frag] /= total_int
        #else:
        #    print(pep, z, NCE, frag2intensity)
        return frag2intensity
    
    def hasModel(self, pep, z):
        return pep in self.pep2z2frag2fit and z in self.pep2z2frag2fit[pep]
    
def leftTrimMean(l, percent):
    return np.mean(sorted(l)[0:int(len(l)*percent)])

def weightedTrimMean(l, percent, weights):
    return np.average(sorted(l)[int(len(l)*percent):int(len(l)*(1-percent))], weights=weights[int(len(l)*percent):int(len(l)*(1-percent))])
    
def SA(s1, s2):
    intersection_set = set(s1.keys()) & set(s2.keys()) # union vs intersection
    if len(intersection_set) <= 2: return 0 #np.nan
    if sum(s2.values()) <= 0: return 0 #np.nan
    
    # observed
    #v1_a = []
    #for f in s2:
    #    if f in s1:
    #        v1_a.append(s1[f])
    #    else:
    #        v1_a.append(0)
    #v1_a = np.array(v1_a)      
    v1_a = np.array([s1[f] for f in intersection_set])
    
    # predicted
    #v2_a = []
    #for f in s2:
    #    if f in s2:
    #        v2_a.append(s2[f])
    #    else:
    #        v2_a.append(0)
    #v2_a = np.array(v2_a) 
    v2_a = np.array([s2[f] for f in intersection_set])
    
    return spectralAngle(v1_a, v2_a, 0)
    #return scribe(v1_a, v2_a, 0)


def scan2dict(scan):
    frag2intensity = dict()
    for i, peak in enumerate(scan.spectrum):
        # check mask
        if scan.mask[i] == 0 or scan.mask[i] == 3:
            #if scan.annotations[i].annotationName()[0] != "p": # not using precursor
            #if all([annot.error <= 10 for annot in scan.annotations[i].entries]):
                frag2intensity[scan.annotations[i].annotationName()] = peak.getIntensity()
            
    return frag2intensity

def get_SAs_for_scan(spline_mods, seq, z, scan, NCE_target):
    step_size = 0.1 #0.01
    min_NCE = min(5, NCE_target-20)
    max_NCE = max(55, NCE_target+20)
    steps = int(np.ceil((max_NCE - min_NCE) / step_size))
    frag2intensity = scan2dict(scan)
    SAs = []
    NCEs = []
    for i in range(steps):
        NCE = min_NCE + (i * step_size)
        NCEs.append(NCE)
        SAs.append(SA(frag2intensity, spline_mods.predictSpectrum(seq, z, NCE)))
    return SAs, NCEs

def compute_SA_offsets(all_scans, pep, pep_z, NCE_target):
    scan2SAs = []
    weights = []
    
    # Group scans by pep, z, and NCE
    for scan in all_scans:
        seq = scan.peptide.toString()
        z = scan.metaData.z
        
        if scan.metaData.NCE != NCE_target: continue
        if seq != pep: continue
        if z != pep_z: continue

        if not spline_mods.hasModel(seq, z): continue
        if len(scan2dict(scan)) < 2: continue
        
        SAs, NCEs = get_SAs_for_scan(spline_mods, seq, z, scan, NCE_target)
        scan2SAs.append(SAs)
        weights.append(np.sqrt(scan.metaData.rawOvFtT * scan.metaData.purity))
    
    if len(scan2SAs) == 0: return None, None, None
    
    #weighted_SAs = np.array([np.average(np.array([s[i] for s in scan2SAs]), weights=weights) for i in range(len(scan2SAs[0]))])
    weighted_SAs = np.array([np.median(np.array([s[i] for s in scan2SAs])) for i in range(len(scan2SAs[0]))])
    
    return weighted_SAs, sum(weights), NCEs



def model_lumos(all_scans):
    pep2z2fit = defaultdict(dict)
    with open(out_path, 'w') as outfile:
        
        outfile.write("pep z SA NCE_Lumos NCE_QE offset\n")
        
        for pep in spline_mods.pep2z2frag2fit:
            for pep_z in spline_mods.pep2z2frag2fit[pep]:
                #print(pep, pep_z)
                
                valid_NCEs = []
                values = []
                tot_weight = 0
                
                for NCE_target in [10,15,20,22,24,26,28,30,32,34,36,38,40,45,50]:
                    
                    weighted_SAs, weight, NCEs = compute_SA_offsets(all_scans, pep, pep_z, NCE_target)
                    if weighted_SAs is None: continue
                    
                    values.append(weighted_SAs)
                    valid_NCEs.append(NCE_target)
                    tot_weight += weight
                    
                    outfile.write(pep + " " + str(pep_z) + " " + str(np.max(weighted_SAs)) + " "  + str(NCE_target) + " " + str(NCEs[np.argmax(weighted_SAs)]) + " " + str(np.round(NCEs[np.argmax(weighted_SAs)]-NCE_target,2)) + "\n")
                    
                    
                if len(valid_NCEs) == 0: 
                    #print("none valid")
                    continue
                
                #points = [np.array(valid_NCEs), np.linspace(5, 55, num=len(values[0]))]
                #interp = RegularGridInterpolator(points, np.array(values), bounds_error=False, fill_value=0)
                #pep2z2fit[pep][pep_z] = [interp, tot_weight]
                
                #ut, vt = np.meshgrid(np.linspace(min(valid_NCEs), max(valid_NCEs), 500), np.linspace(5, 55, 500),  indexing='ij')
                #grid_points = np.array([ut.ravel(), vt.ravel()]).T
                
                #fig, axes = plt.subplots(1, 3, figsize=(10, 6))
                #axes = axes.ravel()
                #fig_index = 0
                #for method in ['linear', 'slinear', 'cubic']:
                #    im = interp(grid_points, method=method).reshape(500, 500)
                #    axes[fig_index].imshow(im)
                #    axes[fig_index].set_title(method)
                #    axes[fig_index].axis("off")
                #    fig_index += 1
                #fig.tight_layout()
                #plt.savefig('/Users/dennisgoldfarb/Downloads/'+pep+"_"+str(pep_z)+".pdf")
                #plt.close()
    return pep2z2fit       


def compute_xcorr(interp, test_SAs, NCE_target):
    method = "cubic"
    step_size = 0.01
    min_NCE = max(10, NCE_target-2)
    max_NCE = min(50, NCE_target+2)
    steps = 1+int(np.ceil((max_NCE - min_NCE) / step_size))
    xcorrs = []
    NCEs = []
    
    
    for i in range(steps):
        NCE = min_NCE + (i * step_size)
        ut, vt = np.meshgrid(np.linspace(NCE, NCE, 1), np.linspace(5, 55, len(test_SAs)),  indexing='ij')
        grid_points = np.array([ut.ravel(), vt.ravel()]).T
        im = interp(grid_points, method=method)
        im[im < 0] = 0
        xcorr = spectralAngle(np.array(im), np.array(test_SAs), 0)
        xcorrs.append(xcorr)
        NCEs.append(NCE)
        #print(NCE, xcorr)
    
    return xcorrs, NCEs

def calibrate_lumos(pep2z2fit, all_scans):
    diffs = []
    for NCE_target in [10,15,20,22,24,26,28,30,32,34,36,38,40,45,50]:
        values = []
        weights = []
        for pep in spline_mods.pep2z2frag2fit:
            for pep_z in spline_mods.pep2z2frag2fit[pep]:
                #print(pep, pep_z)
                
                weighted_SAs, weight = compute_SA_offsets(all_scans, pep, pep_z, NCE_target)
                if weighted_SAs is None: continue
                
                interp, model_weight = pep2z2fit[pep][pep_z]
                xcorrs, NCEs = compute_xcorr(interp, weighted_SAs, NCE_target)
                
                values.append(xcorrs)
                weights.append(np.sqrt(weight))
                
        if len(values) == 0: continue
        
        #avg_SAs = [np.average(np.array([s[i] for s in values])) for i in range(len(values[0]))]
        avg_SAs = [np.average(np.array([s[i] for s in values]), weights=weights) for i in range(len(values[0]))]
        #avg_SAs = [ weightedTrimMean(np.array([s[i] for s in values]), 0.1, weights=weights) for i in range(len(values[0]))]
       
        
        index = max(range(len(avg_SAs)), key=lambda i: avg_SAs[i])
        diffs.append(round(NCE_target - NCEs[index], 3))
        #print(NCE_target, NCEs[index], round(NCE_target - NCEs[index], 3), round(avg_SAs[index], 3))
    return diffs


spline_mods = spline_models()            

all_scans = [scan for scan in msp.read_msp_file(data_path) if scan.fileMetaData.model == "Orbitrap Fusion Lumos"]

# output models
pep2z2fit = model_lumos(all_scans)
