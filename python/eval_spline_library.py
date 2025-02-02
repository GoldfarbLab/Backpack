import json
import numpy as np
import pyopenms as oms
from scipy.interpolate import BSpline
import annotation
from collections import defaultdict
import csv
import utils
from bisect import bisect_left
import msp
import similarity
import sys
import os
import spline_library
import pickle




picke_path = sys.argv[1]
out_path = sys.argv[2]
msp_path = "/scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/test_raw.msp"
#msp_path = "/scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/Part1/01625b_GB4-TUM_first_pool_26_01_01-2xIT_2xHCD-1h-R1/results/annotated/01625b_GB4-TUM_first_pool_26_01_01-2xIT_2xHCD-1h-R1.msp"

library = pickle.load(open(picke_path, 'rb')) 

print(len(library.precursors), flush=True)
bin_width = 0.02
hits = 0
misses = 0

with open(out_path, 'w') as outfile:   
    outfile.write("SA\t" + 
                "bestSA\t" + 
                "scribe\t" + 
                "bestScribe\t" + 
                "matches\t" + 
                "offset\t" + 
                "filename\t" + 
                "scanID\t" + 
                "NCE\t" + 
                "seq\t" + 
                "z\n") 
    # Read MSP
    for i, scan in enumerate(msp.read_msp_file(msp_path)):
        #batch_idx = int(i / batch_size)
        #if batch_idx % num_jobs != job_id: continue  
        if i > 0 and i % 10000 == 0: print(i, flush=True)
        
        # convert to sparse binned spectrum
        observed_spectrum = spline_library.binSpectrum(scan.spectrum, bin_width)
        scan_seq = scan.peptide.toString()
        
        target_SA = 0
        target_scribe = 0
        best_SA = 0
        best_scribe = 0
        num_matches = 0
        # get precursors in range
        for j, entry in enumerate(library.getPrecursorsInRange(scan.peptide.getMZ(scan.metaData.z), 5, scan_seq)):   
            
            NCE = float(scan.metaData.key2val["NCE_aligned"])
            #NCE = float(scan.metaData.NCE)
            
            if j == 0:
                # align NCE
                best_offset = 0
                best_SA_offset = 0
                for NCE_offset in np.linspace(-10,10,201):
                    predicted_spectrum = entry.getSpectrum(NCE+NCE_offset, scan.metaData.iso2eff, bin_width, library)
                    v1,v2 = spline_library.alignBinnedSpectra(observed_spectrum, predicted_spectrum)
                    if sum(v1 > 0) <= 3 or sum(v2 > 0) <= 3: continue
                    # LOD
                    v1 /= max(v1)
                    LOD = min(v1[v1>0])
                    # compute SA
                    SA = similarity.scribe(v1,v2,LOD)
                    if SA > best_SA_offset:
                        best_SA_offset = SA
                        best_offset = NCE_offset
                #print("offset:", best_offset, best_SA_offset, NCE)
             
            #best_offset = -0.6 
                        
            predicted_spectrum = entry.getSpectrum(NCE+best_offset, scan.metaData.iso2eff, bin_width, library)
            v1,v2 = spline_library.alignBinnedSpectra(observed_spectrum, predicted_spectrum)
            if sum(v1 > 0) <= 3 or sum(v2 > 0) <= 3: continue
            # LOD
            v1 /= max(v1)
            LOD = min(v1[v1>0])
            # compute SA
            SA = similarity.spectralAngle(v1,v2,LOD)
            # compute scribe
            scribe = similarity.scribe(v1,v2,LOD)
            
            #if scan_seq == entry.peptide:
            if j == 0:
                target_SA = SA
                target_scribe = scribe
                print("offset:", hits, target_SA, target_scribe, NCE, flush=True)
            else:
                best_SA = max(SA, best_SA)
                best_scribe = max(scribe, best_scribe)
                num_matches += 1
            
        # output
        if target_SA > 0:
            hits+=1
            #print("hit:", hits, "misses:", misses, flush=True)
            outfile.write(str(target_SA) + "\t" + 
                str(best_SA) + "\t" + 
                str(target_scribe) + "\t" + 
                str(best_scribe) + "\t" + 
                str(num_matches) + "\t" + 
                str(best_offset) + "\t" + 
                str(scan.fileMetaData.filename) + "\t" + 
                str(scan.metaData.scanID) + "\t" + 
                str(scan.metaData.NCE) + "\t" + 
                scan.peptide.toString() + "\t" + 
                str(scan.metaData.z) + "\n")
            if hits >= 5000: break
        else:
            misses += 1
        #    print("No match", scan.peptide.toString(), str(scan.metaData.z), best_SA, best_scribe, num_matches, flush=True)
            
            #hits+=1
            #if hits>=10: sys.exit()
    