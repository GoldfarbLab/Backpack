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
import requests
import time




picke_path = sys.argv[1]
out_path = sys.argv[2]
msp_path = "/scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/train_raw.msp"
#msp_path = "/scratch1/fs1/d.goldfarb/Backpack/ProteomeTools/Part1/01625b_GB4-TUM_first_pool_26_01_01-2xIT_2xHCD-1h-R1/results/annotated/01625b_GB4-TUM_first_pool_26_01_01-2xIT_2xHCD-1h-R1.msp"

library = pickle.load(open(picke_path, 'rb')) 
print(len(library.precursors), flush=True)

def predictKoinaList(peptides, charges, NCEs, iso2eff):
    num_pred = len(peptides)
    
    json_data = {
        'id': '0',
        'inputs': [
            {"name": "peptide_sequences", 
             "shape": [num_pred,1], 
             "datatype": "BYTES", 
             "data": peptides},
            
            {"name": "precursor_charges", 
             "shape": [num_pred,1], 
             "datatype": "INT32",
             "data": charges},
            
            {"name": "collision_energies", 
             "shape": [num_pred,1], 
             "datatype": "FP32", 
             "data": NCEs}
        ]
    } 
    # execute
    retries = 0
    max_retries = 10
    while retries < max_retries:
        try:
            response = requests.post('https://koina.wilhelmlab.org:443/v2/models/Prosit_2020_intensity_HCD/infer', json=json_data)
            # If the request is successful, return the response
            if response.status_code == 200:
                return parseJSON(response.json(), peptides, iso2eff)

            # Log the failed attempt
            print(f"Attempt {retries + 1}: Received status code {response.status_code}.")
        except requests.RequestException as e:
            print(f"Attempt {retries + 1}: Request failed with error: {e}")

        # Wait before retrying
        retries += 1
        time.sleep(1 * retries)
    

def predictKoinaSingle(peptide, z, NCE, offsets, iso2eff):
    
    # create koina peptide sequence
    seq = peptide.toUniModString().upper().replace("(","[").replace(")","]")
    # create request of NCEs
    NCEs = NCE + offsets
    num_pred = NCEs.shape[0]
    # create full curl
    json_data = {
        'id': '0',
        'inputs': [
            {"name": "peptide_sequences", 
             "shape": [num_pred,1], 
             "datatype": "BYTES", 
             "data": [seq] * num_pred},
            
            {"name": "precursor_charges", 
             "shape": [num_pred,1], 
             "datatype": "INT32",
             "data": [z] * num_pred},
            
            {"name": "collision_energies", 
             "shape": [num_pred,1], 
             "datatype": "FP32", 
             "data": NCEs.tolist()}
        ]
    } 
    # execute
    retries = 0
    max_retries = 10
    while retries < max_retries:
        try:
            response = requests.post('https://koina.wilhelmlab.org:443/v2/models/Prosit_2020_intensity_HCD/infer', json=json_data)
            # If the request is successful, return the response
            if response.status_code == 200:
                return parseJSON(response.json(), [peptide.toUniModString()]*num_pred, iso2eff)

            # Log the failed attempt
            print(f"Attempt {retries + 1}: Received status code {response.status_code}.")
        except requests.RequestException as e:
            print(f"Attempt {retries + 1}: Request failed with error: {e}")

        # Wait before retrying
        retries += 1
        time.sleep(1 * retries)
    

def parseJSON(json_in, peptides, iso2eff):
    for output in json_in['outputs']:
        # get shape
        shape = output["shape"]
        # get type
        datatype = output["datatype"]

        if output['name'] == "mz":
            mz = np.array(output['data'], dtype=np.float32).reshape(tuple(shape)) #, dtype=np.float16
        elif output['name'] == "annotation":
            annotations = np.array(output['data']).reshape(tuple(shape))
        elif output['name'] == "intensities":
            intensities = np.array(output['data'], dtype=np.float32).reshape(tuple(shape)) #, dtype=np.float16) 
    
    # annotation version
    # need to do reisotoping here to bin, so need iso2eff
    spectra = []
    for pep_i in range(mz.shape[0]):
        oms_peptide = oms.AASequence.fromString(peptides[pep_i].replace("[","(").replace("]",")"))
        mz_bin2int = defaultdict(float)
        for frag_i in range(mz.shape[1]):
            if intensities[pep_i, frag_i] > 0:
                annot = annotation.annotation.from_entry(annotations[pep_i, frag_i], 0)
                #print(peptides[pep_i], oms_peptide.toString(), annot.getName(), flush=True)
                iso_dist = annot.getTheoreticalIsotopeDistribution(oms_peptide, iso2eff)
                for iso, prob in enumerate(iso_dist):
                    # compute mz
                    iso_mz = mz[pep_i, frag_i] + ((oms.Constants.C13C12_MASSDIFF_U * iso) / annot.z)
                    # compute isotope intensity
                    iso_int = prob * intensities[pep_i, frag_i]
                    # compute mz index
                    mz_bin = int(iso_mz / bin_width)
                    # add to binned spectrum
                    mz_bin2int[mz_bin] += iso_int
        base_peak = max(mz_bin2int.values())
        for k in mz_bin2int:
            mz_bin2int[k] /= base_peak
        
        spectra.append(mz_bin2int)
            
    # loop through each precursor and create entry
    #spectra = []
    #for pep_i in range(mz.shape[0]):
    #    mz_bin2int = defaultdict(float)
    #    for frag_i in range(mz.shape[1]):
    #        if intensities[pep_i, frag_i] > 0:
    #            mz_bin = int(mz[pep_i, frag_i] / bin_width)
                # add to binned spectrum
    #            mz_bin2int[mz_bin] += intensities[pep_i, frag_i]
                
    #    base_peak = max(mz_bin2int.values())
    #    for k in mz_bin2int:
    #        mz_bin2int[k] /= base_peak
        
    #    spectra.append(mz_bin2int)
                
    return spectra


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
        
        # make list of all to predict
        peptides = []
        charges = []
        NCEs = []
        # get precursors in range
        best_offset = 0
        best_SA_offset = 0
        for j, entry in enumerate(library.getPrecursorsInRange(scan.peptide.getMZ(scan.metaData.z), 5, scan_seq)):   
            
            NCE = float(scan.metaData.key2val["NCE_aligned"])
            #NCE = float(scan.metaData.NCE)

            #if j == 0:
                # predict spectra for all NCE
            #    offsets = np.linspace(-10,10,201)
            #    spectra = predictKoinaSingle(scan.peptide, scan.metaData.z, NCE, offsets, scan.metaData.iso2eff)
                # align NCE
            #    for NCE_offset, predicted_spectrum in zip(offsets, spectra):
            #        v1,v2 = spline_library.alignBinnedSpectra(observed_spectrum, predicted_spectrum)
            #        if sum(v1 > 0) <= 3 or sum(v2 > 0) <= 3: continue
                    # LOD
            #        v1 /= max(v1)
            #        LOD = min(v1[v1>0])
                    # compute SA
            #        SA = similarity.scribe(v1,v2,LOD)
            #        if SA > best_SA_offset:
            #            best_SA_offset = SA
            #            best_offset = NCE_offset
            
            best_SA_offset = 1
            best_offset = -0.7  
            peptides.append(entry.peptide)
            charges.append(entry.z)
            NCEs.append(NCE + best_offset)
        
        if best_SA_offset == 0: continue   
        
        # predict full batch
        spectra = predictKoinaList(peptides, charges, NCEs, scan.metaData.iso2eff)
        for j, predicted_spectrum in enumerate(spectra):
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
                print("offset:", hits, target_SA, target_scribe, NCE, best_offset, flush=True)
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
    