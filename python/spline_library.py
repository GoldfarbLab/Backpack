import json
import numpy as np
import pyopenms as oms
from scipy.interpolate import BSpline
import annotation
from collections import defaultdict
import utils
from bisect import bisect_left
import sys

def parseJSON(json_in, peptides, charges, species, decoys, library):

    for output in json_in['outputs']:
        # get shape
        shape = output["shape"]
        # get type
        datatype = output["dataype"]

        if output['name'] == "mz":
            mz = np.array(output['data'], dtype=np.float32).reshape(tuple(shape)) #, dtype=np.float16
        elif output['name'] == "annotations":
            annotations = np.array(output['data']).reshape(tuple(shape))
        elif output['name'] == "coefficients":
            coefficients = np.array(output['data'], dtype=np.float32).reshape(tuple(shape)) #, dtype=np.float16)
            
    
    # loop through each precursor and create entry
    for pep_i in range(mz.shape[0]):

        if "HUMAN" not in species[pep_i] or decoys[pep_i]: continue
        
        mz_out = []
        annot_out = []
        coefs_out = []
        
        for frag_i in range(mz.shape[1]):
            if annotations[pep_i, frag_i] == "NA": continue
            
            frag_name = annotations[pep_i, frag_i]
            if frag_name not in library.name2index:
                index = len(library.name2index)
                library.index2annot[index] = annotation.annotation.from_entry(annotations[pep_i, frag_i], charges[pep_i])
                library.name2index[frag_name] = index
                
            annot_index = library.name2index[frag_name]
            annot = library.index2annot[annot_index]
            if annot.getType() in ["Imm", "p"] : continue
            
            mz_out.append(mz[pep_i, frag_i])
            annot_out.append(annot_index)
            coefs_out.append(coefficients[pep_i, :, frag_i])
            
            if len(mz_out) >= 50: break
            
        yield precursor_spline_entry(peptides[pep_i], charges[pep_i], coefs_out, annot_out, mz_out)



class precursor_spline_entry:
    def __init__(self, peptide, z, coefs, annots, frag_mzs):
        #self.peptide = peptide.toUniModString().upper().replace("(","[").replace(")","]")
        self.peptide = peptide.toString()
        self.z = z
        self.annots = annots
        self.frag_mzs = frag_mzs
        self.coefs = coefs
        self.mz = peptide.getMZ(self.z)
        
    
    def getSpectrum(self, nce, iso2eff, bin_width, library):
        # binned sparse spectrum
        mz_bin2int = defaultdict(float)
        peptide = oms.AASequence.fromString(self.peptide)
        
        for i, frag_id in enumerate(self.annots):
            # evaluate frag splines
            frag = library.index2annot[frag_id]
            coefs = np.concatenate((np.repeat(0, 3), self.coefs[i], np.repeat(0, 3)))
            bs = BSpline(library.knots, coefs, 3, extrapolate=False)
            total_int = bs(nce)
            # compute isotope distribution
            iso_dist = frag.getTheoreticalIsotopeDistribution(peptide, iso2eff)
            for iso, prob in enumerate(iso_dist):
                # compute mz
                iso_mz = self.frag_mzs[i] + ((oms.Constants.C13C12_MASSDIFF_U * iso) / frag.z)
                # compute isotope intensity
                iso_int = prob * total_int
                # compute mz index
                mz_bin = int(iso_mz / bin_width)
                # add to binned spectrum
                mz_bin2int[mz_bin] += iso_int
        
        base_peak = max(mz_bin2int.values())

        for k in mz_bin2int:
            mz_bin2int[k] /= base_peak
        
        return mz_bin2int
    

class spline_library:
    def __init__(self):
        self.precursors = []
        self.knots = []
        self.index2annot = dict()
        self.name2index = dict()
    
    def extend(self, new_precursors):
        self.precursors.extend(new_precursors)
        
    def append(self, new_precursor):
        self.precursors.append(new_precursor)
        
    def sort(self):
        self.precursors.sort(key=lambda x: x.mz)
        
    def getPrecursorsInRange(self, mz, tol_ppm, seq):   
        found = False
        i = self.getIndexByMZ(mz)
        while i < len(self.precursors) and self.precursors[i].mz <= mz+(1e-5):
            if self.precursors[i].peptide == seq:
                found = True
                yield self.precursors[i]
                break
            i+=1
        
        if not found: return
        
        tol_da = utils.ppmToMass(tol_ppm, mz)
        low_mz = mz - tol_da
        high_mz = mz + tol_da
        
        found_i = i
        i = found_i-1
        
        while i > 0 and self.precursors[i].mz >= low_mz:
            yield self.precursors[i]
            i -= 1
            
        i = found_i+1
        while i < len(self.precursors) and self.precursors[i].mz <= high_mz:
            yield self.precursors[i]
            i += 1
        
    def getIndexByMZ(self, mz):
        return bisect_left(self.precursors, mz, key=lambda x: x.mz)
        
    



def binSpectrum(spectrum, bin_width):
    # binned sparse spectrum
    mz_bin2int = defaultdict(float)
        
    for p in spectrum:
        # compute mz index
        mz_bin = int(p.getMZ() / bin_width)
        # add to binned spectrum
        mz_bin2int[mz_bin] += p.getIntensity()
    
    base_peak = max(mz_bin2int.values())

    for k in mz_bin2int:
        mz_bin2int[k] /= base_peak
    
    return mz_bin2int

def alignBinnedSpectra(obs, pred):
    v1 = []
    v2 = []
    
    for mz_bin in pred:
        if mz_bin in obs: 
            v1.append(obs[mz_bin])
            v2.append(pred[mz_bin])
    
    #for mz_bin in pred:
    #    v2.append(pred[mz_bin])
    #    if mz_bin in obs: 
    #        v1.append(obs[mz_bin])
    #    else:
    #        v1.append(0.0)

    return np.array(v1), np.array(v2)


def get_mod_seq(seq, mods):
    mods_oms = mods.replace("Unimod:4", "Carbamidomethyl").replace("Unimod:35", "Oxidation")
    if len(mods_oms) > 0:
        index2mod = defaultdict(list)
        for mod in mods_oms.split("(")[1:]:
            mod = mod.strip(")")
            index, aa, ptm = mod.split(",")
            index = int(index)-1
            index2mod[index].append(ptm)
        mod_seq = ""
        for i, aa in enumerate(seq):
            if i in index2mod:
                for mod in index2mod[i]:
                    if mod == "Acetyl":
                        mod_seq += "(" + mod + ")"
            mod_seq += aa
            if i in index2mod:
                for mod in index2mod[i]:
                    if mod != "Acetyl":
                        mod_seq += "(" + mod + ")"
        peptide = oms.AASequence.fromString(mod_seq)
    else:
        peptide = oms.AASequence.fromString(seq)
                
    return peptide

