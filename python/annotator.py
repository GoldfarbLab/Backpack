import pyopenms as oms
import pyopenms.Constants
import msp
import annotation
import utils
import math
import sys
import numpy as np
import random
import statistics
from copy import deepcopy
from collections import defaultdict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor

AAs = "ACDEFGHIKLMNPQRSTVWY"
TERMINAL_FRAG_TYPES = "aby"

IMMONIUM_ION_FORMULA = {
    'IA' : "C2H5N",
    'IC' : "C2H5NS",
    'IC(Carbamidomethyl)' : "C4H8N2OS",
    'ID' : "C3H5NO2",
    'IE' : "C4H7NO2",
    'IF' : "C8H9N",
    'IG' : "CH3N",
    'IH' : "C5H7N3",
    'II' : "C5H11N",
    'IK' : "C5H12N2",
    'IL' : "C5H11N",
    'IM' : "C4H9NS",
    'IM(Oxidation)' : "C4NOSH9",
    'IN' : "C3H6N2O",
    'IP' : "C4H7N",
    'IQ' : "C4H8N2O",
    'IR' : "C5H12N4",
    'IS' : "C2H5NO",
    'IT' : "C3H7NO",
    'IV' : "C4H9N",
    'IW' : "C10H10N2",
    'IY' : "C8H9NO"
}

IMMONIUM_NLs = {
    'IA' : [([], ["CO"])],
    
    'IC' : [([], ["CO"]), 
            (["H2O"], ["CO"])],
    
    'IC(Carbamidomethyl)' : [(["H2O"], []),
                             (["NH3"], [])],
    
    'ID' : [(["H2O"], []), 
            ([], ["CO"]), 
            (["H2O"], ["CO"])],
    
    'IE' : [(["H2O"], []), 
            ([], ["CO"]), 
            (["H2O"], ["CO"])],
    
    'IF' : [(["CH3N"], []),
            (["NH3"], []),
            (["C2H5N"], []),
            (["C2H3N"], ["O"]),
            (["C2H5"], ["N"]),
            ([], ["CO"]),
            (["H2O"], ["CO"])],
    
    'IG' : [],
    
    'IH' : [(["CH2N"], []),
            (["NH3"], ["CO"]), 
            (["NH"], ["CO"]), 
            ([], ["CO"]), 
            ([], ["H2O", "CO"]),
            (["NH3"], []),
            (["H2O"], ["CO"]),
            (["CH3N"], []),
            (["CHN"], []),
            (["HN"], []),
            ([], ["2CO"])],
    
    'II' : [(["C3H6"], []),
            (["CH2"], []),
            ([], ["CO"]),
            (["H2O"], ["CO"])],
    
    'IK' : [(["NH3"], ["CO"]),
            (["NH3"], []),
            ([], ["CO"]), 
            (["C2H7N"], []),
            ([], ["CO2", "H2N2"]),
            (["H2O"], ["CO"]),
            (["CH5N"], []),
            (["H5N"], [])],
    
    'IL' : [(["C3H6"], []),
            (["CH2"], []),
            (["H2O"], ["CO"]),
            ([], ["CO"])],
    
    'IM' : [(["C2H5N"], []), 
            ([], ["CO"])],
    
    'IM(Oxidation)' : [],
    
    'IN' : [(["NH3"], []), 
            ([], ["CO"]), 
            (["H2O"], ["CO"]), 
            (["NH3"], ["CO"])], 
    
    'IP' : [([], ["CO"]), 
            (["H2O"], ["CO"])], 
    
    'IQ' : [(["CH3NO"], []), 
            (["NH3"], []),
            ([], ["CO"]),
            (["H2O"], ["CO"]), 
            (["NH3"], ["CO"])],
    
    'IR' : [(["C3H6N2"], []), 
            (["CH5N3"], []), 
            (["C2H4N2"], []), 
            (["CH2N2"], []), 
            (["CH3N"], []), 
            (["NH3"], []),
            (["C4H7N"], []),
            (["N3H3"], ["O2"]), 
            ([], ["H2O", "CO"]), 
            (["CH4N2"], []), 
            (["2NH2"], ["O"]), 
            (["H2N2"], ["O"]), 
            ([], ["CO"]), 
            (["H2O"], ["CO"]), 
            (["NH3"], ["CO"])],
    
    'IS' : [([], ["CO"]),  
            (["H2O"], ["CO"])], 
    
    'IT' : [([], ["CO", "NH3"]), 
            (["H2O"], ["CO"]), 
            ([], ["CO"])],
    
    'IV' : [(["CH5N"], []),  
            (["NH3"], []),
            (["CH5N"], ["CO"])],
    
    'IW' : [(["C4H6N2"], []),
            (["C2H4N"], []),
            (["CH3N"], []),
            (["CHN"], []),
            (["NH3"], ["CO"]),
            (["C3H6N2"], []),
            (["NH3"], []),
            (["C4H6"], []),
            (["CH4N2"], []),
            (["HN"], []),
            (["C2H3N"], ["CO"]),
            ([], ["CO"]),
            (["H2O"], ["CO"]),
            (["NH"], ["CO2"])],
    
    'IY' : [(["CH3NO"], []),  
            (["CH3N"], []),
            (["C2H5NO"], []),
            (["C2H3N"], []),
            (["C2H5O"], ["N"]),
            (["NH3"], []),
            ([], ["CO"]),
            (["H2O"], ["CO"])]
}

class annotator:
    def __init__(self):
        self.match2stats = defaultdict(list)
        for imm in IMMONIUM_ION_FORMULA:
            self.match2stats[imm] = [0, 0]
            
    def getModCount(self, peptide, mod, mod_aa):
        mod_count = 0
        for aa in peptide:
            if aa.getOneLetterCode() in mod_aa and aa.isModified() and aa.getModificationName() == mod:
                mod_count += 1
        return mod_count

    
    def generateBYFragments(self, peptide):
        terminal_fragments = []
        for frag_length in range(1, peptide.size()):
            anno = annotation.annotation("y"+str(frag_length), [], [], 0, 0, None)
            terminal_fragments.append([anno.getName(), anno])
            anno = annotation.annotation("b"+str(frag_length), [], [], 0, 0, None)
            terminal_fragments.append([anno.getName(), anno])
            
        #if peptide.hasNTerminalModification():
        #    anno = annotation.annotation("y"+str(peptide.size()), [], [], 0, 0, None)
        #    terminal_fragments.append([anno.getName(), anno])
        #if peptide.hasCTerminalModification():
        #    anno = annotation.annotation("b"+str(peptide.size()), [], [], 0, 0, None)
        #    terminal_fragments.append([anno.getName(), anno])
            
        return terminal_fragments
    
    def generateIntFragments(self, peptide):
        terminal_fragments = []
        for frag_start in range(1, peptide.size()-1):
            for frag_end in range(frag_start+1, peptide.size()-1):
                anno = annotation.annotation("m"+str(frag_start)+":"+str(frag_end), [], [], 0, 0, None)
                terminal_fragments.append([anno.getName(), anno])
        return terminal_fragments
                

    def generateImmoniumFragments(self, peptide):
        immonium_fragments = set()
        for aa in peptide:
            if aa.isModified():
                if aa.getModificationName() == "Carbamidomethyl":
                    immonium_fragments.add('IC(Carbamidomethyl)')
                elif aa.getModificationName() == "Oxidation" and aa.getOneLetterCode() == "M":
                    immonium_fragments.add('IM(Oxidation)')
            immonium_fragments.add("I"+aa.getOneLetterCode())

        out_fragments = []
        for frag_name in immonium_fragments:
            annot = annotation.annotation(frag_name, [], [], 1, 0, None)
            out_fragments.append([frag_name, annot])
            for neutral_pairs in IMMONIUM_NLs[frag_name]:
                (NL, NG) = neutral_pairs
                annot = annotation.annotation(frag_name, NL, NG, 1, 0, None)
                out_fragments.append([frag_name, annot])
        
        return out_fragments
    
    def getMZ(self, mono_mass, z, iso):
        return (mono_mass + (pyopenms.Constants.C13C12_MASSDIFF_U * iso)) / z

    def annotatePeak(self, scan, base_annot, mz, iso, error_tol):
        if scan.metaData.lowMz <= mz and scan.metaData.highMz >= mz:
            index = scan.spectrum.findNearest(mz, utils.ppmToMass(error_tol, mz))

            if index > -1:
                obs_mz = scan.spectrum[index].getMZ()
                error_ppm = utils.getPPM(obs_mz, mz)
                
                annot_iso = deepcopy(base_annot)
                annot_iso.isotope = iso
                annot_iso.error = error_ppm
                
                scan.annotations[index].entries.append(annot_iso)
                scan.mask[index] = 0
                return True
        elif iso == 0:
            scan.tmp_mask.append([mz, deepcopy(base_annot), 1])
        return False
        
    
    def scanForIsotopes(self, scan, base_annot, error_tol, count_matches):
        # Make sure there aren't any negative elemental compositions
        if not base_annot.isValidAnnot(scan.peptide): return False
        annot_name = base_annot.getName()
        if annot_name not in self.match2stats:
            self.match2stats[annot_name] = [0, 0]
        
        main_isos = base_annot.getMostAbundantIsotopes(scan.peptide, scan.metaData.iso2eff, threshold=0.50)
        mono_mass = base_annot.getMonoMass(scan.peptide)
        # check if most abundant iso is present
        for iso in main_isos:
            mz = self.getMZ(mono_mass, base_annot.z, iso)
            isFound = self.annotatePeak(scan, base_annot, mz, iso, error_tol)
            if isFound:
                main_iso = iso
                break
        if isFound:
            # check smaller isotopes
            for iso in range(main_iso-1, -1, -1):
                mz = self.getMZ(mono_mass, base_annot.z, iso)
                if not self.annotatePeak(scan, base_annot, mz, iso, error_tol): break
            # check larger isotopes
            for iso in range(main_iso+1, scan.getMaxIsotope()+1):
                mz = self.getMZ(mono_mass, base_annot.z, iso)
                if not self.annotatePeak(scan, base_annot, mz, iso, error_tol): break
            
            if count_matches:
                self.match2stats[annot_name][0] += 1
        if count_matches:
            self.match2stats[annot_name][1] += 1
        
        return isFound
                
    def annotateScan(self, scan, config, error_tol=30, count_matches=True):
        
        if config['immonium_ions']: 
            imm_fragments = self.generateImmoniumFragments(scan.peptide)
            for frag_name, annot in imm_fragments:
                isFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
            
        by_fragments = self.generateBYFragments(scan.peptide)
        for frag_name, annot in by_fragments:
            for z in range(1, 1+min(annot.length, scan.metaData.z)):
                annot.z = z
                annot.NL = []
                isFound = False; isH2OFound = False; isNH3Found = False; is2H2OFound = False; is2NH3Found = False; isH2O_NH3Found = False
                is2H2O_NH3Found = False; isH2O_2NH3Found = False; is2H2O_2NH3Found = False
                isH2OpCOFound = False; is2H2OpCOFound = False
                
                isFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                
                if config['NLs']:
                    annot.NL = ["H2O"]
                    isH2OFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                    
                    annot.NL = ["NH3"]
                    isNH3Found = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                
                """ if isFound and config['NLs']:
                    
                    if frag_name[0] == "b" or "Int" in frag_name:
                        annot.NL = ["CO"]
                        isCOFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                    
                    if isH2OFound:
                        annot.NL = ["2H2O"]
                        is2H2OFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                        
                        if frag_name[0] == "y" and annot.length < scan.peptide.size():
                            annot.NL = ["H2O"]
                            annot.NG = ["CO"]
                            isH2OpCOFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                            annot.NG.clear()
                            
                            if is2H2OFound and isH2OpCOFound:
                                annot.NL = ["2H2O"]
                                annot.NG = ["CO"]
                                is2H2OpCOFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                                annot.NG.clear()
                        
                    if isNH3Found:
                        annot.NL = ["2NH3"]
                        is2NH3Found = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                        
                    if isH2OFound and isNH3Found:
                        annot.NL = ["H2O", "NH3"]
                        isH2O_NH3Found = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                        
                    if isH2OFound and isNH3Found and is2H2OFound: # Potentially change AND or OR, do we need to see every version or just one?
                        annot.NL = ["2H2O", "NH3"]
                        is2H2O_NH3Found = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                        
                    if isNH3Found and isH2OFound and is2NH3Found: # Potentially change AND or OR
                        annot.NL = ["H2O", "2NH3"]
                        isH2O_2NH3Found = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                        
                    if isNH3Found and isH2OFound and isH2O_2NH3Found and is2H2O_NH3Found: # Potentially change AND or OR
                        annot.NL = ["2H2O", "2NH3"]
                        is2H2O_2NH3Found = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                    
                    if frag_name[0] in "by":
                        if frag_name[0] == "b":
                            numOxM = self.getModCount(scan.peptide.getPrefix(annot.length), "Oxidation", "M")
                            numCarC = self.getModCount(scan.peptide.getPrefix(annot.length), "Carbamidomethyl", "C")
                        elif frag_name[0] == "y":
                            numOxM = self.getModCount(scan.peptide.getSuffix(annot.length), "Oxidation", "M")
                            numCarC = self.getModCount(scan.peptide.getSuffix(annot.length), "Carbamidomethyl", "C")
                        
                        
                        for i in range(1, numOxM+1):
                            if i == 1:
                                annot.NL = ["CH4SO"]
                            else:
                                annot.NL = [str(i)+"CH4SO"]
                            isCH4SOFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                            
                            if isCH4SOFound:
                                annot.NL.append("H2O")
                                isCH4SO_H2OFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                                if isCH4SO_H2OFound:
                                    annot.NL[-1] = "2H2O"
                                    self.scanForIsotopes(scan, annot, error_tol, count_matches)
                                    
                                annot.NL[-1] = "NH3"
                                isCH4SO_NH3Found = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                                if isCH4SO_NH3Found:
                                    annot.NL[-1] = "2NH3"
                                    self.scanForIsotopes(scan, annot, error_tol, count_matches)
                            if not isCH4SOFound: break
                        
                            
                        for i in range(1, numCarC+1):
                            if i == 1:
                                annot.NL = ["C2H5SNO"]
                            else:
                                annot.NL = [str(i)+"C2H5SNO"]
                            isC2H5SNOFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                            
                            if isC2H5SNOFound:
                                annot.NL.append("H2O")
                                isC2H5SNO_H2OFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                                if isC2H5SNO_H2OFound:
                                    annot.NL[-1] = "2H2O"
                                    self.scanForIsotopes(scan, annot, error_tol, count_matches)
                                    
                                annot.NL[-1] = "NH3"
                                isC2H5SNO_NH3Found = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                                if isC2H5SNO_NH3Found:
                                    annot.NL[-1] = "2NH3"
                                    self.scanForIsotopes(scan, annot, error_tol, count_matches)
    
                            if not isC2H5SNOFound: break """
    
        if config['precursor_ions']:   
            annot = annotation.annotation("p", [], [], 0, 0, None)
            for z in range(1, 1+scan.metaData.z):
                annot.z = z
                annot.NL = []
                self.scanForIsotopes(scan, annot, error_tol, count_matches)
                
                #for z in [scan.z]:
                #    annot.z = z
                #    annot.NL = []
                #    self.scanForIsotopes(scan, annot, error_tol, count_matches)
                
                if config['NLs']:  
                    annot.NL = ["H2O"]
                    isH2OFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                    
                    annot.NL = ["NH3"]
                    isNH3Found = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                    
                    """ numOxM = self.getModCount(scan.peptide, "Oxidation", "M")
                    numCarC = self.getModCount(scan.peptide, "Carbamidomethyl", "C")
            
                    for i in range(1,numOxM+1):
                        if i == 1:
                            annot.NL = ["CH4SO"]
                        else:
                            annot.NL = [str(i)+"CH4SO"]
                        isCH4SOFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                        if isCH4SOFound:
                            annot.NL.append("H2O")
                            isCH4SO_H2OFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                            if isCH4SO_H2OFound:
                                annot.NL[-1] = "2H2O"
                                self.scanForIsotopes(scan, annot, error_tol, count_matches)
                                
                            annot.NL[-1] = "NH3"
                            isCH4SO_NH3Found = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                            if isCH4SO_NH3Found:
                                annot.NL[-1] = "2NH3"
                                self.scanForIsotopes(scan, annot, error_tol, count_matches)
                        if not isCH4SOFound: break
                    
                        
                    for i in range(1,numCarC+1):
                        if i == 1:
                            annot.NL = ["C2H5SNO"]
                        else:
                            annot.NL = [str(i)+"C2H5SNO"]
                        isC2H5SNOFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                        
                        if isC2H5SNOFound:
                            annot.NL.append("H2O")
                            isC2H5SNO_H2OFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                            if isC2H5SNO_H2OFound:
                                annot.NL[-1] = "2H2O"
                                self.scanForIsotopes(scan, annot, error_tol, count_matches)
                                
                            annot.NL[-1] = "NH3"
                            isC2H5SNO_NH3Found = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                            if isC2H5SNO_NH3Found:
                                annot.NL[-1] = "2NH3"
                                self.scanForIsotopes(scan, annot, error_tol, count_matches)
                                
                        if not isC2H5SNOFound: break """
        
        
        
        if config['internal_ions']:
            internal_fragments = self.generateIntFragments(scan.peptide)
            for frag_name, annot in internal_fragments:
                for z in range(1, 1+min(annot.length, scan.metaData.z)):
                    annot.z = z
                    annot.NL = []
                    
                    isFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                    
                    if isFound and config['NLs']:
                        annot.NL = ["H2O"]
                        isH2OFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                        
                        """ annot.NL = ["CO"]
                        isCOFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                        
                        if ">" in frag_name:
                            frag_start = int(frag_name[4:].split(">")[0])

                            numOxM = self.getModCount(scan.peptide.getSubsequence(frag_start, annot.length), "Oxidation", "M")
                            numCarC = self.getModCount(scan.peptide.getSubsequence(frag_start, annot.length), "Carbamidomethyl", "C")
                        else:
                            numOxM=frag_name.count("Oxidation")
                            numCarC=frag_name.count("Carbamidomethyl")
                
                        for i in range(1,numOxM+1):
                            if i == 1:
                                annot.NL = ["CH4SO"]
                            else:
                                annot.NL = [str(i)+"CH4SO"]
                            isCH4SOFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                            if isCH4SOFound:
                                annot.NL.append("H2O")
                                isCH4SO_H2OFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                                if isCH4SO_H2OFound:
                                    annot.NL[-1] = "2H2O"
                                    self.scanForIsotopes(scan, annot, error_tol, count_matches)
                            if not isCH4SOFound: break
                        
                            
                        for i in range(1,numCarC+1):
                            if i == 1:
                                annot.NL = ["C2H5SNO"]
                            else:
                                annot.NL = [str(i)+"C2H5SNO"]
                            isC2H5SNOFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)

                            if not isC2H5SNOFound: break
                            
                            if isC2H5SNOFound:
                                annot.NL.append("H2O")
                                isC2H5SNO_H2OFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                                if isC2H5SNO_H2OFound:
                                    annot.NL[-1] = "2H2O"
                                    self.scanForIsotopes(scan, annot, error_tol, count_matches)
                                    
                                annot.NL[-1] = "NH3"
                                isC2H5SNO_NH3Found = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                                if isC2H5SNO_NH3Found:
                                    annot.NL[-1] = "2NH3"
                                    self.scanForIsotopes(scan, annot, error_tol, count_matches) """
        
        # update annotation counts by subtracting away ambigous annotations
        #if count_matches:
        #    subtracted_annots = set()
        #    for annot_list in scan.annotations:
        #        if len(annot_list.entries) > 1:
        #            iso2count = defaultdict(int)
        #            # are there the same isotopes present?
        #            for annot in annot_list.entries:
        #                iso2count[annot.isotope] += 1
        #            # subtract the same isotopes:
        #            for annot in annot_list.entries:
        #                annot_name = annot.getName()
        #                if annot_name not in subtracted_annots:
        #                    if iso2count[annot.isotope] > 1:
        #                        subtracted_annots.add(annot_name)
        #                        self.match2stats[annot_name][0] -= 1
        #                        self.match2stats[annot_name][1] -= 1
                        
        return scan
    
        
    def calibrateSpectrumOLS(self, scan, eps):
        # get median error
        ppms = []
        for i, annot_list in enumerate(scan.annotations):
            mz = scan.spectrum[i].getMZ()
            int = scan.spectrum[i].getIntensity()
            if (int <= 0): continue # skip masked ions
            for annot in annot_list.entries:
                mz_theo = annot.getMZ(scan.peptide)
                ppm = utils.getPPM(mz, mz_theo)
                ppms.append(ppm)
        if len(ppms) < 3: return self.calibrateSpectrumMedian(scan, eps)
        eps_offset = statistics.median(ppms)
        
        # make all pairs within eps
        x = []
        y = []
        ppms = []
        for i, annot_list in enumerate(scan.annotations):
            mz = scan.spectrum[i].getMZ()
            int = scan.spectrum[i].getIntensity()
            if (int <= 0): continue # skip masked ions
            
            for annot in annot_list.entries:
                mz_theo = annot.getMZ(scan.peptide)
                ppm = utils.getPPM(mz, mz_theo)
                ppms.append(ppm)
                
                if abs(ppm - eps_offset) <= eps:
                    x.append(mz)
                    y.append(mz_theo)
                    #print(mz, mz_theo, ppm, annot.toString())
                    
        if len(x) < 4:
            print(scan.name, len(ppms))
            return self.calibrateSpectrumMedian(scan, eps)
    
        # least squares
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)

        model = make_pipeline(PolynomialFeatures(2), RANSACRegressor(random_state=42))
        model.fit(x, y)
        
        mzs, intensities = scan.spectrum.get_peaks()
    
        mzs = model.predict(np.array(mzs).reshape(-1, 1)).tolist()
        
        scan.spectrum.set_peaks([mzs, intensities])
        return scan
    
    def calibrateSpectrumMedian(self, scan, eps):
        # get median error
        ppms = []
        #abs_diffs = []
        for i, annot_list in enumerate(scan.annotations):
            mz = scan.spectrum[i].getMZ()
            int = scan.spectrum[i].getIntensity()
            if (int <= 0): continue # skip masked ions
            for annot in annot_list.entries:
                mz_theo = annot.getMZ(scan.peptide)
                #abs_diff = mz - mz_theo
                ppm = utils.getPPM(mz, mz_theo)
                ppms.append(ppm)
                #abs_diffs.append(abs_diff)
                
        if len(ppms) == 0: return scan
        eps_offset = statistics.median(ppms)
        #abs_offset = statistics.median(abs_diffs)
        
        mzs, intensities = scan.spectrum.get_peaks()
        for i, annot_list in enumerate(scan.annotations):
            abs_offset = utils.ppmToMass(eps_offset, scan.spectrum[i].getMZ())
            mz = scan.spectrum[i].getMZ() - abs_offset
            if (intensities[i] <= 0): continue # skip masked ions
            mzs[i] = mz
        
        scan.spectrum.set_peaks([mzs, intensities])
        return scan
    
    
    
    
    
    
    
    
    