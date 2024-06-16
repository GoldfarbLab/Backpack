import pyopenms as oms
import pyopenms.Constants
import msp
from annotation import annotation, annotation_list
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
IMMONIUM_IONS = {'A' : ["IAA", "IAB"],
                 'C' : ["ICA", "ICB", "ICC"],
                 'D' : ["IDA", "IDB", "IDC", "IDD"],
                 'E' : ["IEA", "IEB", "IEC", "IED"],
                 'F' : ["IFA", "IFB", "IFC", "IFD", "IFE", "IFF", "IFG"],
                 'G' : ["IGA"],
                 'H' : ["IHA", "IHB", "IHC", "IHD", "IHE", "IHF", "IHG", "IHH", "IHI", "IHJ", "IHK", "IHL"],
                 'I' : ["IIA", "IIB", "IIC", "IID", "IIE"],
                 'K' : ["IKA", "IKB", "IKC", "IKD", "IKE", "IKF", "IKG", "IKH", "IKI"],
                 'L' : ["ILA", "ILB", "ILC", "ILD", "ILE"],
                 'M' : ["IMA", "IMB", "IMD", "IME"],
                 'N' : ["INA", "INB", "INC", "IND", "INE"],
                 'P' : ["IPA", "IPB", "IPC"],
                 'Q' : ["IQA", "IQB", "IQC", "IQD", "IQE", "IQF"],
                 'R' : ["IRA", "IRB", "IRC", "IRD", "IRE", "IRF", "IRG", "IRH", "IRI", "IRJ", "IRK", "IRL", "IRM", "IRN", "IRO", "IRP"],
                 'S' : ["ISA", "ISB", "ISC"],
                 'T' : ["ITA", "ITB", "ITC", "ITD"],
                 'V' : ["IVA", "IVB", "IVC", "IVD"],
                 'W' : ["IWA", "IWB", "IWC", "IWD", "IWE", "IWF", "IWG", "IWH", "IWI", "IWJ", "IWK", "IWL", "IWM", "IWN", "IWO"],
                 'Y' : ["IYA", "IYB", "IYC", "IYD", "IYE", "IYF", "IYG", "IYH", "IYI"]}

IMMONIUM_ION_FOMRULA = {
                        'ICCAMA' : "C4H8N2OS",  #   ImCCAM (Immonium ion of cysteine carbamidomethylation )
                        'ICCAMB' : "C4H6N2S",   #   ImCCAM - H2O 
                        'ICCAMC' : "C4H5NOS",   #   ImCCAM - NH3  
                        
                        'IMOC' : "C4NOSH9",  #       M with Oxidation modification, also known as IMOC
                        
                        'IAA' : "C2H5N",    #	    44.04948	ImA
                        'IAB' : "C3H5NO",	# 	    72.04439	A-H2O
                        'ICA' : "C2H5NS",	#	    76.02155	ImC
                        'ICB' : "C3H5NOS",  #	    104.0165	C-H2O
                        'ICC' : "C3H3NS",   #	    86.0059	    C-2H2O
                        'IDA' : "C3H5NO2",  #	    88.0393		ImD
                        'IDB' : "C3H3NO",   #	    70.02874	ImD-H2O
                        'IDC' : "C4H5NO3",  #	    116.0342	D-H2O
                        'IDD' : "C4H3NO2",  #	    98.02365	D-2H2O
                        'IEA' : "C4H7NO2",  #	    102.055		ImE
                        'IEB' : "C4H5NO",   #	    84.04439	ImE-H2O
                        'IEC' : "C5H7NO3",  #	    130.0499	E-H2O
                        'IED' : "C5H5NO2",  #	    112.0393	E-2H2O
                        'IFA' : "C8H9N",    #	    120.0808	ImF
                        'IFB' : "C7H6",     #	    91.05423	fF
                        'IFC' : "C8H6",     #	    103.0542	ImF-NH3
                        'IFD' : "C6H4",     # 	    77.03858	fF
                        'IFE' : "C6H6O",    #	    95.04914	fF
                        'IFF' : "C6H4N2",   #	    105.0447	fF
                        'IFG' : "C9H9NO",   #	    148.0757	F-H2O
                        'IFH' : "C9H7N",    #	    130.0651	F-2H2O
                        'IGA' : "CH3N",     #	    30.03383	ImG
                        'IHA' : "C5H7N3",   #	    110.0713	ImH
                        'IHB' : "C4H5N2",   #	    82.05255	ImH-H2O
                        'IHC' : "C6H4N2O",  #	    121.0396	H-H2O
                        'IHD' : "C6H6N2O",  #	    123.0553	fH
                        'IHE' : "C6H7N3O",  #	    138.0662	H-H2O
                        'IHF' : "C6H9N3O2", #	    156.076753	Histidine -- From UniSpec
                        'IHG' : "C5H4N2",   #	    93.04472	ImH-NH3
                        'IHH' : "C6H5N3",   #	    120.0556	H-2H2O
                        'IHI' : "C4H4N2",   #	    81.04472	fH
                        'IHJ' : "C4H6N2",   #	    83.06037	fH
                        'IHK' : "C5H6N2",   #	    95.06037	fH
                        'IHL' : "C7H7N3O2", #	    166.0611	fH
                        'IIA' : "C5H11N",   #		86.09643	ImI
                        'IIB' : "C2H5N",    #		44.04948	fI
                        'IIC' : "C4H9N",    #		72.080776	Fragment of isoleucine side-chain loss -- From UniSpec
                        'IID' : "C6H11NO",  #		114.0913	I-H2O
                        'IIE' : "C6H9N",    #		96.08078	I-2H2O
                        'IKA' : "C5H12N2",  #		101.1073	ImK
                        'IKB' : "C6H9NO",   #		112.0757	K-H2O-NH3
                        'IKC' : "C5H9N",    #		84.08078	ImK-NH3
                        'IKD' : "C6H12N2O", #		129.1022	K-H2O
                        'IKE' : "C3H5N",    #		56.04948	fK
                        'IKF' : "C6H14N4O2",    #		175.118952	Arginine. # seems like the wrong annotation "Arginine" on a lysine
                        'IKG' : "C6H10N2",  #		111.0917	K-2H2O
                        'IKH' : "C4H7N",    #		70.06513	fK
                        'IKI' : "C5H7N",    #		82.06513	fK
                        'ILA' : "C5H11N",   #		86.09643	ImL
                        'ILB' : "C2H5N",    #		44.04948	fL
                        'ILC' : "C4H9N",    #		72.080776	Fragment of leucine side-chain loss -- From UniSpec
                        'ILD' : "C6H9N",    #		96.08078	L-2H2O
                        'ILE' : "C6H11NO",  #		114.0913	L-H2O
                        'IMA' : "C4H9NS",   #		104.0529	ImM
                        'IMB' : "C2H4S",    #		61.01065	fM
                        'IMD' : "C5H9NOS",  #		132.0478	M-H2O
                        'IME' : "C4H8NOS",  #		120.047761	Fragment of methionine side-chain loss -- From UniSpec
                        'INA' : "C3H6N2O",  #		87.05529	ImN
                        'INB' : "C3H3NO",   #		70.02874	ImN-NH3
                        'INC' : "C4H6N2O2", #	    115.0502	N-H2O
                        'IND' : "C4H4N2O",  #		97.03964	N-2H2O
                        'INE' : "C4H3NO2",  #		98.02365	N-H2O-NH3
                        'IPA' : "C4H7N",    #	    70.06513	ImP
                        'IPB' : "C5H7NO",   #	    98.06004	P-H2O
                        'IPC' : "C5H5N",    #	    80.04948	P-2H2O
                        'IQA' : "C4H8N2O",  #	    101.0709	ImQ
                        'IQB' : "C3H5N",    #	    56.04948	fQ
                        'IQC' : "C4H5NO",   #	    84.04439	ImQ-NH3
                        'IQD' : "C5H8N2O2", #	    129.0659	Q-H2O
                        'IQE' : "C5H6N2O",  #	    111.0553	Q-2H2O
                        'IQF' : "C5H5NO2",  #	    112.0393	Q-H2O-NH3
                        'IRA' : "C5H12N4",  #	    129.1135	ImR
                        'IRB' : "C2H6N2",   #	    ?	        -- From MS_Piano
                        'IRC' : "C4H7N",    #	    70.06513	fR
                        'IRD' : "C3H8N2",   #	    ?	        -- From MS_Piano
                        'IRE' : "C4H10N2",  #	    87.09167	fR
                        'IRF' : "C4H9N3",   #	    100.0869	fR
                        'IRG' : "C5H9N3",   #	    112.0869	ImR-NH3
                        'IRH' : "CH5N3",    #	    60.05562	fR
                        'IRI' : "C5H9NO2",  #	    116.0706	fR
                        'IRJ' : "C6H14N4O2",#		175.1189	R
                        'IRK' : "C4H8N2",   #	    85.07602	fR
                        'IRL' : "C5H8N2O",  #	    113.0709	fR
                        'IRM' : "C5H10N2O", #	    115.0866	fR
                        'IRN' : "C6H12N4O", #	    157.1084	R-H2O
                        'IRO' : "C6H10N4",  #	    139.0978	R-2H2O
                        'IRP' : "C6H9N3O",  #	    140.0818	R-H2O-NH3
                        'ISA' : "C2H5NO",   #	    60.04439	ImS
                        'ISB' : "C3H5NO2",  #	    88.0393	    S-H2O
                        'ISC' : "C3H3NO",   #	    70.02874	S-2H2O
                        'ITA' : "C3H7NO",   #	    74.06004	ImT
                        'ITB' : "C4H10N2O2",#	    119.0815	Threonine involved fragment -- From UniSpec
                        'ITC' : "C4H5NO",   #	    84.04439	T-2H2O
                        'ITD' : "C4H7NO2",  #	    102.055	    T-H2O
                        'IVA' : "C4H9N",    #	    72.08078	ImV
                        'IVB' : "C3H4",     #	    41.03858	fV
                        'IVC' : "C4H6",     #	    55.05423	ImV-NH3
                        'IVD' : "C4H4O",    #	    ?	        -- From MS_Piano
                        'IWA' : "C10H10N2", #	    159.0917	ImW
                        'IWB' : "C6H4",     #	    77.03858	fW
                        'IWC' : "C8H6N",    #	    117.0573	fW
                        'IWD' : "C9H7N",    #	    130.0651	fW
                        'IWE' : "C9H9N",    #	    132.0808	fW
                        'IWF' : "C11H7NO",  #	    170.06	    W-H2O-NH3
                        'IWG' : "C7H4",     #	    89.03858	fW
                        'IWH' : "C10H7N",   #	    142.0651	ImW-NH3
                        'IWI' : "C6H4N2",   #	    105.0447	fW
                        'IWJ' : "C9H6",     #	    115.0542	fW
                        'IWK' : "C10H9N",   #	    144.0808	fW
                        'IWL' : "C9H7NO",   #	    146.06	    fW
                        'IWM' : "C11H10N2O",#	    187.0866	W-H2O
                        'IWN' : "C11H8N2",  #	    169.076	    W-2H2O
                        'IWO' : "C11H9NO2", #	    188.0706	W-NH3
                        'IYA' : "C8H9NO",   #	    136.0757	ImY
                        'IYB' : "C7H6",     #	    91.05423	fY
                        'IYC' : "C7H6O",    #	    107.0491	fY
                        'IYD' : "C6H4",     #	    77.03858	fY
                        'IYE' : "C6H6O",    #	    95.04914	fY
                        'IYF' : "C6H4N2",   #	    105.0447	fY
                        'IYG' : "C8H6O",    #	    119.0491	ImY-NH3
                        'IYH' : "C9H9NO2",  #	    164.0706	Y-H2O
                        'IYI' : "C9H7NO",   #	    146.06	    Y-2H2O
                        
                        # Merged identical ions
                        'IXA' : "C4H7NO2",  #       102.055     ITD, IEA
                        'IXB' : "C9H7N",    #       130.0651    IFH, IWD
                        'IXC' : "C6H11NO",  #       114.0913    IID, ILE
                        'IXD' : "C5H11N",   #       86.09643    IIA, ILA
                        'IXE' : "C4H5NO",   #       84.04439    ITC, IQC, IEB
                        'IXF' : "C9H7NO",   #       146.06      IWL, IYI
                        'IXG' : "C4H9N",    #       72.08078    IIC, IVA, ILC
                        'IXH' : "C4H7N",    #       70.06513    IPA, IRC, IKH
                        'IXI' : "C3H5NO2",  #       88.0393     ISB, IDA
                        'IXJ' : "C5H5NO2",  #       112.0393    IQF, IED
                        'IXK' : "C7H6",     #       91.05423    IFB, IYB
                        'IXL' : "C3H3NO",   #       70.02874    IDB, ISC, INB
                        'IXM' : "C3H5N",    #       56.04948    IQB, IKE
                        'IXN' : "C4H3NO2",  #       98.02365    IDD, INE
                        'IXO' : "C6H4N2",   #       105.0447    IWI, IYF, IFF
                        'IXP' : "C6H9N",    #       96.08078    IIE, ILD
                        'IXQ' : "C6H6O",    #       95.04914    IYE, IFE
                        'IXR' : "C6H4",     #       77.03858    IWB, IYD, IFD
                        'IXS' : "C6H14N4O2",     #       175.1189    IKF, IRJ
                        }

class annotator:
    def __init__(self):
        self.match2stats = defaultdict(list)
        for imm in IMMONIUM_ION_FOMRULA:
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
            anno = annotation("y"+str(frag_length), [], [], 0, 0, None)
            terminal_fragments.append([anno.getName(), anno])
            anno = annotation("b"+str(frag_length), [], [], 0, 0, None)
            terminal_fragments.append([anno.getName(), anno])
            
        #if peptide.hasNTerminalModification():
        #    anno = msp_parser.annotation("y"+str(peptide.size()), [], [], 0, 0, None)
        #    terminal_fragments.append([anno.getName(), anno])
        #if peptide.hasCTerminalModification():
        #    anno = msp_parser.annotation("b"+str(peptide.size()), [], [], 0, 0, None)
        #    terminal_fragments.append([anno.getName(), anno])
            
        return terminal_fragments
    
    def generateIntFragments(self, peptide):
        terminal_fragments = []
        for frag_start in range(1, peptide.size()-1):
            for frag_length in range(2, peptide.size()-frag_start):
                anno = annotation("Int_"+str(frag_start)+">"+str(frag_length), [], [], 0, 0, None)
                terminal_fragments.append([anno.getName(), anno])
        return terminal_fragments
                

    def generateImmoniumFragments(self, peptide):
        immonium_fragments = set()
        for aa in peptide:
            if aa.isModified():
                if aa.getModificationName() == "Carbamidomethyl":
                    for imm in ["ICCAMA", "ICCAMB", "ICCAMC"]:
                        immonium_fragments.add(imm)
                elif aa.getModificationName() == "Oxidation" and aa.getOneLetterCode() == "M":
                    immonium_fragments.add("IMOC")
            for imm in IMMONIUM_IONS[aa.getOneLetterCode()]:
                immonium_fragments.add(imm)

        out_fragments = []
        for frag_name in immonium_fragments:
            annot = annotation(frag_name, [], [], 1, 0, None)
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
                scan.mask[index] = "0"
                return True
        elif iso == 0:
            scan.tmp_mask.append([mz, deepcopy(base_annot), "1"])
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
                
                if isFound and config['NLs']:
                    
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
    
                            if not isC2H5SNOFound: break
    
        if config['precursor_ions']:   
            annot = annotation("p", [], [], 0, 0, None)
            annot.z = scan.metaData.z
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
                
                numOxM = self.getModCount(scan.peptide, "Oxidation", "M")
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
                            
                    if not isC2H5SNOFound: break
        
        
        
        if config['internal_ions']:
            internal_fragments = self.generateIntFragments(scan.peptide, max_comp_length=2)
            for frag_name, annot in internal_fragments:
                for z in range(1, 1+min(annot.length, scan.metaData.z)):
                    annot.z = z
                    annot.NL = []
                    
                    isFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                    
                    if isFound and config['NLs']:
                        annot.NL = ["H2O"]
                        isH2OFound = self.scanForIsotopes(scan, annot, error_tol, count_matches)
                        
                        annot.NL = ["CO"]
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
                                    self.scanForIsotopes(scan, annot, error_tol, count_matches)
        
        # update annotation counts by subtracting away ambigous annotations
        subtracted_annots = set()
        for annot_list in scan.annotations:
            if len(annot_list.entries) > 1:
                iso2count = defaultdict(int)
                # are there the same isotopes present?
                for annot in annot_list.entries:
                    iso2count[annot.isotope] += 1
                # subtract the same isotopes:
                for annot in annot_list.entries:
                    annot_name = annot.getName()
                    if annot_name not in subtracted_annots:
                        if iso2count[annot.isotope] > 1:
                            subtracted_annots.add(annot_name)
                            self.match2stats[annot_name][0] -= 1
                            self.match2stats[annot_name][1] -= 1
                        
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
            print(scan.getName(), len(ppms))
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