import raw_utils
import yaml
import os
import pyopenms as oms
from datetime import datetime

with open(os.path.join(os.path.dirname(__file__), "../config/mods.yaml"), 'r') as stream:
    mods_config = yaml.safe_load(stream)









def createMSPName(mod_seq, z, NCE, min_mz, max_mz, LOD, iso2eff):
    msp_seq = modSeqToMSP(mod_seq, z)
    name = "_".join([msp_seq, "NCE" + "{:.2f}".format(NCE), "{:.1f}".format(min_mz) + "-" +"{:.1f}".format(max_mz), "{:.1f}".format(LOD), iso2name(iso2eff)])
    return name

def iso2name(iso2eff):
    return ",".join(["(" +  "{:.2f}".format(iso2eff[iso]) + ")" + str(iso) for iso in iso2eff])

def mods2string(mods):
    out = str(len(mods))
    for mod in mods:
        out += "(" + str(mod[0]) + "," + mod[1] + "," + mod[2] + ")"
    return out
    
def modSeqToMSP(mod_seq, z):
    unmod_seq = ""
    mods = []
    in_mod = False
    # get unmodified sequence
    for i, char in enumerate(mod_seq):
        if char == "-": continue
        if char == "[":
            index = max(0, len(unmod_seq)-1)
            if index > 0:
                AA = unmod_seq[-1]
            else:
                AA = mod_seq[i+1]
            mods.append([index, AA, ""])
            in_mod = True
        elif char == "]":
            in_mod = False
            # convert mass shift to name
            mod_string = mods[-1][1] + "[" + mods[-1][2] + "]"
            if mod_string in mods_config["mods"]:
                mods[-1][2] = mods_config["mods"][mod_string]
            elif "X[" + mods[-1][2] + "]" in mods_config["mods"]:
                mods[-1][2] = mods_config["mods"]["X[" + mods[-1][2] + "]"]
        elif in_mod:
            mods[-1][2] += char
        else:
            unmod_seq += char
    msp_seq = unmod_seq + "/" + str(z) + "_" + mods2string(mods)
    return msp_seq






class scan:
    def __init__(self, mod_seq, pep, RawFileMetaData, MS2MetaData, spectrum, annotations=[], mask=[]):
        self.seq = mod_seq
        self.fileMetaData = RawFileMetaData
        self.metaData = MS2MetaData
        self.spectrum = spectrum
        self.annotations = annotations
        self.mask = mask
        self.peptide = pep
        self.tmp_mask = []
        
    def getMaxIsotope(self):
        return max(self.metaData.iso2eff.keys())
        
    def createComment(self):
        fields = ["scan_id=" + str(self.metaData.scanID),
                  "RawFile=" + self.fileMetaData.filename,
                  "NCE=" + "{:.2f}".format(self.metaData.NCE),
                  "z=" + str(self.metaData.z),
                  "LowMZ=" + "{:.1f}".format(self.metaData.lowMz),
                  "HighMZ=" + "{:.1f}".format(self.metaData.highMz),
                  "IsoWidth=" + "{:.1f}".format(self.metaData.isoWidth),
                  "IsoCenter=" + "{:.4f}".format(self.metaData.isoCenter),
                  "MonoMZ=" + "{:.4f}".format(self.metaData.monoMz),
                  "RT=" + "{:.2f}".format(self.metaData.RT),
                  "RawOvFtT=" + "{:.0f}".format(self.metaData.rawOvFtT),
                  "Purity=" + "{:.2f}".format(self.metaData.purity),
                  "IsoFit=" + "{:.2f}".format(self.metaData.isoFit),
                  "LOD=" + "{:.1f}".format(self.metaData.LOD),
                  "Resolution=" + "{:.0f}".format(self.metaData.resolution),
                  "Analyzer=" + self.metaData.analyzer,
                  "Reaction=" + self.metaData.reactionType,
                  "InstrumentID=" + self.fileMetaData.instrument_id,
                  "SoftwareVersion=" + self.fileMetaData.softwareVersion
        ]
        # Optional
        if hasattr(self.metaData, 'eV'):
            fields.append("eV=" + "{:.2f}".format(self.metaData.eV))
        return " ".join(fields)
        
    def getName(self):
        return createMSPName(self.seq, self.metaData.z, self.metaData.NCE, self.metaData.lowMz, self.metaData.highMz, self.metaData.LOD, self.metaData.iso2eff)
    
    def writeScan(self, outfile, write_unannotated=True, int_prec=1):
        num_peaks = self.spectrum.size() + len(self.tmp_mask)

        # write header
        outfile.write("Name: " + self.getName() + "\n")
        outfile.write("Scan Filter: " + self.metaData.scanFilter + "\n")
        outfile.write("Instrument Model: " + self.fileMetaData.model + "\n")
        outfile.write("Created: " + str(self.fileMetaData.created_date) + "\n")
        outfile.write("Comment: " + self.createComment() + "\n")
        outfile.write("Num peaks: " + str(num_peaks) + "\n")

        # write peaks
        for i, [peak, annotation] in enumerate(zip(self.spectrum, self.annotations)):
            annotation_string = annotation.annotationString()
            if not write_unannotated and annotation_string == "?": continue
            if int_prec == 1:
                outfile.write("{:.4f}".format(peak.getMZ())+ "\t" + "{:.1f}".format(peak.getIntensity()) + "\t" + annotation_string + "\t" + self.mask[i] + '\n')
            else:
                outfile.write("{:.4f}".format(peak.getMZ())+ "\t" + "{:.5f}".format(peak.getIntensity()) + "\t" + annotation_string + "\t" + self.mask[i] + '\n')
          
        # write newly masked   
        for mz, annot, mask in self.tmp_mask:
            outfile.write("{:.4f}".format(mz)+ "\t" + "0" + "\t" + annot.getName() + "\t" + mask + '\n')
        
        outfile.write("\n")
        
    def clearAnnotations(self):
        for annotation_list in self.annotations:
            annotation_list.entries.clear()
            
    def clearMask(self):
        self.mask.clear()
        
    def clearTempMask(self):
        self.tmp_mask.clear()




'''
class scan:
    def __init__(self, RawFileMetaData, MS2MetaData, spectrum, annotations=[], mask=[]):
        self.fileMetaData = RawFileMetaData
        self.metaData = MS2MetaData
        self.spectrum = spectrum
        self.annotations = annotations
        self.mask = mask

        self.LOD = -1
        self.tmp_mask = []       
        self.parseName()
        self.parseComment()
        self.parseIsotopes()
        if self.LOD == -1:
            self.computeLOD()
            
        
    def computeLOD(self):
        _, intensities = self.spectrum.get_peaks()
        intensities = sorted(intensities)
        self.LOD = intensities[0] #int(len(intensities)*0.1)
        
    def getNumAnnotated(self):
        numAnnot = 0
        for annot_list in self.annotations:
            if annot_list.annotationString() != "?":
                numAnnot += 1
        return numAnnot
        
    def parseName(self):
        name_split = self.name.split("_")
        seq = self.name.split("/")[0]
        self.z = int(self.name.split("/")[1].split("_")[0])
        num_mods = int(self.name.split("/")[1].split("_")[1].split("(")[0])
        self.peptide = oms.AASequence.fromString(seq)
        index2mod = defaultdict(list)
        if num_mods > 0:
            self.mod_string = self.name.split("/")[1].split("_")[1][1+len(str(num_mods)):]
            for mod in self.mod_string.split("("):
                mod = mod.strip(")")
                index, _, ptm = mod.split(",")
                index = int(index)
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
            self.peptide = oms.AASequence.fromString(mod_seq)
            self.mod_string = self.name.split("/")[1].split("_")[1]
        else:
            self.mod_string = "0"
        self.mono_mz_pep = self.peptide.getMonoWeight(oms.Residue.ResidueType.Full, self.z) / self.z
        if len(name_split) == 7:
            self.LOD = float(name_split[5])
        
    def parseComment(self):
        self.PEP = 0
        for entry in self.Comment.strip().split(" "):
            if "ms2IsolationWidth" in entry:
                if entry.split("=")[1] == "":
                    if "Kuster" in self.dataset:
                        self.iso_width = 1.3
                        self.Comment = self.Comment.replace("ms2IsolationWidth=", "ms2IsolationWidth=1.3")
                    else:
                        self.iso_width = 0
                else:
                    self.iso_width = float(entry.split("=")[1])
            if "@" in entry:
                self.center = float(entry.split("@")[0].split(" ")[-1])
            if "Sample" in entry:
                self.dataset = entry.split("=")[1]
            if "Purity" in entry:
                self.purity = float(entry.split("=")[1])
            if "[" in entry and "]" in entry and "-" in entry and entry[-1] == '"':
                self.min_mz = float(entry.split("-")[0][1:])
                self.max_mz = float(entry.split("-")[1][0:-2])
            if "PEP=" in entry:
                self.PEP = float(entry.split("=")[1])
                
    def getName(self, is_deisotoped=False, recompute=False):
        if len(self.name.split("_")) == 7 and not recompute: return self.name
        else:
            name = "_".join(self.name.split("_")[0:4])
            name += "_" + "{:.1f}".format(self.min_mz) + "-" + "{:.1f}".format(self.max_mz)
            name += "_" + "{:.5f}".format(self.LOD)
            if is_deisotoped:
                return name + "_(1)0"
            else:
                return name + "_" + ",".join(["(1)"+str(iso) for iso in self.isotopes])
            
    def parseIsotopes(self):
        self.isotopes = getIsolatedIsotopes(self.center, self.iso_width, self.mono_mz_pep, self.z)
        
    def getIsoEfficiency(self):
        iso2efficiency = dict()
        if "NCE" in self.name.split("_")[-1].split(",")[0]:
            for isotope in self.isotopes:
                iso2efficiency[isotope] = 1
        else:
            for iso_entry in self.name.split("_")[-1].split(","):
                isotope = int(iso_entry.split(")")[-1])
                efficiency = float(iso_entry.split(")")[0][1:])
                iso2efficiency[isotope] = efficiency
        return iso2efficiency
        
    def isExpectedTarget(self, tolerancePPM):
        toleranceDa = utils.ppmToMass(tolerancePPM, self.mono_mz_pep)
        diff = (self.center - self.mono_mz_pep) / (oms.Constants.C13C12_MASSDIFF_U / self.z)
        return diff - round(diff) <= toleranceDa and round(diff) >= 0
        
    def isPure(self, theshold):
        return self.purity == -1 or self.purity >= theshold
    
    def isInScanRange(self):
        return True# self.max_mz < self.mono_mz_pep * self.z and (self.max_mz != 2000 or self.max_mz != 4000)
    
    def isValid(self, purityThreshold, tolerancePPM):
        isValidScan = self.isPure(purityThreshold) and self.isExpectedTarget(tolerancePPM) and self.iso_width > 0 and len(self.isotopes) > 0 and self.isInScanRange()
        #if not isValidScan: 
        #    print(self.isPure(purityThreshold), self.isExpectedTarget(tolerancePPM), self.iso_width > 0, len(self.isotopes) > 0)
        return isValidScan
    
    def getBasePeakIntensity(self):
        base_peak_index =  self.spectrum.findHighestInWindow(0, 1e99, 1e99)
        base_peak_intensity = self.spectrum[base_peak_index].getIntensity()
        return base_peak_intensity
    
    def getAnnotatedBasePeakIntensity(self):
        annotated_base_peak = 0
        for i, peak in enumerate(self.spectrum):
            if len(self.annotations[i].entries) > 0 and self.annotations[i].entries[0].toString() != "?": 
                annotated_base_peak = max(annotated_base_peak, peak.getIntensity())
        return annotated_base_peak
    
    def normalizeToBasePeak(self, doLOD=True):
        base_peak_intensity = self.getBasePeakIntensity()
        mzs = []
        intensities = []
        annotated_base_peak = self.getAnnotatedBasePeakIntensity()
        for i, peak in enumerate(self.spectrum):
            mzs.append(peak.getMZ())
            intensities.append(peak.getIntensity() / base_peak_intensity)
        if doLOD:
            self.LOD /= annotated_base_peak
            
        self.spectrum.set_peaks([mzs, intensities])
        
    def clearAnnotations(self):
        for annotation_list in self.annotations:
            annotation_list.entries.clear()
            
    def clearMask(self):
        self.mask.clear()
        
    def clearTempMask(self):
        self.tmp_mask.clear()
        
    def filterAnnotations(self, ion_dictionary, drop_peaks, drop_isotopes):
        #total_intensity_before = 0
        #total_intensity_after = 0
        indices_to_keep = []
        for i, annotation_list in enumerate(self.annotations):
            valid_entries = []
            for entry in annotation_list.entries:
                if drop_isotopes:
                    if entry.getIsoName() in ion_dictionary:
                        valid_entries.append(entry)
                elif entry.getName() in ion_dictionary:
                    valid_entries.append(entry)
            annotation_list.entries = valid_entries
            if len(valid_entries) > 0:
                indices_to_keep.append(i)
                #total_intensity_after += self.spectrum[i].getIntensity()
            #total_intensity_before += self.spectrum[i].getIntensity()
            
        if drop_peaks:
            self.annotations = [self.annotations[i] for i in indices_to_keep]
            self.mask = [self.mask[i] for i in indices_to_keep]
            mzs = [self.spectrum[i].getMZ() for i in indices_to_keep]
            intensities = [self.spectrum[i].getIntensity() for i in indices_to_keep]
            self.spectrum.set_peaks([mzs, intensities])
            
        #percentAnnotated = float(self.Comment.split("percentAnnotated=")[-1]) * (total_intensity_after/total_intensity_before)
        #print(percentAnnotated)

    def writeScan(self, outfile, write_unannotated=True, strip_annotations=False, is_deisotoped=False, write_mask=False, doSort=True, int_prec=1, write_annotComment = True):
        TIC = 0
        TIC_annotated = 0
        num_peaks = self.spectrum.size()
        for peak, annotation in zip(self.spectrum, self.annotations):
            annotation_string = annotation.annotationString(doSort)
            TIC += peak.getIntensity()
            if not write_unannotated :
                if annotation_string == "?": num_peaks -= 1
            if annotation_string != "?": 
                TIC_annotated += peak.getIntensity()
        if TIC > 0:
            percentAnnotated = TIC_annotated/TIC
        else:
            percentAnnotated = -1
        if num_peaks <= 0: return
        if not write_mask: num_peaks -= len([m for m in self.mask if m != 1])
        
        # write header
        outfile.write("Name: " + self.getName(is_deisotoped, recompute=True) + "\n")
        outfile.write("eV: " + self.ev + "\n")
        outfile.write("NCE: " + self.NCE + "\n")
        outfile.write("InstrumentModel: " + self.model + "\n")
        outfile.write("MW: " + self.MW + "\n")
        if write_annotComment:
            outfile.write("Comment: " + self.Comment + " percentAnnotated=" + "{:.4f}".format(percentAnnotated) + "\n")
        else:
            outfile.write("Comment: " + self.Comment + "\n")
        outfile.write("Num peaks: " + str(num_peaks) + "\n")
        
        for i, [peak, annotation] in enumerate(zip(self.spectrum, self.annotations)):
            annotation_string = annotation.annotationString(doSort)
            if not write_unannotated and annotation_string == "?": continue
            if strip_annotations:
                if int_prec == 1:
                    outfile.write("{:.4f}".format(peak.getMZ())+ "\t" + "{:.1f}".format(peak.getIntensity()) + "\n")
                else:
                    outfile.write("{:.4f}".format(peak.getMZ())+ "\t" + "{:.5f}".format(peak.getIntensity()) + "\n")
            else:
                if int_prec == 1:
                    outfile.write("{:.4f}".format(peak.getMZ())+ "\t" + "{:.1f}".format(peak.getIntensity()) + "\t" + annotation_string + "\t" + self.mask[i] + '\n')
                else:
                    outfile.write("{:.4f}".format(peak.getMZ())+ "\t" + "{:.5f}".format(peak.getIntensity()) + "\t" + annotation_string + "\t" + self.mask[i] + '\n')
                
        if write_mask:
            for mz, annot, mask in self.tmp_mask:
                outfile.write("{:.4f}".format(mz)+ "\t" + "0" + "\t" + annot.getName() + "\t" + mask + '\n')
        
        outfile.write("\n")
        '''