import yaml
import os
import pyopenms as oms
from datetime import datetime
from collections import defaultdict
import annotation
import numpy as np

with open(os.path.join(os.path.dirname(__file__), "../config/mods.yaml"), 'r') as stream:
    mods_config = yaml.safe_load(stream)









def createMSPName(mod_seq, metaData):
    msp_seq = modSeqToMSP(mod_seq, metaData.z)
    NCE = metaData.key2val["NCE_aligned"] if "NCE_aligned" in metaData.key2val else metaData.NCE
    name = "_".join([msp_seq, "NCE" + "{:.2f}".format(NCE), "{:.1f}".format(metaData.lowMz) + "-" +"{:.1f}".format(metaData.highMz), "{:.7f}".format(metaData.LOD)])
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





class RawFileMetaData:
    def __init__(self, key2val):
        self.key2val = dict()
        for k,v in key2val.items():
            if k == "Instrument Model":
                self.model = v
            elif k == "InstrumentID":
                self.instrument_id = v
            elif k == "Created":
                self.created_date = v
            elif k == "RawFile":
                self.filename = v
            elif k == "SoftwareVersion":
                self.softwareVersion = v
            else: # Remaining optional meta data
                self.key2val[k]=v
        
class MS2MetaData:
    def __init__(self, key2val):
        self.key2val = dict()
        for k,v in key2val.items():
            if k == "scan_id":
                self.scanID = int(v)
            elif k == "NCE":
                self.NCE = float(v)
            elif k == "IsoWidth":
                self.isoWidth = float(v)
            elif k == "IsoCenter":
                self.isoCenter = float(v)
            elif k == "RT":
                self.RT = float(v)
            elif k == "z":
                self.z = int(v)
            elif k == "Resolution":
                self.resolution = float(v)
            elif k == "RawOvFtT":
                self.rawOvFtT = float(v)
            elif k == "MonoMZ":
                self.monoMz = float(v)
            elif k == "Reaction":
                self.reactionType = v
            elif k == "Analyzer":
                self.analyzer = v
            elif k == "LowMZ":
                self.lowMz = float(v)
            elif k == "HighMZ":
                self.highMz = float(v)
            elif k == "Scan Filter":
                self.scanFilter = v
            elif k == "Purity":
                self.purity = float(v)
            elif k == "Abundance":
                self.abundance = float(v)
            elif k == "IsoFit":
                self.isoFit = float(v)
            elif k == "IsoTargInt":
                self.isoTargInt = float(v)
            elif k == "LOD":
                self.LOD = float(v)
            elif k == "Polarity":
                self.polarity = v
            elif k == "fillTime":
                self.fillTime = float(v)
            else: # Remaining optional meta data
                self.key2val[k]=v
        #print(key2val)
        if self.isoCenter != 0:
            self.iso2eff = getUniformIsoEfficiency(getIsolatedIsotopes(self.isoCenter, self.isoWidth, self.monoMz, self.z))
        else:
            self.iso2eff = set([0])
            
            
def getIsolatedIsotopes(center, width, mono, z):
    isotopes = set()
    min_mz = center - width/2
    max_mz = center + width/2
    
    for i in range(5):
        iso_mz = mono + (i * oms.Constants.C13C12_MASSDIFF_U) / z
        if min_mz <= iso_mz <= max_mz:
            isotopes.add(i)
    return isotopes

def getUniformIsoEfficiency(isotopes):
    iso2eff = dict()
    for iso in isotopes:
        iso2eff[iso] = 1
    return iso2eff

class scan:
    def __init__(self, name, pep, RawFileMetaData, MS2MetaData, spectrum, annotations=[], mask=[]):
        self.name = name
        self.fileMetaData = RawFileMetaData
        self.metaData = MS2MetaData
        self.spectrum = spectrum
        self.annotations = annotations
        self.mask = mask
        self.peptide = pep
        self.tmp_mask = []
        
        
    def getModString(self):
        num_mods = int(self.name.split("/")[1].split("_")[1].split("(")[0])
        if num_mods > 0:
            return self.name.split("/")[1].split("_")[1]
        return "0"
        
    def updateMSPName(self):
        NCE = self.metaData.key2val["NCE_aligned"] if "NCE_aligned" in self.metaData.key2val else self.metaData.NCE
        split_name = self.name.split("_")
        name_l = "_".join(split_name[0:2])
        name_r = "_".join(split_name[3:])
        self.name = name_l +  "_NCE" + "{:.2f}".format(NCE) + "_" + name_r
        #self.name = "_".join([self.peptide.toString()+"/"+ str(self.metaData.z),  "NCE" + "{:.2f}".format(NCE), "{:.1f}".format(self.metaData.lowMz) + "-" +"{:.1f}".format(self.metaData.highMz), "{:.7f}".format(self.metaData.LOD), iso2name(self.metaData.iso2eff)])

    def getLOD(self):
        return float(self.name.split("_")[4])
    
    def getMaxIsotope(self):
        return max(self.metaData.iso2eff.keys())
    
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
    
    def getTotalAnnotatedIntensity(self):
        annotated_intensity = 0
        for i, peak in enumerate(self.spectrum):
            if len(self.annotations[i].entries) > 0 and self.annotations[i].entries[0].toString() != "?": 
                annotated_intensity += peak.getIntensity()
        return annotated_intensity
    
    def getTotalIntensity(self):
        total_intensity = 0
        for i, peak in enumerate(self.spectrum):
            total_intensity += peak.getIntensity()
        return total_intensity
    
    def normalizeToBasePeak(self, doLOD=True):
        base_peak_intensity = self.getBasePeakIntensity()
        mzs = []
        intensities = []
        annotated_base_peak = self.getAnnotatedBasePeakIntensity()
        for i, peak in enumerate(self.spectrum):
            mzs.append(peak.getMZ())
            intensities.append(peak.getIntensity() / base_peak_intensity)
        if doLOD:
            self.metaData.LOD /= annotated_base_peak
            name_split = self.name.split("_")
            name_split[4] = "{:.7f}".format(self.metaData.LOD)
            self.name = "_".join(name_split)
        self.spectrum.set_peaks([mzs, intensities])
        
    def updateLOD(self):
        min_annotated = 1e9
        for i, peak in enumerate(self.spectrum):
            if len(self.annotations[i].entries) > 0 and self.annotations[i].entries[0].toString() != "?": 
                min_annotated = min(peak.getIntensity(), min_annotated)
        self.metaData.LOD = min_annotated
        name_split = self.name.split("_")
        name_split[4] = "{:.7f}".format(self.metaData.LOD)
        self.name = "_".join(name_split)
        
    def normalizeToTotalAnnotated(self, doLOD=False):
        total_annotated_intensity = self.getTotalAnnotatedIntensity()
        mzs = []
        intensities = []
        for i, peak in enumerate(self.spectrum):
            mzs.append(peak.getMZ())
            intensities.append(peak.getIntensity() / total_annotated_intensity)
        if doLOD:
            self.metaData.LOD /= total_annotated_intensity
            name_split = self.name.split("_")
            name_split[4] = "{:.7f}".format(self.metaData.LOD)
            self.name = "_".join(name_split)
        self.spectrum.set_peaks([mzs, intensities])
        
    def normalizeToTotal(self):
        total_intensity = self.getTotalIntensity()
        mzs = []
        intensities = []
        for i, peak in enumerate(self.spectrum):
            mzs.append(peak.getMZ())
            intensities.append(peak.getIntensity() / total_intensity)
        self.spectrum.set_peaks([mzs, intensities])
        
    def getAnnotationsByMask(self, mask_criteria):
        return [self.annotations[i] for i,m in enumerate(self.mask) if m in mask_criteria and self.annotations[i].annotationName() != "?"]
    
    def getAnnotationIntensity(self, frag):
        for i, annot_list in enumerate(self.annotations):
            for annot in annot_list.entries:
                if annot.getName() == frag:
                    return self.spectrum[i].getIntensity()
        return 0
        
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
                  "Purity=" + "{:.5f}".format(self.metaData.purity),
                  "Abundance=" + "{:.2f}".format(self.metaData.abundance),
                  "IsoFit=" + "{:.2f}".format(self.metaData.isoFit),
                  "IsoTargInt=" + "{:.2f}".format(self.metaData.isoTargInt),
                  "LOD=" + "{:.7f}".format(self.metaData.LOD),
                  "Resolution=" + "{:.0f}".format(self.metaData.resolution),
                  "Analyzer=" + self.metaData.analyzer,
                  "Reaction=" + self.metaData.reactionType,
                  "SoftwareVersion=" + self.fileMetaData.softwareVersion
        ]
        # Optional
        if 'eV' in self.metaData.key2val:
            #fields.append("eV=" + "{:.2f}".format(self.metaData.key2val['eV']))
            fields.append("eV=" + self.metaData.key2val['eV'])
        if 'NCE_aligned' in self.metaData.key2val:
            if isinstance(self.metaData.key2val['NCE_aligned'], float):
                fields.append("NCE_aligned=" + "{:.2f}".format(self.metaData.key2val['NCE_aligned']))
            else:
                fields.append("NCE_aligned=" + self.metaData.key2val['NCE_aligned'])
        return " ".join(fields) 
    
    def writeScan(self, outfile, write_unannotated=True, int_prec=1):
        num_peaks = self.spectrum.size() + len(self.tmp_mask)

        # write header
        outfile.write("Name: " + self.name + "\n")
        outfile.write("Scan Filter: " + self.metaData.scanFilter + "\n")
        outfile.write("Instrument Model: " + self.fileMetaData.model + "\n")
        outfile.write("Instrument ID: " + self.fileMetaData.instrument_id + "\n")
        outfile.write("Created: " + str(self.fileMetaData.created_date) + "\n")
        outfile.write("Comment: " + self.createComment() + "\n")
        outfile.write("Num peaks: " + str(num_peaks) + "\n")

        # write peaks
        for i, [peak, annotation] in enumerate(zip(self.spectrum, self.annotations)):
            annotation_string = annotation.annotationString()
            if not write_unannotated and annotation_string == "?": continue
            if int_prec == 1:
                outfile.write("{:.4f}".format(peak.getMZ())+ "\t" + "{:.1f}".format(peak.getIntensity()) + "\t" + annotation_string + "\t" + str(self.mask[i]) + '\n')
            else:
                outfile.write("{:.4f}".format(peak.getMZ())+ "\t" + "{:.7f}".format(peak.getIntensity()) + "\t" + annotation_string + "\t" + str(self.mask[i]) + '\n')
          
        # write newly masked   
        for mz, annot, mask in self.tmp_mask:
            outfile.write("{:.4f}".format(mz)+ "\t" + "0" + "\t" + annot.getName() + "\t" + str(mask) + '\n')
        
        outfile.write("\n")
        
    def clearAnnotations(self):
        for annotation_list in self.annotations:
            annotation_list.entries.clear()
            
    def clearMask(self):
        self.mask.clear()
        
    def clearTempMask(self):
        self.tmp_mask.clear()
        
    def filterAnnotations(self, ion_dictionary, drop_peaks, drop_isotopes):
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
            
        if drop_peaks:
            self.annotations = [self.annotations[i] for i in indices_to_keep]
            self.mask = [self.mask[i] for i in indices_to_keep]
            mzs = [self.spectrum[i].getMZ() for i in indices_to_keep]
            intensities = [self.spectrum[i].getIntensity() for i in indices_to_keep]
            self.spectrum.set_peaks([mzs, intensities])



def pepFromMSPName(name):
    seq = name.split("/")[0]
    num_mods = int(name.split("/")[1].split("_")[1].split("(")[0])
    peptide = oms.AASequence.fromString(seq)
    index2mod = defaultdict(list)
    if num_mods > 0:
        mod_string = name.split("/")[1].split("_")[1][1+len(str(num_mods)):]
        for mod in mod_string.split("("):
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
                        if mod[0] in ["+", "-"]:
                            mod_seq += "[" + mod + "]"
                        else:
                            mod_seq += "(" + mod + ")"
        peptide = oms.AASequence.fromString(mod_seq)

    return peptide


def parseComment(comment):
    key2val = dict()
    for entry in comment.split(" "):
        [k,v] = entry.split("=")
        key2val[k] = v
    return key2val

def read_msp_file(path):
    with open(path,'r') as f:
        mzs = []
        intensities = []
        annotations = []
        mask = []
        spectrum = oms.MSSpectrum()
        key2val = dict()
        for line in f:
            if line.startswith("Name:"):
                key2val["Name"] = line.strip()[len("Name: "):]
            elif line.startswith("Scan Filter:"):
                key2val["Scan Filter"] = line.strip()[len("Scan Filter: "):]
                key2val["Polarity"] = "+" if any(["+" == f for f in key2val["Scan Filter"].split()]) else "-"  
            elif line.startswith("Instrument Model:"):
                key2val["Instrument Model"] = line.strip()[len("Instrument Model: "):]
            elif line.startswith("Instrument ID:"):
                key2val["InstrumentID"] = line.strip()[len("Instrument ID: "):]
            elif line.startswith("Created:"):
                key2val["Created"] = line.strip()[len("Created: "):]
            elif line.startswith("Comment:"):
                key2val.update(parseComment(line.strip()[len("Comment: "):]))
                metaData = MS2MetaData(key2val)
                fileMetaData = RawFileMetaData(key2val)
            elif line.startswith("Num peaks:"):
                pass
            elif len(line.strip()) == 0:
                # Create scan
                if len(key2val) == 0: continue
                
                # sort everything
                sort_mz = np.array(mzs).argsort() 
                mzs = [mzs[i] for i in sort_mz]
                intensities = [intensities[i] for i in sort_mz]
                annotations = [annotations[i] for i in sort_mz]
                mask = [mask[i] for i in sort_mz]
                
                spectrum.set_peaks([mzs, intensities])
                peptide = pepFromMSPName(key2val["Name"])
                yield scan(key2val["Name"], peptide, fileMetaData, metaData, spectrum, annotations=annotations, mask=mask)
                # reset values
                key2val = dict()
                spectrum = oms.MSSpectrum()
                mzs = []
                intensities = []
                annotations = []
                mask = []
            else:
                # Peaks
                split_peak = line.strip().split("\t")
                mz = float(split_peak[0])
                intensity = float(split_peak[1])
                
                annotations.append(annotation.annotation_list([annotation.annotation.from_entry(anno, metaData.z) for anno in split_peak[2].strip('"').split(",")]))
                mask.append(int(split_peak[3]) if split_peak[3] != "?" else -1)

                mzs.append(mz)
                intensities.append(intensity)
                
        #if len(key2val) > 0:
        #    print("yielding last", key2val)
        #    spectrum.set_peaks([mzs, intensities])
        #    yield scan(key2val["Name"], peptide, fileMetaData, metaData, spectrum, annotations=[], mask=[])