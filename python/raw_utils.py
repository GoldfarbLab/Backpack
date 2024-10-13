import utils
from bisect import bisect_left
import numpy as np
import os
import clr
import pyopenms as oms
import re as re
from datetime import datetime
from msp import RawFileMetaData, MS2MetaData

from System import *
from System.Collections.Generic import *

clr.AddReference(os.path.join(os.path.dirname(__file__),"../libs/ThermoFisher.CommonCore.Data.dll"))
clr.AddReference(os.path.join(os.path.dirname(__file__),"../libs/ThermoFisher.CommonCore.RawFileReader.dll"))
#clr.AddReference(os.path.normpath("/RawFileReader/Libs/Net471/ThermoFisher.CommonCore.Data.dll"))
#clr.AddReference(os.path.normpath("/RawFileReader/Libs/Net471/ThermoFisher.CommonCore.RawFileReader.dll"))

from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter
from ThermoFisher.CommonCore.Data.Business import Device, GenericDataTypes, SampleType, Scan
from ThermoFisher.CommonCore.Data.FilterEnums import IonizationModeType, MSOrderType
from ThermoFisher.CommonCore.Data.Interfaces import IScanEventBase, IScanFilter



#class IdentificationData:
#    def __init__(self):
        



def getFileMetaData(rawFile):
    key2val = dict()
    key2val["Instrument Model"] = rawFile.GetInstrumentData().Model
    key2val["InstrumentID"] = rawFile.GetInstrumentData().SerialNumber
    key2val["Created"] = rawFile.FileHeader.CreationDate
    key2val["RawFile"] = os.path.basename(rawFile.FileName)
    key2val["SoftwareVersion"] = rawFile.GetInstrumentData().SoftwareVersion
    
    return RawFileMetaData(key2val)


def getMS2ScanMetaData(rawFile, scan_id, scanEvent, scanFilter, peptide, ppm_tol):
    key2val = dict()
    
    reaction = scanEvent.GetReaction(0)
    collisionEnergy = reaction.CollisionEnergy
    isolationWidth = reaction.IsolationWidth
    rt = rawFile.RetentionTimeFromScanNumber(scan_id)
    
    key2val["scan_id"] = scan_id
    key2val["NCE"] = collisionEnergy
    key2val["IsoWidth"] = isolationWidth
    key2val["RT"] = rt
    
    # Get the trailer extra data for this scan and then look
    # for the monoisotopic m/z value in the trailer extra data
    # list
    trailerData = rawFile.GetTrailerExtraInformation(scan_id)
    
    for i in range(trailerData.Length):
        k = trailerData.Labels[i]
        v = trailerData.Values[i]
        if k == "Charge State:":
            key2val["z"] = v.strip()
        elif k == "Orbitrap Resolution:":
            key2val["Resolution"] = v.strip()
        elif k == "FT Resolution:":
            key2val["Resolution"] = v.strip()
        elif k == "RawOvFtT:":
            key2val["RawOvFtT"] = v.strip()
        elif k == 'Monoisotopic M/Z:':
            key2val["MonoMZ"] = v.strip()
            #if float(v) <= 0: print("no monoMz"); return None
        elif k == "HCD Energy eV:":
            key2val["eV"] = v.strip()
        elif k == "HCD Energy V:":
            key2val["eV"] = v.strip()
        elif k == "Ion Injection Time (ms):":
            key2val["fillTime"] = float(v.strip())
    
    filterString = scanFilter.ToString()
    
    # Don't include multiple reactions
    if len(filterString.split()[-2].split("@")) > 2: print("multiple reactions"); return None
    
    key2val["Polarity"] = "+" if any(["+" == f for f in filterString.split()]) else "-"   
    key2val["Reaction"] = re.findall("[a-zA-Z]+", filterString.split()[-2].split("@")[1])[0]
    key2val["Analyzer"] = filterString.split()[0]
    key2val["IsoCenter"] = filterString.split()[-2].split("@")[0]
    key2val["NCE"] = re.split("[a-zA-Z]+", filterString.split()[-2].split("@")[1])[1]
    key2val["LowMZ"] = filterString.split()[-1].split("-")[0][1:]
    key2val["HighMZ"] = filterString.split()[-1].split("-")[1][0:-1]
    key2val["Scan Filter"] = filterString
    
    if float(key2val["MonoMZ"]) <= 0: 
        key2val["MonoMZ"] = key2val["IsoCenter"]
        key2val["z"] = str(int(float(key2val["HighMZ"]) / float(key2val["MonoMZ"])))
        
    if key2val["Analyzer"] == "ITMS": return MS2MetaData(key2val)
    
    abundance, purity = getPurity(rawFile, scan_id, rt, isolationWidth, float(key2val["IsoCenter"]), float(key2val["MonoMZ"]), int(key2val["z"]), ppm_tol)
    key2val["Purity"] = purity
    key2val["Abundance"] = abundance
    isoFit, isoTargInt = getIsotopeStats(rawFile, scan_id, rt, isolationWidth, float(key2val["IsoCenter"]), float(key2val["MonoMZ"]), int(key2val["z"]), peptide, ppm_tol)
    key2val["IsoFit"] = isoFit
    key2val["IsoTargInt"] = isoTargInt
    key2val["LOD"] = getLOD(rawFile, scan_id)

    return MS2MetaData(key2val)



def getIsotopeStats(rawFile, scan_id, rt, iso_width, iso_center, target_mz, target_z, peptide, ppm_tol): 
    isoFit = getIsotopeFit(rawFile, scan_id, rt, iso_width, iso_center, target_mz, target_z, peptide, ppm_tol)
    isCentered, iso = isExpectedTarget(iso_center, target_mz, target_z, ppm_tol)
    isoTargInt = getTargetIsoAbundance(iso, 10, peptide) if isCentered else 0
    return isoFit, isoTargInt
        


def getTargetIsoAbundance(iso_center, max_iso, peptide):
    # compute expected isotope distribution
    pep_iso_gen = oms.CoarseIsotopePatternGenerator(max_iso)
    pep_formula = peptide.getFormula()
    theo_iso_dist = utils.isoDist2np(pep_formula.getIsotopeDistribution(pep_iso_gen))
    theo_iso_dist /= theo_iso_dist.max()
    return theo_iso_dist[iso_center] if iso_center < len(theo_iso_dist) else 0.0

def isExpectedTarget(iso_center, target_mz, target_z, tolerancePPM):
    toleranceDa = utils.ppmToMass(tolerancePPM, target_mz)
    diff = (iso_center - target_mz) / (oms.Constants.C13C12_MASSDIFF_U / target_z)
    return diff - round(diff) <= toleranceDa and round(diff) >= 0, round(diff)

def getIsotopeFit(rawFile, scan_id, rt, iso_width, iso_center, target_mz, target_z, peptide, ppm_tol): 
    # find left MS1
    MS1_scan_id_left = findLeftMS1(rawFile, scan_id)
    fit_left = getFitAtMS1(rawFile, iso_width, iso_center, MS1_scan_id_left, target_mz, target_z, peptide, ppm_tol) if MS1_scan_id_left > -1 else -1
    rt_left = rawFile.RetentionTimeFromScanNumber(MS1_scan_id_left) if MS1_scan_id_left > -1 else -1
    
    # find right MS1
    MS1_scan_id_right = findRightMS1(rawFile, scan_id)
    fit_right = getFitAtMS1(rawFile, iso_width, iso_center, MS1_scan_id_right, target_mz, target_z, peptide, ppm_tol)  if MS1_scan_id_right > -1 else -1
    rt_right = rawFile.RetentionTimeFromScanNumber(MS1_scan_id_right) if MS1_scan_id_right > -1 else -1
    
    iso_fit = utils.linearInterpolate(rt, rt_left, fit_left, rt_right, fit_right)
    
    return iso_fit

def getPurity(rawFile, scan_id, rt, iso_width, iso_center, target_mz, target_z, ppm_tol): 
    # find left MS1
    MS1_scan_id_left = findLeftMS1(rawFile, scan_id)
    abundance_left, purity_left = getPurityAtMS1(rawFile, iso_width, iso_center, MS1_scan_id_left, target_mz, target_z, ppm_tol) if MS1_scan_id_left > -1 else [0, 0]
    rt_left = rawFile.RetentionTimeFromScanNumber(MS1_scan_id_left) if MS1_scan_id_left > -1 else -1
    
    # find right MS1
    MS1_scan_id_right = findRightMS1(rawFile, scan_id)
    abundance_right, purity_right = getPurityAtMS1(rawFile, iso_width, iso_center, MS1_scan_id_right, target_mz, target_z, ppm_tol) if MS1_scan_id_right > -1 else [0, 0]
    rt_right = rawFile.RetentionTimeFromScanNumber(MS1_scan_id_right) if MS1_scan_id_right > -1 else -1
    
    purity = utils.linearInterpolate(rt, rt_left, purity_left, rt_right, purity_right)
    abundance = utils.linearInterpolate(rt, rt_left, abundance_left, rt_right, abundance_right)
    
    return abundance, purity

def findLeftMS1(rawFile, scan_id):
    for MS1_scan_id in range(scan_id, rawFile.RunHeaderEx.FirstSpectrum, -1):
        scanFilter = IScanFilter(rawFile.GetFilterForScanNumber(MS1_scan_id))
        if scanFilter.MSOrder == MSOrderType.Ms:
            return MS1_scan_id
    return -1

def findRightMS1(rawFile, scan_id):
    for MS1_scan_id in range(scan_id, rawFile.RunHeaderEx.LastSpectrum):
        scanFilter = IScanFilter(rawFile.GetFilterForScanNumber(MS1_scan_id))
        if scanFilter.MSOrder == MSOrderType.Ms:
            return MS1_scan_id
    return -1

def getFitAtMS1(rawFile, iso_width, iso_center, scan_id, target_mz, target_z, peptide, ppm_tol):
    centroidStream = rawFile.GetCentroidStream(scan_id, False)
    target_int = []
    masked_iso_indices = []
    min_mz = iso_center - iso_width/2
    max_mz = iso_center + iso_width/2
    next_iso = target_mz
    start_i = bisect_left(centroidStream.Masses, min_mz)
    for i in range(start_i, centroidStream.Length):
        if centroidStream.Masses[i] >= min_mz and centroidStream.Masses[i] <= max_mz:
            # find closest match to isotope
            while utils.getPPMAbs(centroidStream.Masses[i], next_iso) > ppm_tol and centroidStream.Masses[i] > next_iso:
                next_iso += oms.Constants.C13C12_MASSDIFF_U / target_z
                target_int.append(0)
                masked_iso_indices.append(len(target_int)-1)
            
            if utils.getPPMAbs(centroidStream.Masses[i], next_iso) <= ppm_tol:
                if i < centroidStream.Length-1 and utils.getPPMAbs(centroidStream.Masses[i+1], next_iso) < utils.getPPMAbs(centroidStream.Masses[i], next_iso):
                    continue
                target_int.append(centroidStream.Intensities[i])
                next_iso += oms.Constants.C13C12_MASSDIFF_U / target_z
        if centroidStream.Masses[i] > max_mz: break
        
    if len(target_int) == 0: return -1
    
    # compute expected isotope distribution
    pep_iso_gen = oms.CoarseIsotopePatternGenerator(len(target_int))
    pep_formula = peptide.getFormula()
    theo_iso_dist = utils.isoDist2np(pep_formula.getIsotopeDistribution(pep_iso_gen))
    for i in masked_iso_indices:
        theo_iso_dist[i] = 0
    exp_iso_dist = np.array(target_int)

    cs = utils.cosineSim(theo_iso_dist, exp_iso_dist)
         
    return cs

def getPurityAtMS1(rawFile, iso_width, iso_center, scan_id, target_mz, target_z, ppm_tol):
    target_int = getTargetIntensityInWindow(rawFile, scan_id, iso_width, iso_center, target_mz, target_z, ppm_tol)
    total_int = getIntensityInWindow(rawFile, scan_id, iso_width, iso_center)
    if total_int == 0: return 0, 0
    else: return target_int, target_int / total_int

def getIntensityInWindow(rawFile, scan_id, iso_width, iso_center):
    centroidStream = rawFile.GetCentroidStream(scan_id, False)
    total_int = 0
    min_mz = iso_center - iso_width/2
    max_mz = iso_center + iso_width/2
    start_i = bisect_left(centroidStream.Masses, min_mz)
    for i in range(start_i, centroidStream.Length):
        if centroidStream.Masses[i] >= min_mz and centroidStream.Masses[i] <= max_mz:
            total_int += centroidStream.Intensities[i]
        if centroidStream.Masses[i] > max_mz: break
    return total_int

def getTargetIntensityInWindow(rawFile, scan_id, iso_width, iso_center, target_mz, target_z, ppm_tol):
    centroidStream = rawFile.GetCentroidStream(scan_id, False)
    target_int = 0
    min_mz = iso_center - iso_width/2
    max_mz = iso_center + iso_width/2
    next_iso = target_mz
    start_i = bisect_left(centroidStream.Masses, min_mz)
    for i in range(start_i, centroidStream.Length):
        if centroidStream.Masses[i] >= min_mz and centroidStream.Masses[i] <= max_mz:
            # find closest match to isotope
            while utils.getPPMAbs(centroidStream.Masses[i], next_iso) > ppm_tol and centroidStream.Masses[i] > next_iso:
                next_iso += oms.Constants.C13C12_MASSDIFF_U / target_z
            
            if utils.getPPMAbs(centroidStream.Masses[i], next_iso) <= ppm_tol:
                if i < centroidStream.Length-1 and utils.getPPMAbs(centroidStream.Masses[i+1], next_iso) < utils.getPPMAbs(centroidStream.Masses[i], next_iso):
                    continue
                target_int += centroidStream.Intensities[i]
                next_iso += oms.Constants.C13C12_MASSDIFF_U / target_z
        if centroidStream.Masses[i] > max_mz: break
                
    return target_int

def getLOD(rawFile, scan_id):
    centroidStream = rawFile.GetCentroidStream(scan_id, False)
    return min(centroidStream.Intensities)