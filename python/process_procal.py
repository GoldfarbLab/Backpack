#!/usr/bin/env/python
import sys
import os
import pandas as pd
import argparse
import csv
import clr
import re as re
import pyopenms as oms
import utils
import raw_utils
import msp
import annotator
import yaml
from collections import defaultdict
from annotation import annotation_list

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

parser = argparse.ArgumentParser(
                    prog='Procal filter',
                    description='Filter Procal raw files for matching spectra')
parser.add_argument("raw_path")
parser.add_argument("procal_path")
parser.add_argument("out_path")
parser.add_argument("--rt_tol", default=0.5, type=float)
parser.add_argument("--min_purity", default=0.95, type=float)
parser.add_argument("--min_iso_cs", default=0.99, type=float)
parser.add_argument("--min_iso_target_int", default=0.8, type=float)
parser.add_argument("--min_z", default=1, type=int)
parser.add_argument("--max_z", default=8, type=int)
parser.add_argument("--ppm_tol", default=10, type=float)
parser.add_argument("--analyzer", default="FTMS")
parser.add_argument("--reaction", default="hcd")
parser.add_argument("--protein_acc_include", default="", type=str)
parser.add_argument("--polarity", default="+", type=str)
args = parser.parse_args()

#################################################################################
with open(os.path.join(os.path.dirname(__file__), "../config/annotator.yaml"), 'r') as stream:
    annot_config = yaml.safe_load(stream)
    
#################################################################################
rawFile = RawFileReaderAdapter.FileFactory(args.raw_path)

rawFile.SelectInstrument(Device.MS, 1)
rawFileMetaData = raw_utils.getFileMetaData(rawFile)

print(rawFileMetaData.model, rawFileMetaData.instrument_id, rawFileMetaData.created_date)

#################################################################################

# RT Map
model = "QE" if rawFileMetaData.model == "Q Exactive Plus Orbitrap" else "Lumos"
procal_path = os.path.join(args.procal_path, "ProCal_" + model + ".txt")
procal = pd.read_csv(procal_path, sep="\t")


annotator = annotator.annotator()

pep2scans = defaultdict(list)

# Get all valid scans
for scan_id in range(rawFile.RunHeaderEx.FirstSpectrum, rawFile.RunHeaderEx.LastSpectrum):
    # Get the scan filter for this scan number
    scanFilter = IScanFilter(rawFile.GetFilterForScanNumber(scan_id))
    if scanFilter.MSOrder != MSOrderType.Ms2: continue

    # Get the scan event for this scan number
    scanEvent = IScanEventBase(rawFile.GetScanEventForScanNumber(scan_id))
    
    # Determine which peptide this is
    rt = rawFile.RetentionTimeFromScanNumber(scan_id)
    filterString = scanFilter.ToString()
    isoCenter = float(filterString.split()[-2].split("@")[0])
    toleranceMz = utils.ppmToMass(args.ppm_tol, isoCenter)
    
    candidates = procal[(procal["mono_mz"] >= isoCenter - toleranceMz) & (procal["mono_mz"] <= isoCenter + toleranceMz) & 
                        (procal["rt"] >= rt - args.rt_tol) & (procal["rt"] <= rt + args.rt_tol)]
    
    if len(candidates.index) != 1: continue
    
    # Get the ionizationMode, MS2 precursor mass, collision energy, and isolation width for each scan
    pep = candidates["pep"].iloc[0]
    peptide = utils.pepFromSage(pep)
    scan_metaData = raw_utils.getMS2ScanMetaData(rawFile, scan_id, scanEvent, scanFilter, peptide, args.ppm_tol)
    
    if scan_metaData is None: continue
    if scan_metaData.z != candidates["z"].iloc[0]: continue
    if scan_metaData.reactionType != args.reaction:  continue
    if scan_metaData.purity < args.min_purity: continue
    if scan_metaData.isoFit < args.min_iso_cs: continue
    if scan_metaData.isoTargInt < args.min_iso_target_int: continue
    if scan_metaData.polarity != args.polarity: continue
    #if scan_metaData.rawOvFtT < 100000: continue
    if model == "QE" and scan_metaData.fillTime >= 50: continue
    if model == "Lumos" and scan_metaData.fillTime >= 22: continue

    #if pep == "HEHISSDYAGK" and scan_metaData.z == 3: print("Passed")

    spectrum = oms.MSSpectrum()
    centroidStream = rawFile.GetCentroidStream(scan_id, False)
    spectrum.set_peaks([centroidStream.Masses, centroidStream.Intensities])
    annotations = [annotation_list([]) for i in range(spectrum.size())]
    masks = ["?" for i in range(spectrum.size())]
    name = msp.createMSPName(pep, scan_metaData)
    
    scan = msp.scan(name, peptide, rawFileMetaData, scan_metaData, spectrum, annotations, masks)
    
    pep2scans[pep].append(scan)
    

# get top N by rawOvFtT
with open(os.path.join(args.out_path, os.path.basename(args.raw_path) + ".msp"), "w") as outfile:
    for pep in pep2scans:
        #top_scans = sorted(pep2scans[pep], key=lambda x: x.metaData.rawOvFtT, reverse=True)[:min(20, len(pep2scans[pep]))]
        
        #for scan in top_scans:
        for scan in pep2scans[pep]:
            scan = annotator.annotateScan(scan, error_tol=20.0, count_matches=False, config=annot_config)
            scan = annotator.calibrateSpectrumOLS(scan, 20.0)
            scan.clearAnnotations()
            scan.clearTempMask()
            scan = annotator.annotateScan(scan, error_tol=20.0, count_matches=True, config=annot_config)
            
            scan.writeScan(outfile)
        
        
        