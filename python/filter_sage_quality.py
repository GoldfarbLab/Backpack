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
from annotation import annotation_list
import scipy

from System import *
from System.Collections.Generic import *

clr.AddReference(os.path.normpath("/RawFileReader/Libs/Net471/ThermoFisher.CommonCore.Data.dll"))
clr.AddReference(os.path.normpath("/RawFileReader/Libs/Net471/ThermoFisher.CommonCore.RawFileReader.dll"))

from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter
from ThermoFisher.CommonCore.Data.Business import Device, GenericDataTypes, SampleType, Scan
from ThermoFisher.CommonCore.Data.FilterEnums import IonizationModeType, MSOrderType
from ThermoFisher.CommonCore.Data.Interfaces import IScanEventBase, IScanFilter

parser = argparse.ArgumentParser(
                    prog='Sage filter',
                    description='Filter Sage results by ID quality metrics')
parser.add_argument("sage_results")
parser.add_argument("raw_path")
parser.add_argument("out_path")
parser.add_argument("--psm_q", default=0.01, type=float)
parser.add_argument("--pep_q", default=0.01, type=float)
parser.add_argument("--post_error", default=-1, type=float)
parser.add_argument("--min_purity", default=0.95, type=float)
parser.add_argument("--min_iso_cs", default=0.99, type=float)
parser.add_argument("--min_z", default=1, type=int)
parser.add_argument("--max_z", default=8, type=int)
parser.add_argument("--min_matched_peaks", default=10, type=int)
parser.add_argument("--ppm_tol", default=20, type=float)
parser.add_argument("--analyzer", default="FTMS")
parser.add_argument("--reaction", default="hcd")
parser.add_argument("--protein_acc_include", default="", type=str)
parser.add_argument("--polarity", default="+", type=str)
args = parser.parse_args()

#################################################################################
with open(os.path.join(os.path.dirname(__file__), "../config/annotator.yaml"), 'r') as stream:
    annot_config = yaml.safe_load(stream)
    



#################################################################################

data = pd.read_csv(args.sage_results, sep="\t")
data = data[data["spectrum_q"] <= args.psm_q]
data = data[data["peptide_q"] <= args.pep_q]
data = data[data["posterior_error"] <= args.post_error]
data = data[data["proteins"].str.contains(args.protein_acc_include)]
data = data[data["label"] == 1]
data = data[data["matched_peaks"] >= args.min_matched_peaks]

data.to_csv(args.sage_results + ".filtered", sep="\t", index=False, quoting=csv.QUOTE_NONE)


#################################################################################
rawFile = RawFileReaderAdapter.FileFactory(args.raw_path)

rawFile.SelectInstrument(Device.MS, 1)
rawFileMetaData = raw_utils.getFileMetaData(rawFile)

print(rawFileMetaData.model, rawFileMetaData.instrument_id, rawFileMetaData.created_date)

annotator = annotator.annotator()

with open(os.path.join(args.out_path, "spectra.msp"), "w") as outfile:
    for index, row in data.iterrows():
        scan_id = int(row["scannr"].split("=")[-1])
        
        # Get the scan filter for this scan number
        scanFilter = IScanFilter(rawFile.GetFilterForScanNumber(scan_id))

        # Get the scan event for this scan number
        scanEvent = IScanEventBase(rawFile.GetScanEventForScanNumber(scan_id))
        
        # Get the ionizationMode, MS2 precursor mass, collision energy, and isolation width for each scan
        peptide = utils.pepFromSage(row["peptide"])
        scan_metaData = raw_utils.getMS2ScanMetaData(rawFile, scan_id, scanEvent, scanFilter, peptide, args.ppm_tol)
        
        if scan_metaData is None: continue
        if scan_metaData.z < args.min_z or scan_metaData.z > args.max_z: continue
        if scan_metaData.reactionType != args.reaction:  continue
        if scan_metaData.purity < args.min_purity: continue
        if scan_metaData.isoFit < args.min_iso_cs: continue
        if scan_metaData.polarity != args.polarity: continue

        spectrum = oms.MSSpectrum()
        centroidStream = rawFile.GetCentroidStream(scan_id, False)
        spectrum.set_peaks([centroidStream.Masses, centroidStream.Intensities])
        annotations = [annotation_list([]) for i in range(spectrum.size())]
        masks = ["?" for i in range(spectrum.size())]
        
        scan = msp.scan(row["peptide"], peptide, rawFileMetaData, scan_metaData, spectrum, annotations, masks)
        
        scan = annotator.annotateScan(scan, error_tol=20.0, count_matches=False, config=annot_config)
        scan = annotator.calibrateSpectrumOLS(scan, 20.0)
        scan.clearAnnotations()
        scan.clearTempMask()
        scan = annotator.annotateScan(scan, error_tol=20.0, count_matches=True, config=annot_config)
        
        scan.writeScan(outfile)
        
        
        