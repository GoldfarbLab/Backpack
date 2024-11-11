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
from pathlib import Path
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

from System import *
from System.Collections.Generic import *

clr.AddReference(os.path.normpath("/RawFileReader/Libs/Net471/ThermoFisher.CommonCore.Data.dll"))
clr.AddReference(os.path.normpath("/RawFileReader/Libs/Net471/ThermoFisher.CommonCore.RawFileReader.dll"))

from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter
from ThermoFisher.CommonCore.Data.Business import Device, GenericDataTypes, SampleType, Scan
from ThermoFisher.CommonCore.Data.FilterEnums import IonizationModeType, MSOrderType
from ThermoFisher.CommonCore.Data.Interfaces import IScanEventBase, IScanFilter

parser = argparse.ArgumentParser(
                    prog='Pioneer filter',
                    description='Filter Pioneer results by ID quality metrics')
parser.add_argument("pioneer_results")
parser.add_argument("raw_path")
parser.add_argument("out_path")
parser.add_argument("--hyperscore", default=30, type=float)
parser.add_argument("--psm_q", default=0.01, type=float)
parser.add_argument("--pep_q", default=0.01, type=float)
parser.add_argument("--post_error", default=-1, type=float)
parser.add_argument("--min_purity", default=0.95, type=float)
parser.add_argument("--min_iso_cs", default=0.99, type=float)
parser.add_argument("--min_iso_target_int", default=0.8, type=float)
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

data = pd.read_csv(args.pioneer_results, sep=",", keep_default_na=False)

print("Post score filter:", len(data.index))

data.to_csv(args.pioneer_results + ".filtered", sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar='\\')

#################################################################################
rawFile = RawFileReaderAdapter.FileFactory(args.raw_path)

print('raw file:', args.raw_path)
rawFile.SelectInstrument(Device.MS, 1)
rawFileMetaData = raw_utils.getFileMetaData(rawFile)

print(rawFileMetaData.model, rawFileMetaData.instrument_id, rawFileMetaData.created_date)

annotator = annotator.annotator()
with open(os.path.join(args.out_path, Path(os.path.basename(args.raw_path)).resolve().stem + ".msp"), "w") as outfile:
    num_target = 0
    num_decoy = 0
    for index, row in data.iterrows():
        scan_id = row["scan_idx"]
        
        # Get the scan filter for this scan number
        scanFilter = IScanFilter(rawFile.GetFilterForScanNumber(scan_id))

        # Get the scan event for this scan number
        scanEvent = IScanEventBase(rawFile.GetScanEventForScanNumber(scan_id))
        
        # Get the ionizationMode, MS2 precursor mass, collision energy, and isolation width for each scan
        peptide, _ = utils.get_mod_seq(row["sequence"], row['structural_mods'])
        scan_metaData = raw_utils.getMS2ScanMetaData(rawFile, scan_id, scanEvent, scanFilter, peptide, args.ppm_tol, pep_z = row['prec_charge'])
        
        if scan_metaData is None: print(scan_id, "no meta data"); continue
        if scan_metaData.reactionType != args.reaction: print(scan_id, "wrong reaction"); continue
        if scan_metaData.analyzer != args.analyzer: print(scan_id, "wrong analyzer", scan_metaData.analyzer); continue
        if scan_metaData.purity < args.min_purity: print(scan_id, "unpure", scan_metaData.purity); continue
        if scan_metaData.isoFit < args.min_iso_cs: print(scan_id, "bad iso fit", scan_metaData.isoFit); continue
        #if scan_metaData.isoTargInt < args.min_iso_target_int: print(scan_id, "bad iso target"); continue
        if scan_metaData.polarity != args.polarity: print(scan_id, "wrong polarity"); continue
        #if scan_metaData.fillTime >= 22: continue
        #if scan_metaData.rawOvFtT < 200000: continue
        
        num_target+=1
        
        print("IDed", scan_id)
        
        spectrum = oms.MSSpectrum()
        centroidStream = rawFile.GetCentroidStream(scan_id, False)
        spectrum.set_peaks([centroidStream.Masses, centroidStream.Intensities])
        annotations = [annotation_list([]) for i in range(spectrum.size())]
        masks = ["?" for i in range(spectrum.size())]
        name = msp.createMSPName(row["modified_sequence"], scan_metaData)
        
        scan = msp.scan(name, peptide, rawFileMetaData, scan_metaData, spectrum, annotations, masks)
        
        scan = annotator.annotateScan(scan, error_tol=20.0, count_matches=False, config=annot_config)
        scan = annotator.calibrateSpectrumOLS(scan, 20.0)
        scan.clearAnnotations()
        scan.clearTempMask()
        scan = annotator.annotateScan(scan, error_tol=20.0, count_matches=True, config=annot_config)

        scan.writeScan(outfile)
        
print("Counts:", num_target, args.raw_path)

        
        