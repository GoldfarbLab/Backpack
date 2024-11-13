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
                    prog='Sage filter',
                    description='Filter Sage results by ID quality metrics')
parser.add_argument("sage_results")
parser.add_argument("chrono_path")
parser.add_argument("raw_path")
parser.add_argument("out_path")
parser.add_argument("dict_path")
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

data = pd.read_csv(args.sage_results, sep="\t")
data = data[data["spectrum_q"] <= args.psm_q]
data = data[data["peptide_q"] <= args.pep_q]
data = data[data["posterior_error"] <= args.post_error]
accs = args.protein_acc_include.split(" ")
filtered_data = []
for acc in accs:
    filtered_data.append(data[data["proteins"].str.contains(acc)])
data = pd.concat(filtered_data).drop_duplicates().reset_index(drop=True)
#data = data[data["label"] == 1]
data = data[data["matched_peaks"] >= args.min_matched_peaks]
data = data[data["hyperscore"] >= args.hyperscore]

print("Post score filter:", len(data.index))

chrono = pd.read_csv(args.chrono_path, sep="\t").drop(['CodedPeptideSeq', 'PeptideLength'], axis=1)

data = data.merge(chrono, left_on='peptide', right_on='PeptideModSeq', how='inner')

print("Post inner join:", len(data.index))

data = data.sort_values('rt')
y_points = np.array(data['Pred_HI'])
x_points = np.array(data['rt'])

model = interpolate.LSQUnivariateSpline(x_points, y_points, np.linspace(np.min(x_points), np.max(x_points), 7)[1:6], k=3)
deviations = np.abs(y_points - model(x_points))
mad = np.median(deviations)

x_points = x_points[deviations <= 10*mad]
y_points = y_points[deviations <= 10*mad]

model = interpolate.LSQUnivariateSpline(x_points, y_points, np.linspace(np.min(x_points), np.max(x_points), 7)[1:6], k=3)

deviations = np.abs(y_points - model(x_points))
mad = np.median(deviations)

print("Post RT filter:", len(data.index))
data = data[np.abs(model(data['rt']) - data['Pred_HI']) <= 3*mad]

data.to_csv(args.sage_results + ".filtered", sep="\t", index=False, quoting=csv.QUOTE_NONE)

print("Pre:", sum(data["label"] == 1), sum(data["label"] == -1))
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
        scan_id = int(row["scannr"].split("=")[-1])
        
        # Get the scan filter for this scan number
        scanFilter = IScanFilter(rawFile.GetFilterForScanNumber(scan_id))

        # Get the scan event for this scan number
        scanEvent = IScanEventBase(rawFile.GetScanEventForScanNumber(scan_id))
        
        # Get the ionizationMode, MS2 precursor mass, collision energy, and isolation width for each scan
        peptide = utils.pepFromSage(row["peptide"])
        scan_metaData = raw_utils.getMS2ScanMetaData(rawFile, scan_id, scanEvent, scanFilter, peptide, args.ppm_tol)
        
        if scan_metaData is None: print(scan_id, "no meta data"); continue
        if scan_metaData.reactionType != args.reaction: print(scan_id, "wrong reaction"); continue
        if scan_metaData.analyzer != args.analyzer: print(scan_id, "wrong analyzer", scan_metaData.analyzer); continue
        if scan_metaData.z < args.min_z or scan_metaData.z > args.max_z: print(scan_id, "wrong charge"); continue
        
        #if not (scan_metaData.z == 2 and int(row["charge"])) < 2: print(scan_id, "same charge"); continue 
        if scan_metaData.z != int(row["charge"]): print(scan_id, "different charge"); continue 
        if scan_metaData.purity < args.min_purity: print(scan_id, "unpure", scan_metaData.purity); continue
        if scan_metaData.isoFit < args.min_iso_cs: print(scan_id, "bad iso fit"); continue
        if scan_metaData.isoTargInt < args.min_iso_target_int: print(scan_id, "bad iso target"); continue
        if scan_metaData.polarity != args.polarity: print(scan_id, "wrong polarity"); continue
        #if scan_metaData.fillTime >= 22: continue
        #if scan_metaData.rawOvFtT < 200000: continue
        
        if row["label"] == 1:
            num_target+=1
        else: 
            num_decoy+=1
            continue
        
        print("IDed", scan_id)
        
        spectrum = oms.MSSpectrum()
        centroidStream = rawFile.GetCentroidStream(scan_id, False)
        spectrum.set_peaks([centroidStream.Masses, centroidStream.Intensities])
        annotations = [annotation_list([]) for i in range(spectrum.size())]
        masks = ["?" for i in range(spectrum.size())]
        name = msp.createMSPName(row["peptide"], scan_metaData)
        
        scan = msp.scan(name, peptide, rawFileMetaData, scan_metaData, spectrum, annotations, masks)
        
        scan = annotator.annotateScan(scan, error_tol=20.0, count_matches=False, config=annot_config)
        scan = annotator.calibrateSpectrumOLS(scan, 20.0)
        scan.clearAnnotations()
        scan.clearTempMask()
        scan = annotator.annotateScan(scan, error_tol=20.0, count_matches=True, config=annot_config)

        scan.writeScan(outfile)
        
print("Counts:", num_target, num_decoy, args.raw_path)


with open(os.path.join(args.dict_path, "ion_dictionary.txt"), 'w') as outfile:
        writer = csv.writer(outfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for frag in annotator.match2stats:
            obs, total = annotator.match2stats[frag]
            writer.writerow([frag, str(obs), str(total)])
    
        
        