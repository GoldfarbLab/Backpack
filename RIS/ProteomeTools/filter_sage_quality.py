#!/usr/bin/env/python
import sys
import os
import pandas as pd
import argparse
import csv
import clr
import re as re

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
parser.add_argument("-m", "--psm_q", default=0.01, type=float)
parser.add_argument("-p", "--pep_q", default=0.01, type=float)
parser.add_argument("-e", "--post_error", default=-1, type=float)
args = parser.parse_args()


def NCE2eV(nce, mz, charge):
    cf = 1.0
    if charge==2: cf=0.9
    elif charge==3: cf=0.85
    elif charge==4: cf=0.8
    elif charge>=5: cf=0.75
    ev = nce*mz*500*cf
    return ev


#################################################################################

data = pd.read_csv(args.sage_results, sep="\t")
data = data[data["spectrum_q"] <= args.psm_q]
data = data[data["peptide_q"] <= args.pep_q]
data = data[data["posterior_error"] <= args.post_error]
data = data[data["proteins"].str.contains("pt\|")]
data = data[data["label"] == 1]

data.to_csv(args.sage_results + ".filtered", sep="\t", index=False, quoting=csv.QUOTE_NONE)


#################################################################################
rawFile = RawFileReaderAdapter.FileFactory(args.raw_path)

rawFile.SelectInstrument(Device.MS, 1)

instrument_model = rawFile.GetInstrumentData().Model
instrument_id = rawFile.GetInstrumentData().SerialNumber

print(instrument_model, instrument_id)

for index, row in data.iterrows():
    scan_id = int(row["scannr"].split("=")[-1])
    
    # Get the scan filter for this scan number
    scanFilter = IScanFilter(rawFile.GetFilterForScanNumber(scan_id))

    # Get the scan event for this scan number
    scanEvent = IScanEventBase(rawFile.GetScanEventForScanNumber(scan_id))
    
    # Get the ionizationMode, MS2 precursor mass, collision
    # energy, and isolation width for each scan
    if scanFilter.MSOrder == MSOrderType.Ms2:
        reaction = scanEvent.GetReaction(0)
        collisionEnergy = reaction.CollisionEnergy
        isolationWidth = reaction.IsolationWidth
        
        # Get the trailer extra data for this scan and then look
        # for the monoisotopic m/z value in the trailer extra data
        # list
        trailerData = rawFile.GetTrailerExtraInformation(scan_id)
        
        key2val = dict()
        for i in range(trailerData.Length):
            k = trailerData.Labels[i]
            v = trailerData.Values[i]
            if k == "Charge State:":
                key2val["z"] = v
                if int(v) < 1 or int(v) > 8: continue
            elif k == "Orbitrap Resolution:":
                key2val["resolution"] = v
            elif k == "HCD Energy:":
                key2val["NCE"] = v
            elif k == "RawOvFtT:":
                key2val["RawOvFtT"] = v
        
        filterString = scanFilter.ToString()
        
        # Don't include multiple reactions
        if len(filterString.split()[-2].split("@")) > 2: continue
        key2val["Reaction Type"] = re.findall("[a-zA-Z]+", filterString.split()[-2].split("@")[1])[0]
        if key2val["Reaction Type"] != "hcd": continue
        
        key2val["Analyzer"] = filterString.split()[0]
        key2val["Isolation Center"] = filterString.split()[-2].split("@")[0]
        key2val["NCE"] = re.split("[a-zA-Z]+", filterString.split()[-2].split("@")[1])[1]
        key2val["LowMz"] = filterString.split()[-1].split("-")[0][1:]
        key2val["HighMz"] = filterString.split()[-1].split("-")[1][0:-1]
        key2val["Scan Filter"] = filterString
        key2val["eV"] = NCE2eV(float(key2val["NCE"]), float(key2val["Isolation Center"]), int(key2val["z"]))
        
        print(key2val)
        sys.exit()