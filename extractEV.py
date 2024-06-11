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
                    prog='eV extractor',
                    description='get data for NCE to eV mapping')
parser.add_argument("raw_path")
args = parser.parse_args()


#################################################################################
rawFile = RawFileReaderAdapter.FileFactory(args.raw_path)

rawFile.SelectInstrument(Device.MS, 1)

instrument_model = rawFile.GetInstrumentData().Model
instrument_id = rawFile.GetInstrumentData().SerialNumber
creation_date = rawFile.FileHeader.CreationDate

print(instrument_model, instrument_id, creation_date)

firstScanNumber = rawFile.RunHeaderEx.FirstSpectrum
lastScanNumber = rawFile.RunHeaderEx.LastSpectrum

for scan_id in range(firstScanNumber, lastScanNumber):
    # Get the scan filter for this scan number
    scanFilter = IScanFilter(rawFile.GetFilterForScanNumber(scan_id))

    # Get the scan event for this scan number
    scanEvent = IScanEventBase(rawFile.GetScanEventForScanNumber(scan_id))
    
    # Get the ionizationMode, MS2 precursor mass, collision
    # energy, and isolation width for each scan
    if scanFilter.MSOrder == MSOrderType.Ms2:
        reaction = scanEvent.GetReaction(0)
        collisionEnergy = reaction.CollisionEnergy
        
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
            elif k == "HCD Energy:":
                key2val["NCE"] = v
            elif k == "HCD Energy V:":
                key2val["eV"] = v
        
        filterString = scanFilter.ToString()
        
        # Don't include multiple reactions
        if len(filterString.split()[-2].split("@")) > 2: continue
        key2val["Reaction Type"] = re.findall("[a-zA-Z]+", filterString.split()[-2].split("@")[1])[0]
        if key2val["Reaction Type"] != "hcd": continue
        
        key2val["Isolation Center"] = filterString.split()[-2].split("@")[0]
        
        print(key2val)
        sys.exit()