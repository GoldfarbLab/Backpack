import sys
import os
import csv
import h5py
import raw_utils
import clr
from pathlib import Path

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


prosit_files = ["/Users/dennisgoldfarb/Downloads/traintest_hcd.hdf5",
                 "/Users/dennisgoldfarb/Downloads/holdout_hcd.hdf5",
                 "/Users/dennisgoldfarb/Downloads/prediction_hcd_train.hdf5",
                 "/Users/dennisgoldfarb/Downloads/prediction_hcd_val.hdf5",
                 "/Users/dennisgoldfarb/Downloads/prediction_hcd_ho.hdf5"
                 ]


filename2NCEs = dict()
for filename in prosit_files:
    with h5py.File(filename, "r") as f:
        nce_arr = f["collision_energy"][()]
        nceAn_arr = f["collision_energy_aligned_normed"][()]
        filename_arr = f["rawfile"][()]
        
        for NCE, NCE_an, filename in zip(nce_arr, nceAn_arr, filename_arr):
            filename2NCEs[filename.decode("utf-8")] = [NCE, 100*NCE_an]
        
        



directories = ["/Volumes/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part1/",
               "/Volumes/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part2/",
               "/Volumes/d.goldfarb/Active/Backpack/raw/ProteomeTools/Part3/"
]

raw_files = set()

with open("/Users/dennisgoldfarb/Downloads/Prosit_NCE_Cal_test.tsv", 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter="\t")
    writer.writerow(["Instrument", "created_date", "NCE", "NCE_A", "NCE_diff", "Cal Date", "filename"])
    
    for directory in directories:
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".raw"):
                raw_path = os.path.join(directory, filename)
                rawFile = RawFileReaderAdapter.FileFactory(raw_path)
                
                rawFile.SelectInstrument(Device.MS, 1)
                NCE_index = -1
                for i, header in enumerate(rawFile.GetTuneDataHeaderInformation()):
                    if header.Label == "NormalizeCollisionEnergy":
                        NCE_index = i
                        
                if NCE_index == -1:
                    print("FAILED!", filename)
                    sys.exit()

                cal_date = rawFile.GetTuneDataValues(0, True).Values[NCE_index]
                 
                rawFileMetaData = raw_utils.getFileMetaData(rawFile)
                filename = Path(filename).stem

                if filename not in filename2NCEs:
                    if "ETD" not in filename: 
                        writer.writerow([rawFileMetaData.instrument_id, rawFileMetaData.created_date, "NA", "NA", "NA", cal_date, filename])
                else:
                    writer.writerow([rawFileMetaData.instrument_id, rawFileMetaData.created_date, float(filename2NCEs[filename][0]), float(filename2NCEs[filename][1]), float(filename2NCEs[filename][0] - filename2NCEs[filename][1]), cal_date, filename])
                    
                raw_files.add(filename)

num_missing = 0
for filename in filename2NCEs:
    if filename not in raw_files:
        print("Missing", filename)
        num_missing += 1
print(num_missing)