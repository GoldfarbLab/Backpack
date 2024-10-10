import sys
import os
import csv
import h5py
from pathlib import Path

hdf_in = "/Users/dennisgoldfarb/Downloads/traintest_hcd.hdf5"

with h5py.File(hdf_in, "r") as f:
    print(f["rawfile"][0])
    print(f["scan_number"][0])
    print(f["intensities_raw"][0])
    print(f["masses_raw"][0])
    #print(list(f.keys()))
    #nce_arr = f["collision_energy"][()]
    #nceAn_arr = f["collision_energy_aligned_normed"][()]
    #filename_arr = f["rawfile"][()]