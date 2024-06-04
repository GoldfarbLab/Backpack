#!/usr/bin/env/python
import sys
import os
import re as re
from ftplib import FTP
from datetime import datetime

#local_path = "/Users/denisgoldfarb/Downloads/Altimeter/data/ProteomeTools/Part2/"
#PXD = "/pride/data/archive/2019/05/PXD010595/"

local_path = sys.argv[1]
PXD = sys.argv[2]

if not os.path.exists(local_path):
    os.makedirs(local_path)


def getProjectRawFiles(PXD):
    ftp = connectToPride(PXD)
    # loop through files and get list of .raw files
    files = []
    ftp.retrlines('NLST', callback=files.append)
    ftp.quit()
    raw_files = [f for f in files if ".raw" in f]
    return raw_files
        
# connect, navigate to folder, and download a specific file
def downloadRawFile(local_path, PXD, rawfile):
    ftp = connectToPride(PXD)
    local_filename = os.path.join(local_path, rawfile)
    with open(local_filename, 'wb') as fp:
        ftp.retrbinary("RETR " + rawfile, fp.write)
    ftp.quit()

def connectToPride(PXD):
    ftp = FTP('ftp.pride.ebi.ac.uk')
    ftp.login()
    ftp.cwd(PXD)
    return ftp



raw_files = getProjectRawFiles(PXD)
print("Raw files found:", len(raw_files))
for i, f in enumerate(raw_files):
    start=datetime.now()
    print("Downloading:", f, i)
    downloadRawFile(local_path, PXD, f)
    print("Finished", datetime.now()-start)
