#!/usr/bin/env/python
import sys
import os
import argparse
import time
import re as re
from ftplib import FTP
from datetime import datetime

parser = argparse.ArgumentParser(
                    prog='PRIDE Downloader',
                    description='Downloads PRIDE raw files for a specific project')
parser.add_argument("local_path")
args = parser.parse_args()

local_path = args.local_path
PXD_path = "/pride/data/archive/"

if not os.path.exists(local_path):
    os.makedirs(local_path)

def getProjectRawFiles(PXD):
    for tries in range(5):
        try:
            # loop through files and get list of .raw files
            ftp = connectToPride(os.path.join(PXD_path, year, month, PXD))
            files = []
            ftp.retrlines('NLST', callback=files.append)
            ftp.quit()
            raw_files = [f for f in files if f.endswith(".raw")]
            return raw_files
        except Exception as e:
            print("Failed, trying again. ", tries, e)
        time.sleep(3)
    
    sys.exit()
    return None

def getDirectories(path):
    for tries in range(5):
        try:
            ftp = connectToPride(path)
            dirs = []
            ftp.retrlines('NLST', callback=dirs.append)
            ftp.quit()
            dirs = [d for d in dirs if "." not in d]
            return dirs
        except Exception as e:
            print("Failed, trying again. ", tries, e)
        time.sleep(3)
    
    # loop through files and get list of .raw files
    sys.exit()
    return None

        
# connect, navigate to folder, and download a specific file
def downloadRawFile(local_path, PXD, rawfile):
    ftp = connectToPride(PXD)
    local_filename = os.path.join(local_path, rawfile)
    with open(local_filename, 'wb') as fp:
        ftp.retrbinary("RETR " + rawfile, fp.write)
    return 1

def connectToPride(PXD):
    ftp = FTP('ftp.pride.ebi.ac.uk')
    ftp.login()
    ftp.cwd(PXD)
    return ftp


years = getDirectories(PXD_path)
for year in years:
    months = getDirectories(os.path.join(PXD_path, year))
    for month in months:
        PXDs = getDirectories(os.path.join(PXD_path, year, month))
        for PXD in PXDs:
            raw_files = getProjectRawFiles(PXD)
            print(year, month, PXD, len(raw_files))
            
            
            
        
