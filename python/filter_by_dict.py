#!/usr/bin/env python
import yaml
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import numpy as np
import msp
import csv
import argparse
from collections import defaultdict
from scipy.optimize import nnls
import annotation

parser = argparse.ArgumentParser(
                    prog='Filter by ion dictionary',
                    description='Filter msp files by ion dictionary')
parser.add_argument("dict_path")
parser.add_argument("msp_path")
parser.add_argument("out_path")
args = parser.parse_args()


# Read ion dictionary
match2stats = dict()
with open(args.dict_path, 'r') as infile:
    reader = csv.reader(infile, delimiter='\t')
    for row in reader:
        match2stats[row[0]] = [int(row[1]), int(row[2])]
valid_ions = match2stats.keys()
        
with open(args.out_path, 'w') as outfile:
    for i, scan in enumerate(msp.read_msp_file(args.msp_path)):
        scan.filterAnnotations(valid_ions, False, False)
        scan.writeScan(outfile, True, int_prec=5)
