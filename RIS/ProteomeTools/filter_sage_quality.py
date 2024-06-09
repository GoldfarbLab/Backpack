#!/usr/bin/env/python
import sys
import pandas as pd
import argparse
import csv

parser = argparse.ArgumentParser(
                    prog='Sage filter',
                    description='Filter Sage results by ID quality metrics')
parser.add_argument("sage_results")
parser.add_argument("-m", "--psm_q", default=0.01, type=float)
parser.add_argument("-p", "--pep_q", default=0.01, type=float)
parser.add_argument("-e", "--post_error", default=-1, type=float)
args = parser.parse_args()

print(args.psm_q, args.pep_q, args.post_error)

data = pd.read_csv(args.sage_results, sep="\t")
data = data[data["spectrum_q"] <= args.psm_q]
data = data[data["peptide_q"] <= args.pep_q]
data = data[data["posterior_error"] <= args.post_error]
data = data[data["proteins"].str.contains("pt|")]

data.to_csv(args.sage_results + ".filtered", sep="\t", index=False, quoting=csv.QUOTE_NONE)