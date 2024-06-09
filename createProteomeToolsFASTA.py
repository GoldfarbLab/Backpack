#!/usr/bin/env/python
import sys
import pandas as pd
import os

seq_infile = pd.read_csv(sys.argv[1], sep="\t")
out_path = sys.argv[2]

for pool in seq_infile["Pool name"].unique():
    pool_seqs = seq_infile[seq_infile["Pool name"] == pool]
    with open(os.path.join(out_path, pool + ".fasta"), 'w') as outfile:
        for seq in pool_seqs["Sequence"]:
            # double C-term
            outfile.write(">" + seq + "_doubleCterm" + "\n")
            outfile.write(seq + seq[-1] + "\n")
            # semi-tryptic
            for i in range(1, len(seq)-4):
                outfile.write(">" + seq + "_semi_" + str(i) + "\n")
                outfile.write(seq[i:] + "\n")
            # single AA deletions
            for i in range(1, len(seq)-1):
                outfile.write(">" + seq + "_del_" + str(i) + "\n")
                outfile.write(seq[0:i]+seq[i+1:] + "\n")
            