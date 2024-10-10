import sys
import os
import numpy as np
import pandas as pd
from scipy.interpolate import splrep, BSpline

# read in traces
traces = pd.read_csv("/Users/dennisgoldfarb/Downloads/procal_traces.tsv", sep="\t")
# for each fragment
for peptide in traces.pep.unique():
    peptide_traces = traces[traces.pep == peptide]
    
    # take traces that have at least X IDs (consecutive?)
    fragment_NCE_counts = peptide_traces[peptide_traces.intensity > 0].groupby(['frag', 'NCE']).size().reset_index(name='count')
    fragment_NCE_counts = fragment_NCE_counts[fragment_NCE_counts["count"] >= 3]
    fragment_NCE_counts = fragment_NCE_counts.groupby('frag').size().reset_index(name='count')
    valid_frags = fragment_NCE_counts[fragment_NCE_counts["count"] >= 3]

    peptide_traces_valid = peptide_traces[peptide_traces.frag.isin(valid_frags.frag)]
    # normalize each scan by total
    
    peptide_traces_valid = peptide_traces_valid.groupby(['NCE', 'scan']).intensity / peptide_traces_valid.groupby(['NCE', 'scan']).intensity.max()
    print(peptide_traces_valid)
    sys.exit()
    
    for frag in peptide_traces.frag.unique():
        frag_trace = peptide_traces[peptide_traces.frag == frag]
        
        # add missing values with 0s
        
        # fit the spline
        
        print(peptide, frag)
# plot the spline
# save the file