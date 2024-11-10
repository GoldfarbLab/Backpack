import csv
import sys
import os
from collections import defaultdict
import yaml
import annotation

dict_infile = sys.argv[1]
dict_outpath = sys.argv[2]


with open(os.path.join(os.path.dirname(__file__), "../config/annotator.yaml"), 'r') as stream:
    config = yaml.safe_load(stream)

match2stats = defaultdict(list)

# read dictionary line by line
with open(dict_infile, 'r') as infile:
    reader = csv.reader(infile, delimiter='\t')
    for [ion_name, obs, total] in reader:
        if not ion_name in match2stats:
            match2stats[ion_name] = [0, 0]
        # accumulate counts
        match2stats[ion_name][0] += int(obs)
        match2stats[ion_name][1] += int(total)


with open(dict_outpath, 'w') as outfile:
    writer = csv.writer(outfile, delimiter='\t', quoting=csv.QUOTE_NONE)
    for frag in match2stats:
        obs, total = match2stats[frag]
        annot = annotation.annotation.from_entry(frag, 0)

        #if total == 0: continue
        if obs < config['min_count'] or obs/total < config['min_percent']: continue
        if annot.isotope > 0 and not config['isotopes']: continue
        if (annot.NL or annot.NG) and not config['NLs']: continue
        if annot.name[0] == "p" and not config['precursor_ions']: continue
        if annot.name[0] == "a" and not config['a_ions']: continue
        if annot.name[0] == "b" and not config['b_ions']: continue
        if annot.name[0] == "y" and not config['y_ions']: continue
        if annot.name[0:2] == "Int" and not config['internal_ions']: continue
        if annot.name[0:2] != "Int" and annot.name[0] == "I" and not config['immonium_ions']: continue
        
        writer.writerow([frag, str(obs), str(total), "{:.5f}".format(obs/total)])