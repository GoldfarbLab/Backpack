import json
import csv
import sys
import spline_library
import pickle
import os
import numpy as np


out_path = sys.argv[1]
#precursor_path = "/storage1/fs1/d.goldfarb/Active/RIS_Goldfarb_Lab/NTW/PIONEER/PIONEER_PAPER/SPEC_LIBS/AltimiterOct13Version/Altimeter101324_MixedSpecies_OlsenAstral_NoEntrapment_101324.poin/precursors_for_altimiter.csv"
#json_base_path = "/storage1/fs1/d.goldfarb/Active/Backpack/libraries/astral/Altimeter101324_MixedSpecies_OlsenAstral_NoEntrapment_101324_bright-u10/"
precursor_path = "/scratch1/fs1/d.goldfarb/Backpack/eval/pickle/human.csv"
json_base_path = "/storage1/fs1/d.goldfarb/Active/Backpack/libraries/human/byp_imm5_rawOffset/" # clear-mountain-new


with open(precursor_path) as predict_infile:
    reader = csv.reader(predict_infile, delimiter=",")
    header = next(reader)
    
    library = spline_library.spline_library()

    peptides = []
    charges = []
    species = []
    decoys = []
    batch = 0
    
    for row_i, row in enumerate(reader):
        [upid, acc, seq, mods, z, _, decoy ,_,_,_,_,_,_,_,_,_] = row
        
        species.append(upid)
        charges.append(int(z))
        decoys.append(decoy=="true")
        peptides.append(spline_library.get_mod_seq(seq, mods))
        
        if len(peptides) == 1000:
            json_data = json.load(open(os.path.join(json_base_path, "Altimeter_" + str(batch) + ".json"), 'r'))
            for entry in spline_library.parseJSON(json_data, peptides, charges, species, decoys, library):
                library.append(entry)
            peptides.clear()
            charges.clear()
            species.clear()
            decoys.clear()
            print(batch, flush=True)
            batch += 1
            #if batch > 2000: break
    
    for output in json_data['outputs']:
        if output['name'] == "knots":
            knots = np.concatenate((np.repeat(output['data'][0], 3), output['data'], np.repeat(output['data'][-1], 3)))
            library.knots = knots      
    
    library.sort()


print(len(library.precursors), flush=True)

with open(out_path, 'wb') as file:  
    pickle.dump(library, file) 