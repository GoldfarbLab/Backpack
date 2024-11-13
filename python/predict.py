import torch
import utils
import utils_unispec
import msp
import yaml
import os
import sys
import numpy as np
import csv
import pyopenms as oms
import matplotlib.pyplot as plt
from annotation import annotation
from collections import defaultdict
import argparse
import json
from lightning_model import LitFlipyFlopy
import pprint
plt.close('all')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(
                    prog='Prediction',
                    description='Prediction fragmentation for a list of peptides')
parser.add_argument("job_id", type=int)
parser.add_argument("num_jobs", type=int)
args = parser.parse_args()

with open(os.path.join(os.path.dirname(__file__), "../config/predict.yaml"), 'r') as stream:
    config = yaml.safe_load(stream)
with open(os.path.join(os.path.dirname(__file__), "../config/mods.yaml"), 'r') as stream:
    mod_config = yaml.safe_load(stream)


# Instantiate DicObj
with open(config['dic_config'], 'r') as stream:
    dconfig = yaml.safe_load(stream)

D = utils_unispec.DicObj(dconfig['ion_dictionary_path'], dconfig['seq_len'], dconfig['chlim'])
L = utils_unispec.LoadObj(D, embed=True)

# Instantiate model
with open(config['model_config']) as stream:
    model_config = yaml.safe_load(stream)
    model = LitFlipyFlopy.load_from_checkpoint(config['model_ckpt'], config=config, model_config=model_config)
    model.eval()


pred_ions = np.array([D.index2ion[ind] for ind in range(len(D.index2ion))])

def rename_mods(pep, mod_string):
    if "+" in mod_string:
        for mass in mod_config["mods2"]:
            mod_string = mod_string.replace(mass, mod_config["mods2"][mass])
        mod_string = mod_string.replace("+", pep[0])
    return mod_string

def flatten(xss):
    return [x for xs in xss for x in xs]


def writeJSON_batch(pred, info, peptides, corrected_mods_list, batch_idx):
    outfile = open(config['out_file'] + '_' + str(batch_idx) + ".json", 'w', encoding="utf-8")
    
    
    frag_mzs = np.zeros_like(pred)
    
    for precursor_i in range(pred.shape[0]):
        seq, _, charge, nce, min_mz, max_mz, LOD, iso2efficiency, weight = info[precursor_i]
        prec_info = (seq, corrected_mods_list[precursor_i], charge, nce, min_mz, max_mz, LOD, iso2efficiency, weight) 
        filt = L.filter_fake(prec_info, pred_ions)
        pred[precursor_i][np.invert(filt)] = -1
        
        for frag_i in range(pred_ions.size):
            if filt[frag_i]:
                annot = annotation.from_entry(pred_ions[frag_i], int(charge))
                frag_mzs[precursor_i][frag_i] = annot.getMZ(peptides[precursor_i])
            else:
                frag_mzs[precursor_i][frag_i] = -1

    intensities = {"name": "intensities",
                   "dataype": "FP32",
                   "shape": list(pred.shape),
                   "data": pred.flatten().tolist()}
    
    mzs = {"name": "mz",
                   "dataype": "FP32",
                   "shape": list(frag_mzs.shape),
                   "data": frag_mzs.flatten().tolist()}
    
    annotations = {"name": "annotations",
                   "dataype": "BYTES",
                   "shape": list(pred.shape),
                   "data": flatten(np.repeat(np.expand_dims(pred_ions,axis=0), pred.shape[0], axis=0).tolist())}
    
    batch_output = {"id": str(batch_idx), 
                    "model_name": "Altimeter", 
                    "model_version": "0.1",
                    "outputs": [intensities, mzs, annotations]}
    
    #pretty_json_str = pprint.pformat(batch_output, compact=False).replace("'",'"')
    #outfile.write(pretty_json_str)
    
    json.dump(batch_output, outfile)
    
   

def get_mod_seq(seq, mods):
    mods_oms = mods.replace("Unimod:4", "Carbamidomethyl").replace("Unimod:35", "Oxidation")
    corrected_mods = ""
    if len(mods_oms) > 0:
        index2mod = defaultdict(list)
        for mod in mods_oms.split("(")[1:]:
            mod = mod.strip(")")
            index, aa, ptm = mod.split(",")
            index = int(index)-1
            index2mod[index].append(ptm)
            corrected_mods += "(" + str(index) + "," + aa + "," + ptm + ")"
        mod_seq = ""
        for i, aa in enumerate(seq):
            if i in index2mod:
                for mod in index2mod[i]:
                    if mod == "Acetyl":
                        mod_seq += "(" + mod + ")"
            mod_seq += aa
            if i in index2mod:
                for mod in index2mod[i]:
                    if mod != "Acetyl":
                        mod_seq += "(" + mod + ")"
        peptide = oms.AASequence.fromString(mod_seq)
        mods_oms = str(len(mods_oms.split("("))-1) + mods_oms
        corrected_mods = str(len(mods_oms.split("("))-1) + corrected_mods
        corrected_mods_list.append(corrected_mods)
    else:
        peptide = oms.AASequence.fromString(seq)
        mods_oms = '0'
        corrected_mods_list.append(mods_oms)
                
    return peptide, mods_oms

def getNCE(NCE, mz, z):
    # Astral - OLD
    #if z == 2:
    #    if mz >= 500: return 32.46 - 5
    #    else: return 11.5145 + (0.041891 * mz) - 5
    #else:
    #    if mz >= 500: return 32.77 - 5
    #    else: return  9.823993 + (0.045892 * mz) - 5
        
    # Exploris - OLD
    #if z == 2:
    #    return 27.59
    #elif z == 3:
    #    return 29.62 
    #else:
    #    return 33.50
    
    # Astral Nathan
    charge_facs = [1, 0.9, 0.85, 0.8, 0.75]
    #return NCE*(charge_facs[2]/charge_facs[z])
    
    # Astral
    NCE_aligned = (-23.5 + (2.346e-1*mz) + (-3.221e-4*pow(mz,2)) + (1.432e-07*pow(mz,3))) - 5
    if z == 2:
        return NCE_aligned
    elif z == 3:
        return (NCE_aligned - 2.372e-2) / charge_facs[z]
    else:
        return (NCE_aligned + 1.75e-1) / charge_facs[z]
    
    # Exploris
    #NCE_aligned = -3.337 + (1.386e-1*mz) + (-1.861e-4*pow(mz,2)) + (7.783e-08*pow(mz,3))
    #if z == 2:
    #    return NCE_aligned
    #elif z == 3:
    #    return NCE_aligned + 2.023
    #else:
    #    return NCE_aligned + 4.811
    
    
    
       

    
def predict_batch(labels):
    samples, info = L.input_from_str(labels)
    samplesgpu = [m.to(device) for m in samples]
    
    pred = model(samplesgpu)
    pred = torch.div(pred, torch.max(pred, dim=1, keepdim=True)[0])
    pred = pred.cpu().detach().numpy()
    
    return pred, info
    
    
    
   

min_int = 0
with torch.no_grad():
    with open(config['to_predict']) as predict_infile:
        batch_size = config['batch_size']
        batch_idx = 0
        job_id = args.job_id-1
        reader = csv.reader(predict_infile, delimiter=",")
        header = next(reader)

        labels = []
        peptides = []
        corrected_mods_list = []
        for row_i, row in enumerate(reader):
            batch_idx = int(row_i / batch_size)
                
            if batch_idx % args.num_jobs != job_id: continue
            
            #[upid, acc, seq, mods, z, NCE, decoy, iRT] = row
            [upid, acc, seq, mods, z,  NCE, _,_,_,_,_,_,_,_,_,_]= row
            #[upid, acc, seq, mods, z, NCE, decoy, entrapment, iRT] = row
            
            peptide, mods_oms = get_mod_seq(seq, mods)
            
            mz = peptide.getMonoWeight(oms.Residue.ResidueType.Full, int(z)) / int(z)
            NCE = getNCE(float(NCE), mz, int(z))
            peptides.append(peptide)
            
            label = seq + "/" + z + "_" + mods_oms + "_NCE" + "{:.2f}".format(NCE) + "_0-2000_0.001_(1)0_1" 
            labels.append(label)
            
            # predict and output previous batch if it was the right job_id
            if len(labels) == batch_size: 
                pred, info = predict_batch(labels)
                writeJSON_batch(pred, info, peptides, corrected_mods_list, batch_idx)
                    
                labels.clear()
                peptides.clear()
                corrected_mods_list.clear()

                
         # last batch
        if batch_idx % args.num_jobs == job_id:
            pred, info = predict_batch(labels)
            writeJSON_batch(pred, info, peptides, corrected_mods_list, batch_idx)
            
            

    