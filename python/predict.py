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
from scipy.interpolate import splint, BSpline
from scipy import integrate
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

D = utils_unispec.DicObj(dconfig['ion_dictionary_path'], mod_config, dconfig['seq_len'], dconfig['chlim']) 
L = utils_unispec.LoadObj(D, embed=True)

# Instantiate model
with open(config['model_config']) as stream:
    model_config = yaml.safe_load(stream)
    model = LitFlipyFlopy.load_from_checkpoint(config['model_ckpt'], config=config, model_config=model_config)
    model.eval()


pred_ions = np.array([D.index2ion[ind] for ind in range(len(D.index2ion))])
spline_knots = np.array([6, 13, 20, 27, 34, 41, 48, 55])

def rename_mods(pep, mod_string):
    if "+" in mod_string:
        for mass in mod_config["mods2"]:
            mod_string = mod_string.replace(mass, mod_config["mods2"][mass])
        mod_string = mod_string.replace("+", pep[0])
    return mod_string

def flatten(xss):
    return [x for xs in xss for x in xs]


def writeJSON_batch(info, coef, peptides, corrected_mods_list, batch_idx):
    outfile = open(config['out_file'] + '_' + str(batch_idx) + ".json", 'w', encoding="utf-8")
    
    
    coef_filtered = np.zeros((coef.shape[0], 4, 50))
    frag_mzs = np.zeros((coef.shape[0], 50))
    annotations = np.empty_like(frag_mzs, dtype="U20")
    integrals = np.zeros_like(frag_mzs)
    
    for precursor_i in range(coef.shape[0]):
        seq, _, charge, nce, min_mz, max_mz, LOD, iso2efficiency, weight = info[precursor_i]
        prec_info = (seq, corrected_mods_list[precursor_i], charge, nce, min_mz, max_mz, LOD, iso2efficiency, weight) 
        filt = L.filter_fake(prec_info, pred_ions)
        
        coef[precursor_i][:, np.invert(filt)] = -1
        temp_integrals = np.zeros(coef.shape[2])
        # Compute integral for each fragment
        for frag_i in range(coef.shape[2]):
            spline_fit = BSpline(spline_knots, coef[precursor_i][:, frag_i], 3)
            temp_integrals[frag_i] = integrate.quad(spline_fit, 20, 40)[0] #splint(20,40,spline_fit)
        # sort by probability
        indices = np.argsort(-temp_integrals)
        # take the top N
        coef_filtered[precursor_i] = coef[precursor_i][:,indices[0:50]]
        integrals[precursor_i] = temp_integrals[indices[0:50]]
        
        # keep track of which frag mzs and annotations to output
        for i, frag_i in enumerate(indices[0:50]):
            if filt[frag_i]:
                annot = annotation.from_entry(pred_ions[frag_i], int(charge))
                frag_mzs[precursor_i][i] = annot.getMZ(peptides[precursor_i])
                annotations[precursor_i][i] = pred_ions[frag_i]
            else:
                frag_mzs[precursor_i][i] = -1
                annotations[precursor_i][i] = "NA"
                integrals[precursor_i][i] = -1
    

    #myspline = BSpline(spline_knots, coef[0][:, 22], 3, extrapolate=False)
    #plt.plot(NCE_steps, myspline(NCE_steps), label = "scipy")
    #plt.legend()
    
    knots = {"name": "knots",
                   "dataype": "INT32",
                   "shape": list(spline_knots.shape),
                   "data": spline_knots.tolist()}
    
    mzs = {"name": "mz",
                   "dataype": "FP32",
                   "shape": list(frag_mzs.shape),
                   "data": frag_mzs.flatten().tolist()}

    AUCs = {"name": "AUC",
                   "dataype": "FP32",
                   "shape": list(integrals.shape),
                   "data": integrals.flatten().tolist()}
    
    annotations = {"name": "annotations",
                   "dataype": "BYTES",
                   "shape": list(annotations.shape),
                   "data": annotations.flatten().tolist()}
    
    coefficients = {"name": "coefficients",
                   "dataype": "FP32",
                   "shape": list(coef_filtered.shape),
                   "data": coef_filtered.flatten().tolist()}
    
    batch_output = {"id": str(batch_idx), 
                    "model_name": "Altimeter", 
                    "model_version": "0.1",
                    "outputs": [knots, mzs, AUCs, annotations, coefficients]}
    
    json.dump(batch_output, outfile)
    #json.dump(batch_output, sys.stdout, indent=4)
    #sys.exit()
    
   

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
    NCE_aligned = (-25.86 + (2.459e-1*mz) + (-3.394e-4*pow(mz,2)) + (1.516e-07*pow(mz,3))) - 5
    if z == 2:
        return NCE_aligned
    elif z == 3:
        return (NCE_aligned - 6.912e-3) * (charge_facs[2]/charge_facs[z])
    else:
        return (NCE_aligned + 4.201e-1) * (charge_facs[2]/charge_facs[z])
    
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
    
    coef, knots, auc = model.forward_coef(samplesgpu)
    coef = coef.cpu().detach().numpy()
    
    return info, coef

def predict_batch_NCEs(labels):
    samples, info = L.input_from_str(labels)
    samplesgpu = [m.to(device) for m in samples]
    
    out = model.forward(samplesgpu)
    out = out.cpu().detach().numpy()
    
    for precursor_i in range(out.shape[0]):
        seq, _, charge, nce, min_mz, max_mz, LOD, iso2efficiency, weight = info[precursor_i]
        prec_info = (seq, corrected_mods_list[0], charge, nce, min_mz, max_mz, LOD, iso2efficiency, weight) 
        filt = L.filter_fake(prec_info, pred_ions)
        
 
        out[precursor_i][np.invert(filt)] = -1
    np.set_printoptions(threshold=np.inf)
    print(out)
        
    plt.plot(NCE_steps, out[:,22], label = "Recursive")
    
    return info, out
    
    
    
min_NCE = 20
max_NCE = 40
NCE_step_size = 1#0.01
NCE_steps = 1+int((max_NCE-min_NCE) / NCE_step_size)
NCE_steps = np.linspace(min_NCE, max_NCE, NCE_steps)

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
            
            #print("batch:", batch_idx, row_i, job_id)
                
            if batch_idx % args.num_jobs != job_id: continue
            
            [upid, acc, seq, mods, z,  NCE, _,_,_,_,_,_,_,_,_,_]= row
            #seq = "PEPTIDEK"
            peptide, mods_oms = get_mod_seq(seq, mods)

            peptides.append(peptide)
            
            #for NCE in NCE_steps:
            #    label = seq + "/" + z + "_" + mods_oms + "_NCE" + "{:.2f}".format(NCE) + "_0-2000_0.001_(1)0_1" 
            #    labels.append(label)
            
            label = seq + "/" + z + "_" + mods_oms + "_NCE30_0-2000_0.001_(1)0_1" 
            labels.append(label)
            
            # predict and output previous batch if it was the right job_id
            if len(labels) >= batch_size: 
                print("PREDICTING:", batch_idx, row_i, job_id)
                #predict_batch_NCEs(labels)
                
                info, coef = predict_batch(labels)
                writeJSON_batch(info, coef, peptides, corrected_mods_list, batch_idx)
                #plt.savefig('/storage1/fs1/d.goldfarb/Active/Projects/Backpack/results/spline_eval.pdf')
                    
                labels.clear()
                peptides.clear()
                corrected_mods_list.clear()
                #sys.exit()
                
         # last batch
        if batch_idx % args.num_jobs == job_id:
            info, coef = predict_batch(labels)
            writeJSON_batch(info, coef, peptides, corrected_mods_list, batch_idx)
            
            

    