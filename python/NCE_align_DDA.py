import torch
from models import FlipyFlopy
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
from collections import defaultdict
import argparse
plt.close('all')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(
                    prog='NCE alignment',
                    description='Align NCE values using DDA data and a model')
parser.add_argument("msp_path")
parser.add_argument("out_path")
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
    model = FlipyFlopy(**model_config, device=device)
    model.load_state_dict(torch.load(config['model_ckpt'], weights_only=True))
    model.to(device)
    model.eval()


pred_ions = np.array([D.index2ion[ind] for ind in range(len(D.index2ion))])


def scan2tensor(scan):
    targ = torch.zeros(D.dicsz, dtype=torch.float32)
    for i, peak in enumerate(scan.spectrum):
        # check mask
        if scan.mask[i] == 0 or scan.mask[i] == 3:
            # get index of fragment
            annot = scan.annotations[i].annotationName()
            if annot in D.ion2index and scan.annotations[i].entries[0].length >= 3:
                targ[D.ion2index[annot]] = peak.getIntensity()
    return targ


CS = torch.nn.CosineSimilarity(dim=-1)
def LossFunc(targ, pred, epsilon=1e-5):
    # only use fragments that were observed in the experimental spectrum
    pred = torch.where(targ==0, 0.0, pred)
    
    cs = CS(targ, pred)
    cs = torch.clamp(cs, min=-(1-epsilon), max=(1-epsilon))
    sa = 1 - 2 * (torch.arccos(cs) / torch.pi)
    return sa

def rename_mods(pep, mod_string):
    if "+" in mod_string:
        for mass in mod_config["mods2"]:
            mod_string = mod_string.replace(mass, mod_config["mods2"][mass])
        mod_string = mod_string.replace("+", pep[0])
    return mod_string


min_NCE = 20
max_NCE = 40
NCE_step_size = 0.01
NCE_steps = 1+int((max_NCE-min_NCE) / NCE_step_size)
NCE_steps = np.linspace(min_NCE, max_NCE, NCE_steps)

with open(os.path.join(args.out_path, "NCE_alignment.tsv"), 'w', newline="", encoding='utf-8') as outfile:
    writer = csv.writer(outfile, delimiter="\t")
    writer.writerow(["scan_id", "mz", "z", "best_NCE", "SA"])

#print("\t".join(["scan_id", "mz", "z", "best_NCE", "SA"])) 
    with torch.no_grad():
        for i, scan in enumerate(msp.read_msp_file(args.msp_path)):
            pep = scan.peptide.toUnmodifiedString()
            mod_string = rename_mods(pep, scan.getModString())
            if any([mod in mod_string for mod in dconfig['peptide_criteria']['modifications_exclude']]): continue
            
            targ = scan2tensor(scan)
            num_frags = torch.sum(targ > 0)
            if num_frags < 3: continue
            
            labels = []
            for NCE in NCE_steps:
                label = scan.peptide.toUnmodifiedString() + "/" + str(scan.metaData.z) + "_" + mod_string + "_NCE" + "{:.2f}".format(NCE) + "_0-2000_0.001_(1)0_1" 
                labels.append(label)

            samples, info = L.input_from_str(labels)
            samplesgpu = [m.to(device) for m in samples]
                
            pred,_,_ = model(samplesgpu, test=False)
            pred = pred.cpu().detach()
            
            targ = targ.unsqueeze(0).expand_as(pred)
            
            SAs = LossFunc(targ, pred)
            
            writer.writerow([scan.metaData.scanID, scan.metaData.isoCenter, scan.metaData.z, NCE_steps[torch.argmax(SAs)], torch.max(SAs).numpy()])
            #row = "\t".join([str(scan.metaData.scanID), "{:.2f}".format(scan.metaData.isoCenter), str(scan.metaData.z), "{:.2f}".format(NCE_steps[torch.argmax(SAs)]), "{:.4f}".format(torch.max(SAs).numpy())])
            #if len(row) > 100: print("WTF", len(row)); sys.exit()
            
            #print(row)

            #if i > 10:
            #    sys.exit()
        