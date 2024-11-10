import os
import sys
import msp
import yaml
import utils_unispec
import numpy as np


job_ID = int(sys.argv[1])
#################################################################################
with open(os.path.join(os.path.dirname(__file__), "../config/mods.yaml"), 'r') as stream:
    mod_config = yaml.safe_load(stream)
with open(os.path.join(os.path.dirname(__file__), "../config/data.yaml"), 'r') as stream:
    config = yaml.safe_load(stream)
D = utils_unispec.DicObj(config['ion_dictionary_path'], config['seq_len'], config['chlim'])
#################################################################################

def rename_mods(pep, mod_string):
    if "+" in mod_string:
        for mass in mod_config["mods2"]:
            mod_string = mod_string.replace(mass, mod_config["mods2"][mass])
        mod_string = mod_string.replace("+", pep[0])
    return mod_string

def processFile(files, dataset, mask_ppm_tol=50):
    
    # Create dataset outfile
    dataset_path = os.path.join(config['base_path'], config['dataset_path'])
    if not os.path.exists(dataset_path): os.makedirs(dataset_path)
    dataset_outfile = open(os.path.join(dataset_path, dataset + ".txt"), 'w')
    # Create position index outfile
    position_path = os.path.join(config['base_path'], config['position_path'])
    if not os.path.exists(position_path): os.makedirs(position_path)
    position_outfile = open(os.path.join(position_path, "fpos" + dataset + ".txt"), 'w')
    
    for file in files:
        for scan_i, scan in enumerate(msp.read_msp_file(file)):
            # check if valid peptide
            if scan.spectrum.size() == 0: continue
            if scan.getBasePeakIntensity() == 0: continue
            
            if scan.peptide.size() > config['peptide_criteria']['max_length']: continue
            if scan.peptide.size() < config['peptide_criteria']['min_length']: continue
            if scan.metaData.z > config['peptide_criteria']['max_charge']: continue
            if scan.metaData.z < config['peptide_criteria']['min_charge']: continue
            pep = scan.peptide.toUnmodifiedString()
            mod_string = rename_mods(pep, scan.getModString())
            if any([mod in mod_string for mod in config['peptide_criteria']['modifications_exclude']]): continue

            # normalize intensities
            #scan.normalizeToBasePeak(doLOD = True)
            #if scan.metaData.lowMz > 0:
            #    scan.normalizeToTotalAnnotated(doLOD = True)
            
            # Need to know how many extra peaks we have due to identical ambiguous annotations
            num_peaks = 0
            for i, annot_list in enumerate(scan.annotations):
                num_peaks += len(annot_list.entries)
                    
            # Write the scan name and observed peaks
            lines = []
            num_valid = 0
            
            weight = np.sqrt(scan.metaData.purity * scan.metaData.rawOvFtT)
            
            lines.append("NAME: %s|%s|%d|%.2f|%.1f|%.1f|%.7f|%.1f|%d\n"%(pep, mod_string, scan.metaData.z, float(scan.metaData.key2val["NCE_aligned"]), scan.metaData.lowMz, scan.metaData.highMz, scan.metaData.LOD, weight, num_peaks))
            
            for i, [annot_list, peak] in enumerate(zip(scan.annotations, scan.spectrum)):
                norm_int = peak.getIntensity()#/max_int
                if len(annot_list.entries) == 1:
                    annot = annot_list.entries[0]
                    annot_name = annot.getName()
                    if norm_int == 0 and scan.mask[i] == "1":
                        # masked outside of scan range
                        lines.append('%s %d %.4f %.5f %s\n'%(annot.getIsoName(), D.ion2index[annot_name], annot.getMZ(scan.peptide), norm_int, scan.mask[i]))
                    elif abs(annot.error) <= mask_ppm_tol :
                        #if peak.getIntensity() < (1-scan.metaData.purity)/4:
                            # mask low abundant and unpure
                        #    lines.append('%s %d %.4f %.5f %s\n'%(annot.getIsoName(), D.ion2index[annot_name], annot.getMZ(scan.peptide), norm_int, "3"))
                        #else:
                            # valid
                            lines.append('%s %d %.4f %.5f %s\n'%(annot.getIsoName(), D.ion2index[annot_name], annot.getMZ(scan.peptide), norm_int, scan.mask[i]))
                            num_valid += 1
                    else:
                        # mask ambig error
                        lines.append('%s %d %.4f %.5f %s\n'%(annot.getIsoName(), D.ion2index[annot_name], annot.getMZ(scan.peptide), norm_int, "4"))
                elif len(annot_list.entries) > 1:
                    # valid ambig annot
                    for annot in annot_list.entries:
                        lines.append('%s %s %.4f %.5f %s\n'%(annot.getIsoName(), str(D.ion2index[annot_name]), peak.getMZ(), norm_int, "5"))
                    #num_valid += 1
                else:
                    # shouldn't happen
                    lines.append('%s %d %.4f %.5f %s\n'%("?", -1, peak.getMZ(), peak.getIntensity(), "0"))
                    
            if num_valid >= 2:
                # write position
                position_outfile.write("%d "%dataset_outfile.tell())
                # write dataset
                for line in lines:
                    dataset_outfile.write(line)
                
                
                
                
###############################################################################
############################# Main part of script #############################
###############################################################################


# Process datasets
if job_ID == 1:
    processFile([config['test_files']], 'test', 15)
elif job_ID == 2:
    processFile([config['val_files']], 'val', 15)
elif job_ID == 3:
    #processFile([config['train_files']], 'train', 15)
    processFile([config['interp_files'], config['train_files']], 'train', 15)