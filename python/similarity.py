from pyopenms import MSSpectrum
from utils import getPPMAbs
import numpy as np
import statistics
import annotation
import sys

def CS_scans(scan1, scan2, unannotated=True, ambiguous=True, terminal=True, precursor=True, immonium=True, neutral_losses=True, internal=True):
    
    v1, v2, mz_aligned, annotations_aligned, mask_aligned = match(scan1, scan2)
    
    
    
    v1, annotations, mask, mzs = filterByAnnotations(v1, annotations_aligned, mask_aligned, mz_aligned, unannotated, ambiguous, terminal, precursor, immonium, neutral_losses, internal)
    v2, annotations, mask, mzs = filterByAnnotations(v2, annotations_aligned, mask_aligned, mz_aligned, unannotated, ambiguous, terminal, precursor, immonium, neutral_losses, internal)
    
    v1[v2 == 0] = 0
    
    scan1.updateLOD()
    scan2.updateLOD()

    #LOD = 2*max(scan1.getLOD(), scan2.getLOD())
    LOD = max(scan1.getLOD(), scan2.getLOD())
    
    #if LOD == 0:
    #    print(scan1.getLOD(), scan2.getLOD())

    sa = spectralAngle(v1, v2, LOD)
    
    #if sa < 0.8:
    #    print("BAD", sa, LOD, scan1.metaData.scanID)
        
    #    v1, v2, mz_aligned, annotations_aligned, mask_aligned = match(scan1, scan2, verbose=True)
        
    #    v1[v2 == 0] = 0
        #filterByAnnotations(v1, annotations_aligned, mask_aligned, mz_aligned, unannotated, ambiguous, terminal, precursor, immonium, neutral_losses, internal, True)
        #filterByAnnotations(v2, annotations_aligned, mask_aligned, mz_aligned, unannotated, ambiguous, terminal, precursor, immonium, neutral_losses, internal, True)
        
        
     #   for i in range(len(scan1.annotations)):
     #       print(i, scan1.spectrum[i].getMZ(), scan1.spectrum[i].getIntensity(), scan1.annotations[i].annotationString())
            
     #   for i in range(len(scan2.annotations)):
     #       print(i, scan2.spectrum[i].getMZ(), scan2.spectrum[i].getIntensity(), scan2.annotations[i].annotationString())

     #   for i in range(v1.size):
     #       if v1[i] > 0 and v2[i] > 0:
     #           print(v1[i], v2[i], mz_aligned[i], annotations_aligned[i].annotationString(), mask_aligned[i])
            
        #sys.exit()

    return sa




def match(scan1, scan2, verbose=False):
    mz_targ, int_targ = scan1.spectrum.get_peaks()
    mz_pred, int_pred = scan2.spectrum.get_peaks()
    
    print(len(mz_targ), len(int_targ), len(mz_pred), len(int_pred), scan2.metaData.scanID)
    mz_aligned = alignMZ(mz_targ, mz_pred, verbose)
    
    int_targ_aligned, targ_annotations, targ_mask = alignToMZ(int_targ, mz_targ, mz_aligned, scan1.annotations, scan1.mask, verbose)
    int_pred_aligned, pred_annotations, pred_mask = alignToMZ(int_pred, mz_pred, mz_aligned, scan2.annotations, scan2.mask, verbose)
    
    annotations_aligned = mergeAnnotations(targ_annotations, pred_annotations)
    mask_aligned = mergeMask(targ_mask, pred_mask)
    
    for i, annot_list in enumerate(annotations_aligned):
        if len(annot_list.entries) == 0:
            int_targ_aligned[i] = 0
            int_pred_aligned[i] = 0
    
    if verbose:
        print(int_targ_aligned)
        print(int_pred_aligned)
    
    return int_targ_aligned, int_pred_aligned, mz_aligned, annotations_aligned, mask_aligned


def alignMZ(mz_targ, mz_pred, verbose=False):
    ppm_tol=20
    mz_aligned = []
    
    targ_i = 0
    pred_i = 0
    
    mz_aligned.append([min(mz_targ[0], mz_pred[0]), min(mz_targ[0], mz_pred[0])])
    
    while targ_i < len(mz_targ) and pred_i < len(mz_pred):
        if getPPMAbs(mz_targ[targ_i], mz_aligned[-1][1]) <= ppm_tol or mz_targ[targ_i] < mz_aligned[-1][1]:
            if verbose: print("first", mz_targ[targ_i], mz_aligned[-1][1])
            mz_aligned[-1][1] = max(mz_aligned[-1][1], mz_targ[targ_i])
            targ_i += 1
        elif getPPMAbs(mz_pred[pred_i], mz_aligned[-1][1]) <= ppm_tol or mz_pred[pred_i] < mz_aligned[-1][1]:
            if verbose: print("second", mz_pred[pred_i], mz_aligned[-1][1], getPPMAbs(mz_pred[pred_i], mz_aligned[-1][1]))
            mz_aligned[-1][1] = max(mz_aligned[-1][1], mz_pred[pred_i])
            pred_i += 1
        else:
            mz_aligned.append([min(mz_targ[targ_i], mz_pred[pred_i]), min(mz_targ[targ_i], mz_pred[pred_i])])
            if verbose: print("third", mz_aligned[-1])
        
    while targ_i < len(mz_targ):
        if getPPMAbs(mz_targ[targ_i], mz_aligned[-1][1]) <= ppm_tol or mz_targ[targ_i] < mz_aligned[-1][1]:
            mz_aligned[-1][1] = max(mz_aligned[-1][1], mz_targ[targ_i])
        else:
            mz_aligned.append([mz_targ[targ_i], mz_targ[targ_i]])
        targ_i += 1
        
    while pred_i < len(mz_pred):
        if getPPMAbs(mz_pred[pred_i], mz_aligned[-1][1]) <= ppm_tol or mz_pred[pred_i] < mz_aligned[-1][1]:
            mz_aligned[-1][1] = max(mz_aligned[-1][1], mz_pred[pred_i])
        else:
            mz_aligned.append([mz_pred[pred_i], mz_pred[pred_i]])
        pred_i += 1
        
    return np.array(mz_aligned, dtype=np.float32)


def alignToMZ(intensities, mzs, mz_aligned, annotations, mask, verbose=False):
    int_targ_aligned = np.zeros(mz_aligned.shape[0])
    annotations_out = [annotation.annotation_list([]) for i in range(mz_aligned.shape[0])]
    mask_out = [0 for i in range(mz_aligned.shape[0])]
    targ_index = 0
    eps = 1e-4
    for i in range(int_targ_aligned.shape[0]):
        while mzs[targ_index] - mz_aligned[i][0] > -eps and mzs[targ_index] - mz_aligned[i][1] < eps:
            int_targ_aligned[i] += intensities[targ_index]
            if verbose: print("align2mz", i,  mzs[targ_index], int_targ_aligned[i])
            mask_out[i] = 1 if mask[targ_index] == 1 else max(mask_out[i],  mask[targ_index])
            annotations_out[i].entries.extend(annotations[targ_index].entries)
            targ_index += 1
            if targ_index >= len(intensities): break
        if targ_index >= len(intensities): break
            
    while targ_index < intensities.shape[0]:
        int_targ_aligned[int_targ_aligned.size-1] += intensities[targ_index]
        mask_out[int_targ_aligned.size-1] = 1 if mask[targ_index] == 1 else max(mask_out[int_targ_aligned.size-1], mask[targ_index])
        annotations_out[int_targ_aligned.size-1].entries.extend(annotations[targ_index].entries)
        targ_index += 1
        
    return int_targ_aligned, annotations_out, mask_out


def mergeMask(targ_mask, pred_mask):
    mask = []
    for i in range(len(targ_mask)):
        if targ_mask[i] == 1 or pred_mask[i] == 1:
            mask.append(1)
        else:
            mask.append(max(targ_mask[i], pred_mask[i]))
    return mask


def mergeAnnotations(targ_annotations, pred_annotations):
    annotations = []
    for i in range(len(targ_annotations)):
        annots = []
        if targ_annotations[i].annotationString() != "?":
            for annot in targ_annotations[i].entries:
                annots.append(annot)
        if pred_annotations[i].annotationString() != "?":
            for annot in pred_annotations[i].entries:
                annots.append(annot)
        # check if there's any overlap. Require at least 1
        annot_name_set = set()
        for annot in annots:
            annot_name_set.add(annot.getIsoName())
        if len(annot_name_set) == len(annots):
            annotations.append(annotation.annotation_list([]))
        else:
            annotations.append(annotation.annotation_list(annots))
    return annotations









def filterByAnnotations(intensities, annotations, mask, mz_aligned, unannotated=True, ambiguous=True, terminal=True, precursor=True, immonium=True, neutral_losses=True, internal=True, verbose=False):
    indices_to_keep = []
    for i, annotation_list in enumerate(annotations):
        if verbose:
            print("filterByAnnotations", i, mask[i], annotation_list.annotationString())
        valid_entries = []
        if mask[i] == 1: continue
        if not unannotated and annotation_list.annotationString() == "?": continue
        if not ambiguous and mask[i] != 0: continue

        for entry in annotation_list.entries:
            if not neutral_losses and (entry.NL is not None or entry.NG is not None): continue
            if not internal and entry.name[0:3] == "Int": continue
            if not immonium and entry.name[0:3] != "Int" and entry.name[0] == "I": continue
            if not precursor and entry.name[0] == "p": continue
            if not terminal and entry.name[0] in "by": continue
            valid_entries.append(entry)
        
        annotation_list.entries = valid_entries
        
        if len(valid_entries) > 0 or unannotated:
            indices_to_keep.append(i)
        
    if verbose:
        print("filterByAnnotations input:", intensities)
            
    intensities = [intensities[i] for i in indices_to_keep]
    annotations = [annotations[i] for i in indices_to_keep]
    mask = [mask[i] for i in indices_to_keep]
    mz_aligned = [mz_aligned[i] for i in indices_to_keep]
    
    if verbose:
        print("filterByAnnotations", intensities, indices_to_keep)
    
    return np.array(intensities), annotations, mask, mz_aligned



        


def spectralAngle(v1, v2, LOD, eps=1e-5):
    cs = cosineSimilarity(v1, v2, LOD)
    cs = np.clip(cs, a_min=-(1-eps), a_max = 1-eps)
    sa = 1 - 2 * (np.arccos(cs) / np.pi)
    return sa

def cosineSimilarity(v1, v2, LOD):
    v1_under_LOD = v1 < LOD
    v2_under_LOD = v2 < LOD
    
    v1[v1_under_LOD & v2_under_LOD] = 0
    v2[v1_under_LOD & v2_under_LOD] = 0
    
    #v1 = np.sqrt(v1)
    #v2 = np.sqrt(v2)
    
    sim = (v1*v2).sum() / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-8)
    sim = min(sim, 1)
    
    return sim

def scribe(v1, v2, LOD):
    v1_under_LOD = v1 < LOD
    v2_under_LOD = v2 < LOD
    
    v1[v1_under_LOD & v2_under_LOD] = 0
    v2[v1_under_LOD & v2_under_LOD] = 0
    
    if sum(v1) == 0 or sum(v2) == 0: return 0
  
    v1 = np.sqrt(v1)
    v2 = np.sqrt(v2)
    
    norm_pred = v1 / sum(v1)
    norm_targ = v2 / sum(v2)
    
    scribe = -np.log(np.sum(np.power(norm_pred - norm_targ, 2)))
    
    return scribe