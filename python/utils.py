import numpy as np
from numpy.linalg import norm
import pyopenms as oms

def cosineSim(A,B):
    return np.dot(A,B)/max(1e-8,(norm(A)*norm(B)))

def getPPMAbs(mz_obs, mz_ref):
    return abs(getPPM(mz_obs, mz_ref))

def getPPM(mz_obs, mz_ref):
    return (mz_obs - mz_ref) / mz_ref * 1e6

def ppmToMass(ppm, mz_ref):
    return (ppm / 1e6) * mz_ref

def isoDist2np(dist):
    iso_prob = np.zeros(len(dist.getContainer()))
    for i in range(len(dist.getContainer())):
        iso_prob[i] =  dist.getContainer()[i].getIntensity()
    return iso_prob

def linearInterpolate(x, x_left, y_left, x_right, y_right):
    if x_left == -1: return y_right
    if x_right == -1: return y_left
    
    percent_right = (x - x_left) / (x_right - x_left)
    diff_y = y_right - y_left
    return (diff_y * percent_right) + y_left

def pepFromSage(mod_seq):
    mod_seq = mod_seq.replace("]-", "]")
    mod_seq = mod_seq.replace("-[", "[")
    return oms.AASequence.fromString(mod_seq)