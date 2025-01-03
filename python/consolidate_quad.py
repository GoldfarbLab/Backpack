import os
import sys
import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

quad_data = sys.argv[1]
out_path = sys.argv[2]

def gaussian(a, b, x):
    return 1 / (1 + pow(abs(x)/a, 2*b))

def fun(params, x, y):
    a, b = params
    return y - gaussian(a, b, x)

def full_fit(cal_df):
    # trimmed approach
    cal_df['offset_round'] = cal_df.offset.round(1)
    res = cal_df.groupby(by="offset_round")["ratio"].quantile([0.05, 0.95]).unstack(level=1)
    cal_df = cal_df.loc[(cal_df.ratio.values == 1) | ((res.loc[cal_df.offset_round, 0.05] < cal_df.ratio.values) & (cal_df.ratio.values < res.loc[cal_df.offset_round, 0.95])).values]
    
    cal_df_left = cal_df[cal_df.offset_round <= 0.1]
    cal_df_right = cal_df[cal_df.offset_round >= -0.1]
    
    cal_df_left = cal_df_left.drop(columns=["date", "mz", "z", "cal", "offset_round"])
    cal_df_right = cal_df_right.drop(columns=["date", "mz", "z", "cal", "offset_round"])
    
    factor=1
    more_rows_left = [{"offset": -1.5, "ratio": 0, } for i in range((int)(len(cal_df_left.index)/factor))]
    more_rows_right = [{"offset": 1.5, "ratio": 0, } for i in range((int)(len(cal_df_right.index)/factor))]
    
    cal_df_left = pd.concat([cal_df_left, pd.DataFrame(more_rows_left)]).reset_index(drop=True)
    cal_df_right = pd.concat([cal_df_right, pd.DataFrame(more_rows_right)]).reset_index(drop=True)
    
    out_left = least_squares(fun, p0, bounds=bounds, loss="soft_l1", ftol = 3e-16, gtol = 3e-16, xtol = 3e-16, max_nfev=10000, args=(cal_df_left.offset.values, cal_df_left.ratio.values))
    out_right = least_squares(fun, p0, bounds=bounds, loss="soft_l1", ftol = 3e-16, gtol = 3e-16, xtol = 3e-16, max_nfev=10000, args=(cal_df_right.offset.values, cal_df_right.ratio.values))
    out = least_squares(fun, p0, bounds=bounds, loss="soft_l1", ftol = 3e-16, gtol = 3e-16, xtol = 3e-16, max_nfev=10000, args=(cal_df.offset.values, cal_df.ratio.values))
    
    # QC
    cal_df['offset_round'] = cal_df.offset.round(1)
    s = cal_df.groupby(by="offset_round").offset_round.count()
    valid_offsets = set(s[s>=100].index.tolist())
    cal_df = cal_df[cal_df.offset_round.isin(valid_offsets)]
    cal_df = cal_df.groupby(by="offset_round").ratio.median().clip(upper=1)
    
    # get left/right side
    cal_df_left = cal_df[cal_df.index <= 0.1]
    cal_df_right = cal_df[cal_df.index >= -0.1]
    left_size = sum((cal_df_left.values <= 0.9) & (cal_df_left.values >= 0.1))
    right_size = sum((cal_df_right.values <= 0.9) & (cal_df_right.values >= 0.1))
    
    print(out_left.success, out_right.success)
    
    if left_size <= 1 or not out_left.success or out_left.x[0] == p0[0]:
        out_left = out
    if right_size <= 1 or not out_right.success or out_right.x[0] == p0[0]:
        out_right = out
    
    print(out_left.x, len(cal_df_left.index), left_size)
    print(out_right.x, len(cal_df_right.index), right_size)
    print(out.x, len(cal_df.index))
    
    return out_left, out_right, out




    

p0 = [0.65, 4.5] # initial guess
bounds = ([0, 1], [1, 100]) 
options = {'max_nfev': 100} 

df = pd.read_csv(quad_data, sep="\t")
df.cal = df.cal.fillna("QE")

x_space = np.linspace(-1.5, 1.5, 301)
x_space_right = np.linspace(0, 1.5, 151)
x_space_left = np.linspace(-1.5, 0, 151)

with open(out_path + "/quad_models.tsv", "w", newline="") as tsvfile:
    writer = csv.writer(tsvfile, delimiter="\t") 
    writer.writerow(["cal", "left_a", "left_b", "right_a", "right_b"])
    
    for cal in df.cal.unique():
        print(cal)
        cal_df = df[df.cal == cal]

        out_left, out_right, out = full_fit(cal_df)
        
        writer.writerow([cal, str(out_left.x[0]), str(out_left.x[1]), str(out_right.x[0]), str(out_right.x[1])])
        
        plt.plot(x_space_left, gaussian(out_left.x[0], out_left.x[1], x_space_left), label = cal)
        plt.plot(x_space_right, gaussian(out_right.x[0], out_right.x[1], x_space_right), label = cal)


plt.savefig(out_path+'/quad.pdf')