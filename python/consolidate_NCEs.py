import os
import sys
import pandas as pd
import msp


frames = []
top_dir = sys.argv[1]
out_path = sys.argv[2]
for f in os.listdir(top_dir):
    top_path = os.fsdecode(f)
    filename = os.path.basename(top_path)
    
    if "ETD" in filename: continue
    
    stats_path = os.path.join(top_dir, top_path, "results/NCE", filename + ".msp.deisotoped.NCE")
    print(stats_path)
    if os.path.exists(stats_path):
        frames.append(pd.read_csv(stats_path, sep="\t", header=None))
    else:
        print("MISSING!!", filename)
        
NCE_align_data = pd.concat(frames).reset_index()
NCE_align_data.columns = ['index', 'file', 'instrument_id', 'cal_date', 'created_date', 'NCE', 'offset']

NCE_align_data['cal_date'] = pd.to_datetime(NCE_align_data['cal_date'])
NCE_align_data['created_date'] = pd.to_datetime(NCE_align_data['created_date'])

NCE_align_data = NCE_align_data.sort_values(by=['created_date'])
###################

smoothed_data = NCE_align_data.groupby(['cal_date', 'instrument_id', 'NCE'])[['offset', 'created_date']].apply(lambda x: x.rolling('24h', min_periods=1, center=True, on='created_date').mean()).reset_index()

smoothed_data['offset_aligned'] = smoothed_data['offset']
smoothed_data.drop(['offset', 'level_3'], axis=1, inplace=True)



################
smoothed_data = smoothed_data.set_index(['instrument_id', 'NCE', 'cal_date', 'created_date'])
NCE_align_data = NCE_align_data.set_index(['instrument_id', 'NCE', 'cal_date', 'created_date'])

NCE_align_data = pd.concat([NCE_align_data, smoothed_data], axis=1, join="inner")

NCE_align_data = NCE_align_data.rename(columns={'index': "pandas is dumb"})
NCE_align_data.drop('pandas is dumb', axis=1, inplace=True)

NCE_align_data.reset_index(inplace=True)


## Impute by NCE + cal date

s = (pd.merge_asof(
         NCE_align_data.sort_values('created_date').reset_index(),            # Full Data Frame
         NCE_align_data.sort_values('created_date').dropna(subset=['offset_aligned']), # Subset with valid scores
         by=['instrument_id','cal_date','NCE'],                                         # Only within `'cn'` group
         on='created_date', direction='nearest'                   # Match closest date 
                  )
       .set_index('index')
       .offset_aligned_y)


NCE_align_data['offset_aligned'] = NCE_align_data['offset_aligned'].fillna(s, downcast='infer')

## Impute by cal date
NCE_align_summarized = NCE_align_data.groupby(['cal_date', 'instrument_id'])['offset_aligned'].mean().reset_index()

s = (pd.merge_asof(
         NCE_align_data.sort_values('cal_date').reset_index(),            # Full Data Frame
         NCE_align_summarized.sort_values('cal_date').dropna(subset=['offset_aligned']), # Subset with valid scores
         by=['instrument_id','cal_date'],                                         # Only within `'cn'` group
         on='cal_date', direction='nearest'                   # Match closest date 
                  )
       .set_index('index')
       .offset_aligned_y)


NCE_align_data['offset_aligned'] = NCE_align_data['offset_aligned'].fillna(s, downcast='infer')

NCE_align_data.to_csv(out_path, sep="\t", index=False)

