import os
import sys
import pandas as pd
import msp


in_path = sys.argv[1]
out_path = in_path
        
NCE_align_data = pd.read_csv(in_path, sep="\t")

NCE_align_data['cal_date'] = pd.to_datetime(NCE_align_data['cal_date'])
NCE_align_data['created_date'] = pd.to_datetime(NCE_align_data['created_date'])

NCE_align_data = NCE_align_data.sort_values(by=['created_date'])
NCE_align_data.drop(['offset_aligned'], axis=1, inplace=True)
###################

# original rolling mean
smoothed_data = NCE_align_data.groupby(['cal_date', 'instrument_id', 'NCE'])[['offset', 'created_date']].apply(lambda x: x.rolling('6h', min_periods=1, center=True, on='created_date').mean()).reset_index()
smoothed_data['offset_aligned'] = smoothed_data['offset']
smoothed_data.drop(['offset', 'level_3'], axis=1, inplace=True)


# EWMA
# smoothed_data['created_date'] = NCE_align_data.created_date
#frames = []
#for inst_id in NCE_align_data.instrument_id.unique():
#    inst_df = NCE_align_data[NCE_align_data.instrument_id == inst_id]
#    for cal in inst_df.cal_date.unique():
#        cal_df = inst_df[inst_df.cal_date == cal]
#        smoothed_offsets = cal_df.offset.ewm(halflife='2 hour', times=pd.DatetimeIndex(cal_df.created_date), ignore_na=True).mean()
#        cal_df['offset_aligned'] = smoothed_offsets
#        frames.append(cal_df)
        
#smoothed_data = pd.concat(frames)


################
# EWMA
# smoothed_data = smoothed_data.set_index(['instrument_id', 'NCE', 'cal_date', 'created_date'])
# NCE_align_data = smoothed_data

smoothed_data = smoothed_data.set_index(['instrument_id', 'NCE', 'cal_date', 'created_date'])
NCE_align_data = NCE_align_data.set_index(['instrument_id', 'NCE', 'cal_date', 'created_date'])

NCE_align_data = pd.concat([NCE_align_data, smoothed_data], axis=1, join="inner")

#print(NCE_align_data)
#NCE_align_data = NCE_align_data.rename(columns={'index': "pandas is dumb"})

#print(NCE_align_data)
#NCE_align_data.drop('pandas is dumb', axis=1, inplace=True)

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

