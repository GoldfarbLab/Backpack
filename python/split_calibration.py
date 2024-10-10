import msp
import sys
import random
import os

data_path = "/Users/dennisgoldfarb/Downloads/ProCal/merged/procal.msp.deisotoped"
train_split = 0.8

with open("/Users/dennisgoldfarb/Downloads/ProCal/merged/procal.msp.deisotoped.train", 'w') as train_outfile:
     with open("/Users/dennisgoldfarb/Downloads/ProCal/merged/procal.msp.deisotoped.test", 'w') as test_outfile:
         for i, scan in enumerate(msp.read_msp_file(data_path)):
            if i > 0 and i % 10000 == 0: print(i)
            if random.random() <= train_split:
                scan.writeScan(train_outfile, False, int_prec=5)
            else:
                scan.writeScan(test_outfile, False, int_prec=5)
            
            #if scan.metaData.NCE == 30:
            #    scan.writeScan(test_outfile, False, int_prec=5)
            #else:
            #    scan.writeScan(train_outfile, False, int_prec=5)