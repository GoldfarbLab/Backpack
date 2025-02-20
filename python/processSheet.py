import os
import sys
import csv
import argparse
import datetime
import shutil
import fileinput
import subprocess
import re as re
from pathlib import Path


parser = argparse.ArgumentParser(
                    prog='Sheet Proccesor',
                    description='Download, search, and annotate PRIDE datasets')
parser.add_argument("sheet_path")
parser.add_argument("--parallel_downloads", "-n", default=2, type=int)
args = parser.parse_args()


def submitJob(script_path, dependent_job_id=None):
    if dependent_job_id is not None:
        result = subprocess.run("bsub -w \"ended(" + dependent_job_id + ")\" < " + script_path, shell=True, stdout=subprocess.PIPE, encoding="UTF8")
    else:
        result = subprocess.run("bsub < " + script_path, shell=True,  stdout=subprocess.PIPE, encoding="UTF8")
    job_id = re.search(".*\\<([0-9]+)\\>.*", result.stdout).group(1)
    return job_id

def createDownloadScript(src, dst, job_name, PXD, include, exclude, row_name):
    shutil.copyfile(src, dst)
    # create dictionary
    param2val = {"JOB_NAME" : job_name,
                 "PXD" : PXD,
                 "INCLUDE_STR" : '"' + include + '"',
                 "EXCLUDE_STR" : '"' + exclude + '"',
                 "ROW_NAME" : row_name}
    
    sed_file(dst, param2val)
    
def createConvertScript(src, dst, job_name, row_name):
    shutil.copyfile(src, dst)
    # create dictionary
    param2val = {"JOB_NAME" : job_name,
                 "ROW_NAME" : row_name}
    
    sed_file(dst, param2val)
    
def createSageScript(src, dst, job_name, row_name, fasta_path, enzyme):
    shutil.copyfile(src, dst)
    # create dictionary
    param2val = {"JOB_NAME" : job_name,
                 "ROW_NAME" : row_name,
                 "FASTA_PATH" : "/storage1/fs1/d.goldfarb/Active/Backpack/fasta/"+fasta_path,
                 "SAGE_SEMI_CONFIG_PATH" : "/storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/semi_config_" + enzyme + ".json",
                 "SAGE_OPEN_CONFIG_PATH" : "/storage1/fs1/d.goldfarb/Active/Projects/Backpack/config/open_config_" + enzyme + ".json"}
    
    sed_file(dst, param2val)
    
def createAnnotateScript(src, dst, job_name, row_name):
    shutil.copyfile(src, dst)
    # create dictionary
    param2val = {"JOB_NAME" : job_name,
                 "ROW_NAME" : row_name}
    
    sed_file(dst, param2val)
    
def createCleanScript(src, dst, job_name, row_name):
    shutil.copyfile(src, dst)
    # create dictionary
    param2val = {"JOB_NAME" : job_name,
                 "ROW_NAME" : row_name}
    
    sed_file(dst, param2val)

def sed_file(path, param2val):
    for line in fileinput.input(path, inplace=True):
        for p in param2val:
            line = line.replace(p, param2val[p])
        print(line.strip())




# get basename
sheet_name = Path(args.sheet_path).stem

# Create folder on scratch -> Download_Timestamp
job_name = sheet_name #datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + "_" + sheet_name
scratch_path = os.path.join("/scratch1/fs1/d.goldfarb/Backpack/", job_name)
os.makedirs(os.path.join(scratch_path, "scripts"), exist_ok=True)
os.makedirs(os.path.join(scratch_path, "logs"), exist_ok=True)

with open(args.sheet_path, 'r') as infile:
    reader = csv.reader(infile, delimiter="\t")
    header = next(reader)

    index_PXD = header.index("PXD")
    index_enzyme = header.index("Enzyme")
    index_database = header.index("Database")
    index_include = header.index("Include")
    index_exclude = header.index("Exclude")

    job_index2job_id = dict()
    
    for i, row in enumerate(reader):
        print("Processing row:", row[index_PXD], row[index_enzyme], row[index_database], row[index_include], row[index_exclude])
        
        PXD = os.path.basename(os.path.normpath(row[index_PXD]))
        row_name = "row"+str(i+2)+"_"+PXD
        src = "/storage1/fs1/d.goldfarb/Active/Projects/Backpack/RIS/Public/download_template.bsub"
        download_script_path = os.path.join(scratch_path, "scripts", "download" + "_" + str(i+2) + "_" + PXD + ".bsub")
        
        createDownloadScript(src, download_script_path, job_name, row[index_PXD], row[index_include], row[index_exclude], row_name)
        
        job_index = i % args.parallel_downloads
        
        # submit download job
        if i >= args.parallel_downloads:
            job_id = submitJob(download_script_path, job_index2job_id[job_index])
        else:
            job_id = submitJob(download_script_path)
        print("Submitted download job for row " + str(i+2) + " " + PXD + " with job ID:", job_id)
        job_index2job_id[job_index] = job_id
        
        
        # submit convert job
        src = "/storage1/fs1/d.goldfarb/Active/Projects/Backpack/RIS/Public/convert_template.bsub"
        convert_script_path = os.path.join(scratch_path, "scripts", "convert_raw" + "_" + str(i+2) + "_" + PXD + ".bsub")
        createConvertScript(src, convert_script_path, job_name, row_name)
        job_id_convert = submitJob(convert_script_path, str(job_id))
        print("Submitted convert job for row " + str(i+2) + " " + PXD + " with job ID:", job_id_convert)
        
        # submit sage job
        src = "/storage1/fs1/d.goldfarb/Active/Projects/Backpack/RIS/Public/sage_template.bsub"
        sage_script_path = os.path.join(scratch_path, "scripts", "sage" + "_" + str(i+2) + "_" + PXD + ".bsub")
        createSageScript(src, sage_script_path, job_name, row_name, row[index_database], row[index_enzyme])
        job_id_sage = submitJob(sage_script_path, str(job_id_convert))
        #job_id_sage = submitJob(sage_script_path)
        print("Submitted sage job for row " + str(i+2) + " " + PXD + " with job ID:", job_id_sage)
        
        # submit annotate job
        src = "/storage1/fs1/d.goldfarb/Active/Projects/Backpack/RIS/Public/annotate_template.bsub"
        annotate_script_path = os.path.join(scratch_path, "scripts", "annotate" + "_" + str(i+2) + "_" + PXD + ".bsub")
        createAnnotateScript(src, annotate_script_path, job_name, row_name)
        job_id_annotate = submitJob(annotate_script_path, str(job_id_sage))
        #job_id_annotate = submitJob(annotate_script_path)
        print("Submitted annotate job for row " + str(i+2) + " " + PXD + " with job ID:", job_id_annotate)
        
        # submit cleanup job
        src = "/storage1/fs1/d.goldfarb/Active/Projects/Backpack/RIS/Public/clean_template.bsub"
        clean_script_path = os.path.join(scratch_path, "scripts", "clean" + "_" + str(i+2) + "_" + PXD + ".bsub")
        createCleanScript(src, clean_script_path, job_name, row_name)
        job_id_clean = submitJob(clean_script_path, str(job_id_annotate))
        print("Submitted clean-up job for row " + str(i+2) + " " + PXD + " with job ID:", job_id_clean)

