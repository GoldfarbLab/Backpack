import sys
import csv

peptides = set()
with open(sys.argv[1], newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    header = next(reader)
    for row in reader:
        if 'U' not in row[1] and 'X' not in row[1] and 'Z' not in row[1]:
            peptides.add(row[1])
    
with open(sys.argv[2], 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(["PeptideModSeq"])
    for peptide in peptides:
         writer.writerow([peptide])