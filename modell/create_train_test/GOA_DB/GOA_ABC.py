import pandas as pd
import numpy as np
from os.path import join
import os
import sys
import gzip
from Bio.UniProt.GOA import _gpa11iterator
CURRENT_DIR = "/gpfs/scratch/alkro105/go_data"
print(CURRENT_DIR)

df = pd.read_pickle(join(CURRENT_DIR, "df_GO_binding.pkl"))
abc_go_terms = list(set(df["GO ID"]))

run = int(sys.argv[1])

df_GO_UID = pd.DataFrame(columns = ["Uniprot ID", "GO Term", 'ECO_Evidence_code'])

overall_count = 0
filename = join(CURRENT_DIR, 'goa_uniprot_all.gpa.gz')
with gzip.open(filename, 'rt') as fp:
    for annotation in _gpa11iterator(fp):                 
        overall_count += 1
        if overall_count >= run*10**6 and overall_count < (run+1)*10**6:
            # Output annotated protein ID   
            UID = annotation['DB_Object_ID']
            GO_ID = annotation['GO_ID']
            ECO_Evidence_code = annotation["ECO_Evidence_code"]
            if GO_ID in abc_go_terms:
                df_GO_UID = df_GO_UID.append({"Uniprot ID" : UID, "GO Term" : GO_ID,
                                             'ECO_Evidence_code' : ECO_Evidence_code}, ignore_index = True)
                
df_GO_UID.to_pickle(join(CURRENT_DIR, "df_GO_UID_part_" + str(run) +".pkl"))    