import torch
import os
from os.path import join

CURRENT_DIR = "/gpfs/project/alkro105/"

new_dict = {}
all_files = os.listdir(join(CURRENT_DIR, "all_transporter_sequences"))

for file in all_files:
    try:
        rep = torch.load(join(CURRENT_DIR, "all_transporter_sequences", file))
        new_dict[file] = rep["mean_representations"][33].numpy()
    except:
        print(file)

torch.save(new_dict, join(CURRENT_DIR, "all_transporter_sequences.pt"))