"""Generate the TEDLIUM 20% training subset and development set for fast model configuration."""
""" This script can also be used for generating the CALLHOME training subset and development set"""

from glob import glob
import numpy as np
from os import path
import shutil
from hyperparams import Hyperparams as hp

perc = 0.2      # percentage of train recordings to form the subset.
dev_add = 31    # number of additional development recordings to sample from train set.

train_sph_dir = hp.tedlium_src_dir
train_stm_dir = hp.tedlium_stm_dir

files = glob(path.join(train_sph_dir, "*.sph"))
n = len(files)
samp_files = np.random.choice(files, int(n * perc) + dev_add, replace=False)

train_dest_dir = hp.tedlium_trsub_src_dir
dev_dest_dir = hp.tedlium_dev_src_dir

train_samp_files = samp_files[0: int(n * perc)]
dev_samp_files = samp_files[int(n * perc):]

# Sample the additional development recordings.
for samp_file in dev_samp_files:
    shutil.copy(samp_file, dev_dest_dir)

    rec_name = path.splitext(path.basename(samp_file))[0]
    stm_file = path.join(train_stm_dir, rec_name + '.stm')
    shutil.copy(stm_file, path.join(path.dirname(dev_dest_dir), 'stm'))

# Sample train recordings.
for samp_file in train_samp_files:
    shutil.copy(samp_file, train_dest_dir)

    rec_name = path.splitext(path.basename(samp_file))[0]
    stm_file = path.join(train_stm_dir, rec_name + '.stm')
    shutil.copy(stm_file, path.join(path.dirname(train_dest_dir), 'stm'))
