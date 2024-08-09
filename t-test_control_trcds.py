#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:59:50 2024

@author: jason
"""

from nilearn import image
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
import pandas as pd
import os
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt

#Set path where data is stored
data_path = '/home/jason/Study_data/Down Syndrome/TRCDS/Raw_images/DSCHOL-A003-2024-07-15a/DATA'

os.chdir(data_path)

# Load the study-specific whole brain mask
wbmask_path = 

# Load PET images
# DS data image paths
trcds_img_paths = [
    ]

# Control cohort image paths
control_img_paths = [
    ]

# Load DS images to 4D nifti
trcds_img = image.concat_imgs([os.path.join(data_path, img) for img in trcds_img_paths])

# Load Control images to 4d nifti
control_img = image.concat_imgs([os.path.join(data_path, img) for img in control_img_paths])

# Count subjects for each group
_, _, _, subjects_ds = trcds_img.shape
_, _, _, subjects_cx = control_img.shape

# Generate design matrix  
unpaired_design_matrix = pd.DataFrame({
    "Down Syndrome": np.concatenate([np.ones(subjects_ds), np.zeros(subjects_cx)])
})

# concat trcds and control images into 1 4D image
all_imgs = image.concat_imgs([trcds_img, control_img])

# second level model
second_level_model = SecondLevelModel(mask_img=wbmask_path, n_jobs=1).fit(
    all_imgs, design_matrix=unpaired_design_matrix
    )

# calculate contrast
stat_map = second_level_model_unpaired.compute_contrast(
    "Down Syndrome", output_type="all"
)

# Perform statsmap, correct for multiple comparisons
thresholded_map, threshold = threshold_stats_img(stat_map, alpha=0.005)

fig = plt.figure(figsize=(5, 3))
display = plotting.plot_stat_map(
    z_map,
    threshold=threshold,
    colorbar=True,
    display_mode="z",
    figure=fig,
)
fig.suptitle("Groupwise t-test (p = 0.005)")
plotting.show()

# Save the thresholded z-map to a NIfTI file
thresholded_map.to_filename('thresholded_groupwise_comparison_z_map.nii')


