#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Spyder Editor
Author: Jason Russell.
Script to perform Voxel-wise linear regression between FEOBV and Braak stages
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


# Load fMRI images
FEOBV_img_paths = [
    'DST3050001/swFEOBV.nii', 'DST3050002/swFEOBV.nii',
    'DST3050003/swFEOBV.nii', 'DST3050012/swFEOBV.nii',
    'DST3050033/swFEOBV.nii', 'DST3050041/swFEOBV.nii',
    'DST3050042/swFEOBV.nii', 'DST3050045/swFEOBV.nii'
]

# Load NIfTI images into a 4D image
FEOBV_imgs = image.concat_imgs([os.path.join(data_path, img) for img in FEOBV_img_paths])

# Import centiloid data
braak_df = pd.read_csv('/home/jason/Study_data/Down Syndrome/TRCDS/DSCHOL_MK6240_BraakCombo.csv')

braak12 = braak_df['stage12'].astype(float)
braak34 = braak_df['stage34'].astype(float)
braak56 = braak_df['stage56'].astype(float)

dimx, dimy, dimz, subjects = FEOBV_imgs.shape


design_matrix12 = pd.DataFrame({
    "stage12": braak12,
    "intercept": np.ones(subjects)
})

design_matrix34 = pd.DataFrame({
    "stage34": braak34,
    "intercept": np.ones(subjects)
})

design_matrix56 = pd.DataFrame({
    "stage56": braak56,
    "intercept": np.ones(subjects)
})

# Load the study-specific GM mask
gmmask_path = '/home/jason/Study_data/Down Syndrome/TRCDS/Raw_images/DSCHOL-A003-2024-07-15a/DATA/study_specific_GM_mask_prob0.3.nii'

#second level model
second_level_model_b12 = SecondLevelModel(
    mask_img=gmmask_path, n_jobs=1
)
second_level_model_b12 = second_level_model_b12.fit(
    FEOBV_img_paths, design_matrix=design_matrix12)

second_level_model_b34 = SecondLevelModel(
    mask_img=gmmask_path, n_jobs=1
)
second_level_model_b34 = second_level_model_b34.fit(
    FEOBV_img_paths, design_matrix=design_matrix34)

second_level_model_b56 = SecondLevelModel(
    mask_img=gmmask_path, n_jobs=1
)
second_level_model_b56 = second_level_model_b56.fit(
    FEOBV_img_paths, design_matrix=design_matrix56)


#calculate zmaps
z_map12 = second_level_model_b12.compute_contrast(
    second_level_contrast=[1, 0],
    output_type="z_score",
)

z_map34 = second_level_model_b34.compute_contrast(
    second_level_contrast=[1, 0],
    output_type="z_score",
)

z_map56 = second_level_model_b56.compute_contrast(
    second_level_contrast=[1, 0],
    output_type="z_score",
)


# Perform statsmap, correct for multiple comparisons
thresholded_map12, threshold12 = threshold_stats_img(z_map12, alpha=0.005)

thresholded_map34, threshold34 = threshold_stats_img(z_map34, alpha=0.005)

thresholded_map56, threshold56 = threshold_stats_img(z_map56, alpha=0.005)

fig = plt.figure(figsize=(5, 3))
display = plotting.plot_stat_map(
    z_map12,
    threshold=threshold12,
    colorbar=True,
    display_mode="z",
    figure=fig,
)
fig.suptitle("Braak stage 1, 2 effect on FEOBV uptake (p = 0.005)")
plotting.show()

fig = plt.figure(figsize=(5, 3))
display = plotting.plot_stat_map(
    z_map34,
    threshold=threshold34,
    colorbar=True,
    display_mode="z",
    figure=fig,
)
fig.suptitle("Braak stage 3, 4 effect on FEOBV uptake (p = 0.005)")
plotting.show()

fig = plt.figure(figsize=(5, 3))
display = plotting.plot_stat_map(
    z_map56,
    threshold=threshold56,
    colorbar=True,
    display_mode="z",
    figure=fig,
)
fig.suptitle("Braak stage 5, 6 effect on FEOBV uptake (p = 0.005)")
plotting.show()

# Save the statistical map
# Save the thresholded z-map to a NIfTI file
thresholded_map12.to_filename('thresholded_braak12_effect_z_map.nii')
thresholded_map34.to_filename('thresholded_braak34_effect_z_map.nii')
thresholded_map56.to_filename('thresholded_braak56_effect_z_map.nii')


