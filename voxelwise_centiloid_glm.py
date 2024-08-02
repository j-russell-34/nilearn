#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Spyder Editor
Author: Jason Russell.
Script to perform Voxel-wise linear regression between FEOBV and centiloid
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
    'DST3050042/swFEOBV.nii', 'DST3050045/swFEOBV.nii',
    'DST3050052/swFEOBV.nii', 'DST3050059/swFEOBV.nii',
    'DST3050060/swFEOBV.nii', 'DST3050061/swFEOBV.nii',
    'DST3050071/swFEOBV.nii'
]

# Load NIfTI images into a 4D image
FEOBV_imgs = image.concat_imgs([os.path.join(data_path, img) for img in FEOBV_img_paths])

# Import centiloid data
centiloid_df = pd.read_csv('/home/jason/Study_data/Down Syndrome/TRCDS/DSCHOL_centiloids.csv')

centiloid = centiloid_df['Centiloid'].astype(float)

dimx, dimy, dimz, subjects = FEOBV_imgs.shape


design_matrix = pd.DataFrame({
    "centiloid": centiloid,
    "intercept": np.ones(subjects)
})


# Load the study-specific GM mask
gmmask_path = '/home/jason/Study_data/Down Syndrome/TRCDS/Raw_images/DSCHOL-A003-2024-07-15a/DATA/study_specific_GM_mask_prob0.3.nii'

#second level model
second_level_model = SecondLevelModel(
    mask_img=gmmask_path, n_jobs=2
)
second_level_model = second_level_model.fit(
    FEOBV_img_paths, design_matrix=design_matrix)


#calculate zmaps
z_map = second_level_model.compute_contrast(
    second_level_contrast=[1, 0],
    output_type="z_score",
)


# Perform statsmap, correct for multiple comparisons
thresholded_map, threshold = threshold_stats_img(z_map, alpha=0.005)

fig = plt.figure(figsize=(5, 3))
display = plotting.plot_stat_map(
    z_map,
    threshold=threshold,
    colorbar=True,
    display_mode="z",
    figure=fig,
)
fig.suptitle("centiloid effect on FEOBV uptake (p = 0.005)")
plotting.show()

# Save the statistical map
# Save the thresholded z-map to a NIfTI file
thresholded_map.to_filename('thresholded_centiloid_effect_z_map.nii')


