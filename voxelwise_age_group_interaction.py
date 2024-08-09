#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:06:35 2024

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

#import age
age = pd.read_csv

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
DS = np.concatenate([np.ones(subjects_ds), np.zeros(subjects_cx)])
interaction = DS * age

design_matrix = pd.DataFrame({
	"Down Syndrome": DS
	"Age":age
	"Interaction": interaction
	"Intercept": np.ones(subjects)
})

# concat trcds and control images into 1 4D image
all_imgs = image.concat_imgs([trcds_img, control_img])

second_level_model = SecondLevelModel(mask_img=wbmask_path, n_jobs=1).fit(
	all_imgs, design_matrix=design_matrix
	)

# calculate contrast
stat_map = second_level_model.compute_contrast(
	second_level_contrast=[0, 0, 1, 0],
	output_type="z_score"
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
fig.suptitle("age x group interaction on FEOBV uptake (p = 0.005)")
plotting.show()

# Save the statistical map
# Save the thresholded z-map to a NIfTI file
thresholded_map.to_filename('age_x_group_z_map.nii')

#perform glm to extract betas from age associations to understand direction
#of interaction effect

#select just ds or control age
age_ds = age.iloc[0:subjects_ds]
age_control = age.iloc[subjects_ds:subjects_ds+subjects_cx]

design_matrix_age_ds = pd.DataFrame({
	"age": age_ds,
	"intercept": np.ones(subjects)
})

design_matrix_age_control = pd.DataFrame({
	"age": age_control,
	"intercept": np.ones(subjects)
})

#calculate model ds
second_level_model_ds = SecondLevelModel(mask_img=wbmask_path, n_jobs=1).fit(
	trcds_img, design_matrix=design_matrix_age_ds
	)

# calculate contrast ds and save to file
stat_map_ds = second_level_model_ds.compute_contrast(
	second_level_contrast=[1, 0],
	output_type="effect_size"
)

stat_map_ds.to_filename('ds_age_effect_size_map.nii')

#calculate model control
second_level_model_cx = SecondLevelModel(mask_img=wbmask_path, n_jobs=1).fit(
	control_img, design_matrix=design_matrix_age_control
	)

# calculate contrast control and save to file
stat_map_cx = second_level_model_cx.compute_contrast(
	second_level_contrast=[1, 0],
	output_type="effect_size"
)

stat_map_cx.to_filename('control_age_effect_size_map.nii')




