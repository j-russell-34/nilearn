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
from nilearn.image import math_img
from nilearn.glm.second_level import non_parametric_inference
from nilearn.image import new_img_like
from matplotlib.backends.backend_pdf import PdfPages


#Set path where data is stored
data_path = '/home/jason/Study_data/Down Syndrome/TRCDS/Raw_images/DSCHOL-A003-2024-07-15a/DATA'

os.chdir(data_path)
output_path = '/home/jason/Study_data/Down Syndrome/Outputs'

#set signficant thresholds for different tests
#significance (p-val) for initial test at cluster threshold 50
threshold_1 = 0.005
threshold_non_para = 0.005
#significance of clusters following non-parametric inference
cluster_thres = -np.log10(0.05)

#set number of permutations for non-parametric inference (10000 when finalized
# but adds compute time, 500 for running on computer)
permutations = 10000


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
thresholded_map12, threshold12 = threshold_stats_img(z_map12,
													 alpha=threshold_1,
													 cluster_threshold=50)

thresholded_map34, threshold34 = threshold_stats_img(z_map34,
													 alpha=threshold_1,
													 cluster_threshold=50)

thresholded_map56, threshold56 = threshold_stats_img(z_map56,
													 alpha=threshold_1,
													 cluster_threshold=50)


# Save the statistical map
# Save the thresholded z-map to a NIfTI file
thresholded_map12.to_filename(f'{output_path}/thresholded_braak12_effect_z_map.nii')
thresholded_map34.to_filename(f'{output_path}/thresholded_braak34_effect_z_map.nii')
thresholded_map56.to_filename(f'{output_path}/thresholded_braak56_effect_z_map.nii')

#perform non parametric inference
corrected_map12 = non_parametric_inference(
	FEOBV_img_paths,
	design_matrix=design_matrix12,
	second_level_contrast=[1,0],
	mask=gmmask_path,
	n_perm=permutations,
	two_sided_test=True,
	n_jobs=1,
	threshold=threshold_non_para
	)

corrected_map34 = non_parametric_inference(
	FEOBV_img_paths,
	design_matrix=design_matrix34,
	second_level_contrast=[1,0],
	mask=gmmask_path,
	n_perm=permutations,
	two_sided_test=True,
	n_jobs=1,
	threshold=threshold_non_para
	)

corrected_map56 = non_parametric_inference(
	FEOBV_img_paths,
	design_matrix=design_matrix56,
	second_level_contrast=[1,0],
	mask=gmmask_path,
	n_perm=permutations,
	two_sided_test=True,
	n_jobs=1,
	threshold=threshold_non_para
	)

# extract cluster significance <0.05
img_data_non_para12 = corrected_map12['logp_max_size'].get_fdata()
img_data_non_para12[img_data_non_para12 < cluster_thres] = 0
img_data_non_para_mask12 = img_data_non_para12 != 0
thresholded_map_np12 = np.where(img_data_non_para_mask12, img_data_non_para12, np.nan)

thresholded_map_np_12ni = new_img_like('DST3050001/swFEOBV.nii', thresholded_map_np12)

img_data_non_para34 = corrected_map34['logp_max_size'].get_fdata()
img_data_non_para34[img_data_non_para34 < cluster_thres] = 0
img_data_non_para_mask34 = img_data_non_para34 != 0
thresholded_map_np34 = np.where(img_data_non_para_mask34, img_data_non_para34, np.nan)

thresholded_map_np_34ni = new_img_like('DST3050001/swFEOBV.nii', thresholded_map_np34)

img_data_non_para56 = corrected_map56['logp_max_size'].get_fdata()
img_data_non_para56[img_data_non_para56 < cluster_thres] = 0
img_data_non_para_mask56 = img_data_non_para56 != 0
thresholded_map_np56 = np.where(img_data_non_para_mask56, img_data_non_para56, np.nan)

thresholded_map_np_56ni = new_img_like('DST3050001/swFEOBV.nii', thresholded_map_np56)

# Save non-parametric inference corrected map
thresholded_map_np_12ni.to_filename(
	f'{output_path}/Braak12_glm_non_parametric_inference_corrected_logP_map.nii')
thresholded_map_np_34ni.to_filename(
	f'{output_path}/Braak34_glm_non_parametric_inference_corrected_logP_map.nii')
thresholded_map_np_56ni.to_filename(
	f'{output_path}/Braak56_glm_non_parametric_inference_corrected_logP_map.nii')

# Generate pdf report
pdf_filename = f"{output_path}/Braak_FEOBV_GLM.pdf"

with PdfPages(pdf_filename) as pdf:

	fig, axs = plt.subplots(3,1, figsize=(10,14))
	
	plotting.plot_stat_map(
		thresholded_map12,
		threshold=threshold12,
		colorbar=True,
		cut_coords=6,
		display_mode="x",
		figure=fig,
		title = "GLM output p < 0.005, cluster size 50 (z-scores)",
		axes=axs[0]
	)
	
	plotting.plot_stat_map(
		corrected_map12['logp_max_size'],
		colorbar=True,
		vmax=-np.log10(1 / permutations),
		threshold = cluster_thres,
		cut_coords=6,
		display_mode="x",
		figure=fig,
		title = "GLM output p < 0.005, non-parametic inference, cluster size (cluster logP)",
		axes=axs[1]
	)
	
	plotting.plot_stat_map(
		corrected_map12['logp_max_mass'],
		colorbar=True,
		vmax=-np.log10(1 / permutations),
		threshold = cluster_thres,
		cut_coords=6,
		display_mode="x",
		figure=fig,
		title = "GLM output p < 0.005, non-parametic inference, cluster mass (cluster logP)",
		axes=axs[2]
	)
	
	fig.suptitle("Association Between Braak stage 1 and 2 ROI SUVR and FEOBV uptake", 
				 fontsize=16,
				 weight='bold')
	
	pdf.savefig(fig, dpi=300)
	plt.close(fig)
	
	fig, axs = plt.subplots(3,1, figsize=(10,14))
	
	plotting.plot_stat_map(
		thresholded_map34,
		threshold=threshold34,
		colorbar=True,
		cut_coords=6,
		display_mode="x",
		figure=fig,
		title = "GLM output p < 0.005, cluster size 50 (z-scores)",
		axes=axs[0]
	)
	
	plotting.plot_stat_map(
		corrected_map34['logp_max_size'],
		colorbar=True,
		vmax=-np.log10(1 / permutations),
		threshold = cluster_thres,
		cut_coords=6,
		display_mode="x",
		figure=fig,
		title = "GLM output p < 0.005, non-parametic inference, cluster size (cluster logP)",
		axes=axs[1]
	)
	
	plotting.plot_stat_map(
		corrected_map34['logp_max_mass'],
		colorbar=True,
		vmax=-np.log10(1 / permutations),
		threshold = cluster_thres,
		cut_coords=6,
		display_mode="x",
		figure=fig,
		title = "GLM output p < 0.005, non-parametic inference, cluster mass (cluster logP)",
		axes=axs[2]
	)
	
	fig.suptitle("Association Between Braak Stage 3 and 4 ROI SUVR and FEOBV uptake", 
				 fontsize=16,
				 weight='bold')
	
	pdf.savefig(fig, dpi=300)
	plt.close(fig)
	
	fig, axs = plt.subplots(3,1, figsize=(10,14))
	
	plotting.plot_stat_map(
		thresholded_map56,
		threshold=threshold56,
		colorbar=True,
		cut_coords=6,
		display_mode="x",
		figure=fig,
		title = "GLM output p < 0.005, cluster size 50 (z-scores)",
		axes=axs[0]
	)
	
	plotting.plot_stat_map(
		corrected_map56['logp_max_size'],
		colorbar=True,
		vmax=-np.log10(1 / permutations),
		threshold = cluster_thres,
		cut_coords=6,
		display_mode="x",
		figure=fig,
		title = "GLM output p < 0.005, non-parametic inference, cluster size (cluster logP)",
		axes=axs[1]
	)
	
	plotting.plot_stat_map(
		corrected_map56['logp_max_mass'],
		colorbar=True,
		vmax=-np.log10(1 / permutations),
		threshold = cluster_thres,
		cut_coords=6,
		display_mode="x",
		figure=fig,
		title = "GLM output p < 0.005, non-parametic inference, cluster mass (cluster logP)",
		axes=axs[2]
	)
	
	fig.suptitle("Association Between Braak Stage 5 and 6 ROI SUVR and FEOBV uptake", 
				 fontsize=16,
				 weight='bold')
	
	pdf.savefig(fig, dpi=300)
	plt.close(fig)





