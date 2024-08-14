# -*- coding: utf-8 -*-
"""
Spyder Editor
Author: Jason Russell.
Script to perform Voxel-wise linear regression between FEOBV and age
"""

from nilearn import image
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
import pandas as pd
import os
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn.glm.second_level import non_parametric_inference
from nilearn.image import new_img_like
from matplotlib.backends.backend_pdf import PdfPages as pdf


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
	'DST3050042/swFEOBV.nii', 'DST3050045/swFEOBV.nii',
	'DST3050052/swFEOBV.nii', 'DST3050059/swFEOBV.nii',
	'DST3050060/swFEOBV.nii', 'DST3050061/swFEOBV.nii',
	'DST3050071/swFEOBV.nii'
]

# Load NIfTI images into a 4D image
FEOBV_imgs = image.concat_imgs([os.path.join(data_path, img) for img in FEOBV_img_paths])

# Import age data
age_df = pd.read_csv('/home/jason/Study_data/Down Syndrome/TRCDS/DSCHOL_dems.csv')

age = age_df['dems_age'].astype(float)

dimx, dimy, dimz, subjects = FEOBV_imgs.shape


design_matrix = pd.DataFrame({
	"age": age,
	"intercept": np.ones(subjects)
})


# Load the study-specific GM mask
gmmask_path = '/home/jason/Study_data/Down Syndrome/TRCDS/Raw_images/DSCHOL-A003-2024-07-15a/DATA/study_specific_GM_mask_prob0.3.nii'

#second level model
second_level_model = SecondLevelModel(
	mask_img=gmmask_path, n_jobs=1
)
second_level_model = second_level_model.fit(
	FEOBV_img_paths, design_matrix=design_matrix)


#calculate zmaps
z_map = second_level_model.compute_contrast(
	second_level_contrast=[1, 0],
	output_type="z_score",
)


# Perform statsmap, correct for multiple comparisons
thresholded_map, threshold = threshold_stats_img(z_map,
												 alpha=threshold_1,
												 cluster_threshold=50)


# Save the statistical map
# Save the thresholded z-map to a NIfTI file
thresholded_map.to_filename(f'{output_path}/thresholded_age_effect_z_map.nii')

#perform non parametric inference
corrected_map = non_parametric_inference(
	FEOBV_img_paths,
	design_matrix=design_matrix,
	second_level_contrast=[1,0],
	mask=gmmask_path,
	n_perm=permutations,
	two_sided_test=True,
	n_jobs=1,
	threshold=threshold_non_para
	)

# extract cluster significance <0.05
img_data_non_para = corrected_map['logp_max_size'].get_fdata()
img_data_non_para[img_data_non_para < cluster_thres] = 0
img_data_non_para_mask = img_data_non_para != 0
thresholded_map_np = np.where(img_data_non_para_mask, img_data_non_para, np.nan)

thresholded_map_np_ni = new_img_like('DST3050001/swFEOBV.nii', thresholded_map_np)

# Save non-parametric inference corrected map
thresholded_map_np_ni.to_filename(f'{output_path}/Age_glm_non_parametric_inference_corrected_logP_map.nii')

# Generate pdf report
pdf_filename = "Age_FEOBV_GLM.pdf"

fig, axs = plt.subplots(3,1, figsize=(10,14))

plotting.plot_stat_map(
	thresholded_map,
	threshold=threshold,
	colorbar=True,
	cut_coords=6,
	display_mode="x",
	figure=fig,
	title = "GLM output p < 0.005, cluster size 50 (z-scores)",
	axes=axs[0]
)

plotting.plot_stat_map(
	corrected_map['logp_max_size'],
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
	corrected_map['logp_max_mass'],
	colorbar=True,
	vmax=-np.log10(1 / permutations),
	threshold = cluster_thres,
	cut_coords=6,
	display_mode="x",
	figure=fig,
	title = "GLM output p < 0.005, non-parametic inference, cluster mass (cluster logP)",
	axes=axs[2]
)

fig.suptitle("Association Between Age and FEOBV uptake", fontsize=16,
			 weight='bold')

pdf.savefig(f'{output_path}/{pdf_filename}', dpi=300)
plt.close()




