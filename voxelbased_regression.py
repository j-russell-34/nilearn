# -*- coding: utf-8 -*-
"""
Spyder Editor
Author: Jason Russell.
Script to perform Voxel-wise linear regression between 2 PET tracers
"""


from nilearn import image
import os
import numpy as np
from scipy.stats import pearsonr
from nilearn.image import new_img_like
from nilearn import plotting
import nibabel as nib

#set significant p-value threshold for masking
sig_p = 0.005

#Set path where data is stored
data_path = '/home/jason/Study_data/Down Syndrome/TRCDS/Raw_images/DSCHOL-A003-2024-07-15a/DATA'

os.chdir(data_path)


#import study specific GM mask
gmmask = image.get_data('study_specific_GM_mask_prob0.3.nii').astype(bool)


#Import smoothed/warped nifti files to generate 4D arrays
PiB_data = image.get_data(['DST3050001/swPIB.nii', 'DST3050002/swPIB.nii',
                           'DST3050003/swPIB.nii', 'DST3050012/swPIB.nii',
                           'DST3050033/swPIB.nii', 'DST3050041/swPIB.nii',
                           'DST3050042/swPIB.nii', 'DST3050045/swPIB.nii',
                           'DST3050052/swPIB.nii', 'DST3050059/swPIB.nii',
                           'DST3050060/swPIB.nii', 'DST3050061/swPIB.nii',
                           'DST3050071/swPIB.nii'])

FEOBV_data = image.get_data(['DST3050001/swFEOBV.nii', 'DST3050002/swFEOBV.nii',
                           'DST3050003/swFEOBV.nii', 'DST3050012/swFEOBV.nii',
                           'DST3050033/swFEOBV.nii', 'DST3050041/swFEOBV.nii',
                           'DST3050042/swFEOBV.nii', 'DST3050045/swFEOBV.nii',
                           'DST3050052/swFEOBV.nii', 'DST3050059/swFEOBV.nii',
                           'DST3050060/swFEOBV.nii', 'DST3050061/swFEOBV.nii',
                           'DST3050071/swFEOBV.nii'])

#get shapes of 4D arrays
dim1, dim2, dim3, subjects = PiB_data.shape

# Initialize arrays to store regression coefficients, p-values and intercepts
correlations = np.zeros((dim1, dim2, dim3))
p_values = np.ones((dim1, dim2, dim3))

# Perform linear regression for each cell
for i in range(dim1):
    for j in range(dim2):
        for k in range(dim3):
           # Create and fit the model and extract correlation and p-value
           correlations[i, j, k], p_values[i, j, k] = pearsonr(
               PiB_data[i,j,k,:], FEOBV_data[i,j,k,:])     


#apply gmmask to calculated coefficients
correl_brain = np.where(gmmask, correlations, np.nan)

#create nifti of all coefficients
correlations_nii = new_img_like('DST3050001/swFEOBV.nii', correlations)
nib.save(correlations_nii, "voxel-based correlations unmasked.nii")


# Use a log scale for p-values
log_p_values = -np.log10(p_values)
# NAN values to zero
log_p_values[np.isnan(log_p_values)] = 0.0
log_p_values[log_p_values > 10.0] = 10.0
       
#generate mask if p< sig p threshold defined earlier
log_p_values[log_p_values < -np.log10(sig_p)] = 0


# self-computed pval mask
bin_p_values = log_p_values != 0

bin_p_values_and_mask = np.logical_and(bin_p_values, gmmask)


#Generate Nifti image type
sig_p_mask_img = new_img_like(
    'DST3050001/swPIB.nii', bin_p_values.astype(int)
)


#apply mask to calculated coefficients
cor_masked = np.where(bin_p_values_and_mask, correlations, np.nan)     


#create nifti of significant coefficients

sig_correlations = new_img_like('DST3050001/swFEOBV.nii', cor_masked)


#generate mosiac plot of significant coefficients
plotting.plot_stat_map(sig_correlations, display_mode="mosaic")

#export significant coefficients as nifti file
nib.save(sig_correlations, "voxel-based correlation.nii")
