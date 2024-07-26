# -*- coding: utf-8 -*-
"""
Spyder Editor
Author: Jason Russell.
Script to perform Voxel-wise linear regression between FEOBV and tau pet
"""


from nilearn import image
import os
import numpy as np
from scipy.stats import pearsonr
from nilearn.image import new_img_like
from nilearn import plotting
import nibabel as nib
import pandas as pd

#set significant p-value threshold for masking
sig_p = 0.01

#Set path where data is stored
data_path = '/home/jason/Study_data/Down Syndrome/TRCDS/Raw_images/DSCHOL-A003-2024-07-15a/DATA'

os.chdir(data_path)


#import study specific GM mask
gmmask = image.get_data('study_specific_GM_mask_prob0.3.nii').astype(bool)

#import tau Braak ROI SUVRs
braak_df = pd.read_csv('/home/jason/Study_data/Down Syndrome/TRCDS/DSCHOL_MK6240_BraakCombo.csv')


#Import smoothed/warped nifti files to generate 4D arrays
FEOBV_data = image.get_data(['DST3050001/swFEOBV.nii', 'DST3050002/swFEOBV.nii',
                           'DST3050003/swFEOBV.nii', 'DST3050012/swFEOBV.nii',
                           'DST3050033/swFEOBV.nii', 'DST3050041/swFEOBV.nii',
                           'DST3050042/swFEOBV.nii', 'DST3050045/swFEOBV.nii'])

#get shapes of 4D arrays
dim1, dim2, dim3, subjects = FEOBV_data.shape

# Initialize arrays to store regression coefficients, p-values and intercepts
correlations_b12 = np.zeros((dim1, dim2, dim3))
p_values_b12 = np.ones((dim1, dim2, dim3))

# Perform linear regression for each cell for Braak stage 12
for i in range(dim1):
    for j in range(dim2):
        for k in range(dim3):
            # Create and fit the model and extract correlation and p-value
            correlations_b12[i, j, k], p_values_b12[i, j, k] = pearsonr(
                FEOBV_data[i,j,k,:], braak_df['stage12'])  

            
# Initialize arrays to store regression coefficients, p-values and intercepts
correlations_b34 = np.zeros((dim1, dim2, dim3))
p_values_b34 = np.ones((dim1, dim2, dim3))

# Perform linear regression for each cell for Braak stage 34
for i in range(dim1):
    for j in range(dim2):
        for k in range(dim3):
            # Create and fit the model and extract correlation and p-value
            correlations_b34[i, j, k], p_values_b34[i, j, k] = pearsonr(
                FEOBV_data[i,j,k,:], braak_df['stage34']) 
            
# Initialize arrays to store regression coefficients, p-values and intercepts
correlations_b56 = np.zeros((dim1, dim2, dim3))
p_values_b56 = np.ones((dim1, dim2, dim3))

# Perform linear regression for each cell for Braak stage 34
for i in range(dim1):
    for j in range(dim2):
        for k in range(dim3):
            # Create and fit the model and extract correlation and p-value
            correlations_b56[i, j, k], p_values_b56[i, j, k] = pearsonr(
                FEOBV_data[i,j,k,:], braak_df['stage56']) 
            

#apply gmmask to calculated coefficients
corr_brainb12 = np.where(gmmask, correlations_b12, np.nan)
corr_brainb34 = np.where(gmmask, correlations_b34, np.nan)
corr_brainb56 = np.where(gmmask, correlations_b56, np.nan)

#create nifti of all coefficients
correlations_nii_b12 = new_img_like('DST3050001/swFEOBV.nii', corr_brainb12)
nib.save(correlations_nii_b12, "voxel-based correlation coef unmasked b12.nii")

correlations_nii_b34 = new_img_like('DST3050001/swFEOBV.nii', corr_brainb34)
nib.save(correlations_nii_b34, "voxel-based correlation coef unmasked b34.nii")

correlations_nii_b56 = new_img_like('DST3050001/swFEOBV.nii', corr_brainb56)
nib.save(correlations_nii_b56, "voxel-based correlation coef unmasked b56.nii")


# Use a log scale for p-values
log_p_values_b12 = -np.log10(p_values_b12)
log_p_values_b34 = -np.log10(p_values_b34)
log_p_values_b56 = -np.log10(p_values_b56)
# NAN values to zero
log_p_values_b12[np.isnan(log_p_values_b12)] = 0.0
log_p_values_b34[np.isnan(log_p_values_b34)] = 0.0
log_p_values_b56[np.isnan(log_p_values_b56)] = 0.0

log_p_values_b12[log_p_values_b12 > 10.0] = 10.0
log_p_values_b34[log_p_values_b34 > 10.0] = 10.0
log_p_values_b56[log_p_values_b56 > 10.0] = 10.0
       
#generate mask if p<0.001
log_p_values_b12[log_p_values_b12 < -np.log10(sig_p)] = 0
log_p_values_b34[log_p_values_b34 < -np.log10(sig_p)] = 0
log_p_values_b56[log_p_values_b56 < -np.log10(sig_p)] = 0

# self-computed pval mask
bin_p_values_b12 = log_p_values_b12 != 0
bin_p_values_b34 = log_p_values_b34 != 0
bin_p_values_b56 = log_p_values_b56 != 0

bin_p_values_and_mask_b12 = np.logical_and(bin_p_values_b12, gmmask)
bin_p_values_and_mask_b34 = np.logical_and(bin_p_values_b34, gmmask)
bin_p_values_and_mask_b56 = np.logical_and(bin_p_values_b56, gmmask)

#Generate Nifti image type
sig_p_mask_img_b12 = new_img_like(
    'DST3050001/swPIB.nii', bin_p_values_and_mask_b12.astype(int)
)

sig_p_mask_img_b34 = new_img_like(
    'DST3050001/swPIB.nii', bin_p_values_and_mask_b34.astype(int)
)

sig_p_mask_img_b56 = new_img_like(
    'DST3050001/swPIB.nii', bin_p_values_and_mask_b56.astype(int)
)



#apply mask to calculated coefficients
coef_masked_b12 = np.where(bin_p_values_and_mask_b12, correlations_b12, np.nan) 
coef_masked_b34 = np.where(bin_p_values_and_mask_b34, correlations_b34, np.nan) 
coef_masked_b56 = np.where(bin_p_values_and_mask_b56, correlations_b56, np.nan)     


#create nifti of significant coefficients

sig_correl_b12 = new_img_like('DST3050001/swFEOBV.nii', coef_masked_b12)
sig_correl_b34 = new_img_like('DST3050001/swFEOBV.nii', coef_masked_b34)
sig_correl_b56 = new_img_like('DST3050001/swFEOBV.nii', coef_masked_b56)


#generate mosiac plot of significant coefficients
plotting.plot_stat_map(sig_correl_b12, display_mode="mosaic")
plotting.plot_stat_map(sig_correl_b34, display_mode="mosaic")
plotting.plot_stat_map(sig_correl_b56, display_mode="mosaic")

#export significant coefficients as nifti file
nib.save(sig_correl_b12, "braak 12 voxel-based correlation with feobv.nii")
nib.save(sig_correl_b34, "braak 34 voxel-based correlation with feobv.nii")
nib.save(sig_correl_b56, "braak 56 voxel-based correlation with feobv.nii")


