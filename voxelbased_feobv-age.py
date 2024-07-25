#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spyder Editor
Author: Jason Russell.
Script to perform Voxel-wise linear regression between FEOBV and age
"""


from nilearn import image
from nilearn import datasets
import os
import numpy as np
import statsmodels.api as sm
from nilearn.image import new_img_like
from nilearn.masking import compute_epi_mask
from nilearn import plotting
import nibabel as nib
import pandas as pd

#set significant p-value threshold for masking
sig_p = 0.005

#Set path where data is stored
data_path = '/home/jason/Study_data/Down Syndrome/TRCDS/Raw_images/DSCHOL-A003-2024-07-15a/DATA'

os.chdir(data_path)

#import ages
age_df = pd.read_csv('/home/jason/Study_data/Down Syndrome/TRCDS/DSChol_dems.csv')


#import generic T1 in MNI space to generate mask
dataset = datasets.fetch_icbm152_2009()

#restrict to voxels within the brain - NEED T1 MR IMAGE
#create brain mask
mask_img = compute_epi_mask('DST3050061/swPIB.nii')
brainmask = image.get_data(mask_img).astype(bool)


#Import smoothed/warped nifti files to generate 4D arrays
FEOBV_data = image.get_data(['DST3050001/swFEOBV.nii', 'DST3050002/swFEOBV.nii',
                           'DST3050003/swFEOBV.nii', 'DST3050012/swFEOBV.nii',
                           'DST3050033/swFEOBV.nii', 'DST3050041/swFEOBV.nii',
                           'DST3050042/swFEOBV.nii', 'DST3050045/swFEOBV.nii',
                           'DST3050052/swFEOBV.nii', 'DST3050059/swFEOBV.nii',
                           'DST3050060/swFEOBV.nii', 'DST3050061/swFEOBV.nii',
                           'DST3050071/swFEOBV.nii'])

#get shapes of 4D arrays
dim1, dim2, dim3, subjects = FEOBV_data.shape

# Initialize arrays to store regression coefficients, p-values and intercepts
coefficients = np.zeros((dim1, dim2, dim3))
intercepts = np.zeros((dim1, dim2, dim3))
p_values = np.ones((dim1, dim2, dim3))

# Perform linear regression for each cell
for i in range(dim1):
    for j in range(dim2):
        for k in range(dim3):
            x = age_df['age']
            y = FEOBV_data[i, j, k, :]
            
            # Skip if all values in x or y are constant
            if np.all(x == x[0]) or np.all(y == y[0]):
                continue
                  
            # Add a column of ones for the intercept term
            x_with_intercept = sm.add_constant(x)
            
            # Create and fit the model
            model = sm.OLS(y, x_with_intercept).fit()
                
            # Store the coefficients and intercepts
            coefficients[i, j, k] = model.params.iloc[1]
            intercepts[i, j, k] = model.params.iloc[0] 
            p_values[i, j, k] = model.pvalues.iloc[1]

# pull unmasked coefficient values and output nifti



#apply brainmask to calculated coefficients
coef_brain = np.where(brainmask, coefficients, np.nan)

#create nifti of all coefficients
coefficients_nii = new_img_like('DST3050001/swFEOBV.nii', coefficients)
nib.save(coefficients_nii, "voxel-based correlation coef unmasked.nii")


# Use a log scale for p-values
log_p_values = -np.log10(p_values)
# NAN values to zero
log_p_values[np.isnan(log_p_values)] = 0.0
log_p_values[log_p_values > 10.0] = 10.0
       
#generate mask if p< sig p threshold defined earlier
log_p_values[log_p_values < -np.log10(sig_p)] = 0

# First argument being a reference image
# and second argument should be p-values data
# to convert to a new image as output.
# This new image will have same header information as reference image.   
#log_p_values_img = new_img_like('DST3050001/swFEOBV.nii', log_p_values)

# self-computed pval mask
bin_p_values = log_p_values != 0

bin_p_values_and_mask = np.logical_and(bin_p_values, brainmask)


#Generate Nifti image type
sig_p_mask_img = new_img_like(
    'DST3050001/swPIB.nii', bin_p_values_and_mask.astype(int)
)


#apply mask to calculated coefficients
coef_masked = np.where(bin_p_values_and_mask, coefficients, np.nan)     


#create nifti of significant coefficients

sig_coefficients = new_img_like('DST3050001/swFEOBV.nii', coef_masked)


#generate mosiac plot of significant coefficients
plotting.plot_stat_map(sig_coefficients, display_mode="mosaic")

#export significant coefficients as nifti file
nib.save(sig_coefficients, "voxel-based age correlation.nii")
