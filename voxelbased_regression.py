# -*- coding: utf-8 -*-
"""
Spyder Editor
Author: Jason Russell.
Script to perform Voxel-wise linear regression between 2 PET tracers
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



#Set path where data is stored
data_path = '/home/jason/Study_data/Down Syndrome/TRCDS/Raw_images/DSCHOL-A003-2024-07-15a/DATA'

os.chdir(data_path)


#import generic T1 in MNI space to generate mask
mrdata = datasets.fetch_icbm152_2009()
gmdata = mrdata['gm']
gmni = image.load_img(gmdata)

mrdim1, mrdim2, mrdim3 = gmni.shape

#restrict to voxels within the brain - NEED T1 MR IMAGE
#create brain mask
mask_img = compute_epi_mask(mrdata["gm"])
brainmask = image.get_data(mask_img).astype(bool)

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

PiB_data_resampled = np.zeros((mrdim1, mrdim2, mrdim3, subjects))
FEOBV_data_resampled = np.zeros((mrdim1, mrdim2, mrdim3, subjects))

#resample PET images to match MRI mask
for i in range(subjects):
    
    #select image
    pib_img = PiB_data[..., i]
    feobv_img = FEOBV_data[..., i]
    
    #convert to nifti for resampling
    pib_img_ni = new_img_like(
        'DST3050001/swPIB.nii', pib_img)
    feobv_img_ni = new_img_like(
        'DST3050001/swPIB.nii', feobv_img)
    
    #resample so voxel size matches MR
    resampled_pib_img = image.resample_to_img(pib_img_ni, gmni, 
                                              interpolation='linear')
    resampled_feobv_img = image.resample_to_img(pib_img_ni, gmni, 
                                              interpolation='linear')
    #convert back to array
    pib_array = image.get_data(resampled_pib_img)
    feobv_array = image.get_data(resampled_feobv_img)
    
    #store in 4D array by subject
    PiB_data_resampled[..., i] = pib_array
    FEOBV_data_resampled[..., i] = feobv_array
    
#get shape of resampled 4D arrays
#get shapes of 4D arrays
rsdim1, rsdim2, rsdim3, subjects = PiB_data_resampled.shape


# Initialize arrays to store regression coefficients, p-values and intercepts
coefficients = np.zeros((rsdim1, rsdim2, rsdim3))
intercepts = np.zeros((rsdim1, rsdim2, rsdim3))
p_values = np.ones((rsdim1, rsdim2, rsdim3))

# Perform linear regression for each cell
for i in range(rsdim1):
    for j in range(rsdim2):
        for k in range(rsdim3):
            x = PiB_data_resampled[i, j, k, :].reshape(-1, 1)
            y = FEOBV_data_resampled[i, j, k, :]
            
            # Skip if all values in x or y are constant
            if np.all(x == x[0]) or np.all(y == y[0]):
                continue
                  
            # Add a column of ones for the intercept term
            x_with_intercept = sm.add_constant(x)
            
            # Create and fit the model
            model = sm.OLS(y, x_with_intercept).fit()
                
            # Store the coefficients and intercepts
            coefficients[i, j, k] = model.params[1]
            intercepts[i, j, k] = model.params[0] 
            p_values[i, j, k] = model.pvalues[1]

# pull unmasked coefficient values and output nifti



#apply brainmask to calculated coefficients
coef_brain = np.where(brainmask, coefficients, np.nan)

#create nifti of all coefficients
coefficients_nii = new_img_like('DST3050001/swFEOBV.nii', coefficients)
nib.save(coefficients_nii, "voxel-based correlation coef unmasked.nii")

# cut in x-direction
sagittal = -25
# cut in y-direction
coronal = -37
# cut in z-direction
axial = -6

# coordinates displaying should be prepared as a list
cut_coords = [sagittal, coronal, axial]

# Use a log scale for p-values
log_p_values = -np.log10(p_values)
# NAN values to zero
log_p_values[np.isnan(log_p_values)] = 0.0
log_p_values[log_p_values > 10.0] = 10.0
       
#generate mask if p<0.001
log_p_values[log_p_values < 3] = 0

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
nib.save(sig_coefficients, "voxel-based correlation.nii")
