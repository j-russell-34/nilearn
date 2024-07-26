#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor
Author: Jason Russell.
Script to Generate Study Specific Mask from Individual T1s
"""

from nilearn import image
from nilearn.image import new_img_like
from nilearn.masking import compute_epi_mask
from nilearn import plotting
import nibabel as nib
import os
import numpy as np

#Set path where data is stored
data_path = '/home/jason/Study_data/Down Syndrome/TRCDS/Raw_images/DSCHOL-A003-2024-07-15a/DATA'

os.chdir(data_path)

#Import individual nifti masks in a 4D array
ind_masks = image.get_data(['DSCHOL-A003-ROI-gm/DST3050001/wROI_gm.nii', 
                            'DSCHOL-A003-ROI-gm/DST3050002/wROI_gm.nii',
                           'DSCHOL-A003-ROI-gm/DST3050003/wROI_gm.nii', 
                           'DSCHOL-A003-ROI-gm/DST3050012/wROI_gm.nii',
                           'DSCHOL-A003-ROI-gm/DST3050033/wROI_gm.nii', 
                           'DSCHOL-A003-ROI-gm/DST3050041/wROI_gm.nii',
                           'DSCHOL-A003-ROI-gm/DST3050042/wROI_gm.nii', 
                           'DSCHOL-A003-ROI-gm/DST3050045/wROI_gm.nii',
                           'DSCHOL-A003-ROI-gm/DST3050052/wROI_gm.nii', 
                           'DSCHOL-A003-ROI-gm/DST3050059/wROI_gm.nii',
                           'DSCHOL-A003-ROI-gm/DST3050060/wROI_gm.nii', 
                           'DSCHOL-A003-ROI-gm/DST3050061/wROI_gm.nii',
                           'DSCHOL-A003-ROI-gm/DST3050071/wROI_gm.nii'])

#Generate single array for value with probability voxel is GM per participant
#get shapes of 4D arrays
rsdim1, rsdim2, rsdim3, subjects = ind_masks.shape

# Initialize array to store probabilities
prob_mask = np.zeros((rsdim1, rsdim2, rsdim3))

#Itterate through array calculating probability of voxel being GM
# Perform linear regression for each cell
for i in range(rsdim1):
    for j in range(rsdim2):
        for k in range(rsdim3):
            if (np.sum(ind_masks[i, j, k, :]))/subjects >= 0.3:
                prob_mask[i, j, k] = 1
            else:
                prob_mask[i, j, k] = 0
                
study_mask_nii = new_img_like('DST3050001/swFEOBV.nii', prob_mask)

nib.save(study_mask_nii, "study_specific_GM_mask_prob0.3.nii")
            


            