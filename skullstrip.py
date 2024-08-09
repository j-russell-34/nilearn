#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:00:45 2024

@author: jason
"""

import ants
import os
from antspynet import brain_extraction


#Set path where data is stored
data_path = '/home/jason/Study_data/Down Syndrome/TRCDS/Raw_images/'

os.chdir(data_path)

#import MRI
orig_file = 'orig.mgz'
raw = ants.image_read(orig_file)

extracted_mask = brain_extraction(raw, modality='t1')

# Apply the mask to the original MRI image
masked_image = ants.mask_image(raw, extracted_mask)

#produce NifTi
ants.image_write(masked_image, "skull_stripped.nii.gz")