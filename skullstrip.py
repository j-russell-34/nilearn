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

extracted_raw = brain_extraction(raw, modality='t1')

#produce NifTi
ants.image_write(extracted_raw, "skull_stripped.nii.gz")
