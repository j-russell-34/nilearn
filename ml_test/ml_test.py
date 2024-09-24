from nilearn.decoding import DecoderRegressor
import os
from nilearn import image
import glob
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error

in_dir = '/home/jason/INPUTS/test_data'

out_dir = '/home/jason/OUTPUTS/ml_test'

#import csv containing amyloid suvrs
amyloid = pd.read_csv(f'{out_dir}/suvrs.csv')

amyloid_suvrs = amyloid.T.iloc[1:,:].to_numpy()

y = amyloid_suvrs

fmri_ls = []

#import images to list
for subject in sorted(os.listdir(in_dir)):
	if subject.startswith('.'):
		# ignore hidden files and other junk
		continue
	
	subject_path = f'{in_dir}/{subject}'
	
	smoothed_fmri_img = image.smooth_img(
		glob.glob(f'{subject_path}/*MR1/scans/801/*.nii.gz'), fwhm=5
		)
	
	fmri_ls.append(smoothed_fmri_img[0])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(fmri_ls, y, test_size=0.25, random_state=42)



mask_img = image.load_img(f'{out_dir}/Brain_mask_prob0_3.nii.gz')

decoder = DecoderRegressor(
	estimator='ridge',
	mask=mask_img,
	smoothing_fwhm=5,
	standardize="zscore_sample",
	screening_percentile=5,
	scoring ="negative_mean_squared_error")

decoder.fit(X_train, y_train)

# Predict amyloid SUVRs on the test set using X_test (fMRI files)
y_pred = decoder.predict(X_test)

# Evaluate the performance on the test set
mse = mean_squared_error(y_test, y_pred)
print(f'Test Set Mean Squared Error: {mse}')

