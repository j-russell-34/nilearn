import nipype.interfaces.io as nio
from nipype.interfaces.spm import Realign, SliceTiming, Coregister, NewSegment, Normalize12, Smooth
from nipype import Node, Workflow, MapNode
import nipype.interfaces.utility as util
from nipype.algorithms.misc import Gunzip
import os
import glob
from nipype.interfaces.freesurfer import MRIConvert

anat_scans = []
func_scans = []

in_dir = '/home/jason/INPUTS/test_data'

for subject in sorted(os.listdir(in_dir)):
	anat = glob.glob(f'{in_dir}/{subject}/{subject}_MR1/assessors/*FS7_v1*/{subject}/*FS7_v1*/out/SUBJ/mri/orig.mgz')[0]
	anat_scans.append(anat)
	func = f'{in_dir}/{subject}/{subject}_MR1/scans/801/fMRI_REST1.nii.gz'
	func_scans.append(func)

inputnode = Node(util.IdentityInterface(fields=["in_file"]), name ="inputnode")
inputnode.inputs.in_file = func_scans

inputnodeanat = Node(util.IdentityInterface(fields=["in_file"]), name ="inputnodeanat")
inputnodeanat.inputs.in_file = anat_scans

from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('/home/jason/matlab_pkg/spm12')

#gunzip
gunzip_func = MapNode(Gunzip(), name='gunzip_func', iterfield=['in_file'])
gunzip_anat = MapNode(Gunzip(), name='gunzip_anat', iterfield=['in_file'])

#preprocess fMRI images using SPM
# Realign motion correction
realigner = Node(interface=Realign(), name = 'realign')
realigner.inputs.register_to_mean = True

# slice timing correction
slice_time = Node(SliceTiming(), name="slice_time")
slice_time.inputs.num_slices = 60
slice_time.inputs.time_repetition = 1.6
slice_time.inputs.time_acquisition = 1.6 - (1.6/60)
slice_time.inputs.slice_order = list(range(1, 61, 2)) + list(range(2, 61, 2))
slice_time.inputs.ref_slice = 30

# coregistration 

coreg = MapNode(Coregister(), iterfield=['target', 'apply_to_files'], name="coreg")
coreg.inputs.jobtype = 'estimate'

# segmentation
tpm_img = '/home/jason/matlab_pkg/spm12/tpm/TPM.nii'
tissue1 = ((tpm_img, 1), 1, (True, False), (False, False))
tissue2 = ((tpm_img, 2), 1, (True, False), (False, False))
tissue3 = ((tpm_img, 3), 2, (True, False), (False, False))
tissue4 = ((tpm_img, 4), 3, (False, False), (False, False))
tissue5 = ((tpm_img, 5), 4, (False, False), (False, False))
tissue6 = ((tpm_img, 6), 2, (False, False), (False, False))
tissues = [tissue1, tissue2, tissue3, tissue4, tissue5, tissue6]

segmentation = Node(NewSegment(tissues=tissues), name="segmentation")

# create node to convert mgz to nii
mri_convert = MapNode(MRIConvert(), name='mri_convert', iterfield=['in_file'])

# normalization
normalization = Node(Normalize12(), name="normalization")
normalization.inputs.tpm = '/home/jason/matlab_pkg/spm12/tpm/TPM.nii'

# smooth
smooth = Node(interface=Smooth(), name="smooth")
smooth.inputs.fwhm = 4

#datasink
datasink = Node(nio.DataSink(base_directory='/home/jason/OUTPUTS/ml_test'),
				name = 'datasink')

preproc = Workflow(name='preproc', base_dir = '/home/jason/OUTPUTS/ml_test')

preproc.connect([
	(inputnode,gunzip_func, [('in_file', 'in_file')]),
	(gunzip_func, realigner, [('out_file','in_files')]),
	(realigner, slice_time, [('realigned_files', 'in_files')]),
	(inputnodeanat, mri_convert, [('in_file', 'in_file')]),
	(mri_convert, gunzip_anat, [('out_file', 'in_file')]),
	(gunzip_anat, segmentation, [('out_file', 'channel_files')]),
	(gunzip_anat, coreg, [('out_file', 'target')]),
	(realigner, coreg, [('mean_image', 'source')]),
	(slice_time, coreg,[('timecorrected_files', 'apply_to_files')]),
	(segmentation, normalization, [('forward_deformation_field', 'deformation_file')]),
	(realigner, normalization, [('mean_image', 'image_to_align')]),
	(coreg, normalization, [('coregistered_files', 'apply_to_files')]),
	(normalization, smooth,[('normalized_files', 'in_files')]),
	(smooth, datasink, [('smoothed_files', 'smoothed')]),
	(normalization, datasink, [('normalized_files', 'normalized')]),
	(segmentation, datasink, [('bias_corrected_images', 'bias_corrected')]),
	(realigner, datasink, [('realignment_parameters', 'realignment_parameters')])
	 ])

preproc.run('MultiProc', plugin_args={'n_procs': 4})
