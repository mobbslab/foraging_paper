import nibabel as nb
import os
import re
from nilearn.plotting import plot_stat_map, plot_roi
import numpy as np
from nltools.data import Brain_Data, Design_Matrix
from nltools.mask import expand_mask

# Get data
base_dir = '../data/derivatives/'
betas_dir = os.path.join(base_dir, 'rsa_revised', 'searchlight_betas')

sub_beta_files = [i for i in os.listdir(betas_dir) if 'sub' in i and 'FSL' in i and 'numba' in i]

print('{0} subjects found'.format(len(sub_beta_files)))

sub_betas = {'comp_diff': [], 
             'survival_value_diff': [], 
             'threat': [], 
             'comp_diff': [], 
             'comp_current': [],  
             'comp_alternative': [], 
             'survival_value_diff': [], 
             'survival_value_current': [], 
             'survival_value_alternative': []}

subs = []

for f in sorted(sub_beta_files):
    sub = re.findall('for[0-9]{2}', f)[0]
    nii = nb.load(os.path.join(betas_dir, f))
    sub_betas['comp_diff'].append(nii.slicer[..., 3])
    sub_betas['comp_current'].append(nii.slicer[..., 4])
    sub_betas['comp_alternative'].append(nii.slicer[..., 5])
    sub_betas['survival_value_diff'].append(nii.slicer[..., 6])
    sub_betas['survival_value_current'].append(nii.slicer[..., 7])
    sub_betas['survival_value_alternative'].append(nii.slicer[..., 8])
    sub_betas['threat'].append(nii.slicer[..., 9])
    subs.append(sub)

for k, v in sub_betas.items():
    if len(v):
        sub_betas[k] = Brain_Data(v)

# Save
if not os.path.exists('../data/derivatives/rsa_revised/second_level/4d_niftis/'):
    os.makedirs('../data/derivatives/rsa_revised/second_level/4d_niftis/')

nb.save(sub_betas['comp_diff'].to_nifti(), '../data/derivatives/rsa_revised/second_level/4d_niftis/comp_diff.nii.gz')
nb.save(sub_betas['comp_current'].to_nifti(), '../data/derivatives/rsa_revised/second_level/4d_niftis/comp_current.nii.gz')
nb.save(sub_betas['comp_alternative'].to_nifti(), '../data/derivatives/rsa_revised/second_level/4d_niftis/comp_alternative.nii.gz')
nb.save(sub_betas['survival_value_diff'].to_nifti(), '../data/derivatives/rsa_revised/second_level/4d_niftis/survival_value_diff.nii.gz')
nb.save(sub_betas['survival_value_current'].to_nifti(), '../data/derivatives/rsa_revised/second_level/4d_niftis/survival_value_current.nii.gz')
nb.save(sub_betas['survival_value_alternative'].to_nifti(), '../data/derivatives/rsa_revised/second_level/4d_niftis/survival_value_alternative.nii.gz')
nb.save(sub_betas['threat'].to_nifti(), '../data/derivatives/rsa_revised/second_level/4d_niftis/threat.nii.gz')
