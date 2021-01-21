# Limit the number of threads used by numpy
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1" 

import os
import matplotlib.pyplot as plt
import nibabel as nb
from nilearn.image import new_img_like
import numpy as np
import re
from nilearn.image import resample_to_img
from sklearn.preprocessing import minmax_scale, scale
from scipy.spatial import distance_matrix
from nltools.data import Adjacency, Brain_Data
import seaborn as sns
import pandas as pd
import datetime
from .searchlight_functions import SearchLightRSA

np.random.seed(100)


base_dir = '../data/derivatives/'

# Get slurm run ID
try:
    runID = int(os.environ['SLURM_ARRAY_TASK_ID'])
except:
    runID = 1 # TESTING

# Select subject, session and run to be processed
subs = pd.read_csv('../subject_list.txt', header=None)
subject = subs[runID-1]


# Get first level estimates
beta_map_fnames = [i for i in os.listdir(os.path.join(base_dir, 
                                        'first_level/rsa_revised/fsl_betas_smoothed/sub-{0}'.format(subject))) 
                                        if 'blocked_with_decision' in i]

# Mask
brain_mask = nb.load(r'../data/derivatives/fmriprep/sub-{0}/anat/sub-{0}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(subject))                                

# Behavioural data
decision_data = pd.read_csv('../data/decision_data_REVISED.csv')

# Put conditions in order
predator_difference = np.array([int(re.search('[-]?[0-9]', i).group()) + int('threat' in i) * 10 + int('B' in i) * 100 for i in beta_map_fnames])
reordering_idx = np.argsort(predator_difference)
beta_map_fnames_ordered = [beta_map_fnames[i] for i in reordering_idx]

# Load first level data 
# Data is represented using Nltools Brain_Data format
first_level_betas = Brain_Data([os.path.join(base_dir, 
                                            'first_level/rsa_revised/fsl_betas_smoothed/sub-{0}/{1}'.format(subject, i)) 
                                            for i in beta_map_fnames_ordered])
first_level_betas_nii = first_level_betas.to_nifti()                                            

# Resample mask
# Mask comes from T1 which is higher resolution than functional images
brain_mask = resample_to_img(brain_mask, first_level_betas_nii)
brain_mask_data = brain_mask.get_data()
brain_mask_data = np.round(brain_mask_data, 0).astype(int)
brain_mask = new_img_like(brain_mask, brain_mask_data)

# Get condition info
labels = [re.search('(?<=blocked_with_decision_)[a-z]+', i).group() + ', ' + re.search('[-]?[0-9](?=.0)', i).group() for i in beta_map_fnames_ordered]
sessions = [re.search('(?<=-d)[12]', i).group() for i in beta_map_fnames_ordered]
runs = [re.search('(?<=_Run-)[0-9]', i).group() for i in beta_map_fnames_ordered]
# competitor_difference = np.array([re.search('[-]?[0-9]', i).group() for i in labels]).astype(int)

total_survival_value = []
survival_value_diff = []
now_or_later = []
chosen_sv = []
unchosen_sv = []
left_sv = []
right_sv = []
left_sv_var = []
right_sv_var = []
decisions = []
left_competitors = []
right_competitors = []

decision_data = decision_data[decision_data['subject'] == subject]
for day, block, id in [(re.search('(?<=-d)[12]', i).group(), re.search('(?<=_Run-)[0-9]', i).group(), re.search('(?<=_id-)[0-9]+', i).group()) for i in beta_map_fnames_ordered]:
    try:
        left, right, diff, later, decision, left_c, right_c = decision_data.loc[(decision_data['day'] == int(day)) & 
                                                               (decision_data['block'] == int(block)+1) & 
                                                               (decision_data['id'] == int(id)), ['survival_value_left_Z', 
                                                                                                  'survival_value_right_Z', 
                                                                                                  'survival_value_diff_Z', 
                                                                                                  'now_or_later', 
                                                                                                  'left_or_right',
                                                                                                  'left_conspecifics_number',
                                                                                                  'right_conspecifics_number']].values[0]
    except Exception as e: # Subject for21 is missing one condition (6 competitors, safe) so we don't know the expected value, fill in with NaNs
        left, right, diff, later, decision, left_c, right_c = (np.nan, np.nan, np.nan, True, np.nan, np.nan, np.nan)
    
    total_survival_value.append(left+right)
    survival_value_diff.append(diff)
    now_or_later.append(later)
    decisions.append(decision)
    left_competitors.append(left_c)
    right_competitors.append(right_c)
    if np.isnan(decision):
        chosen_sv.append(np.nan)
        unchosen_sv.append(np.nan)
        left_sv.append(np.nan)
        right_sv.append(np.nan)
    else:
        chosen_sv.append([left, right][int(decision)])
        unchosen_sv.append([left, right][1 - int(decision)])
        left_sv.append(left)
        right_sv.append(right)

decision_binary = np.maximum(0, np.array(decisions).astype(int))
decision_binary = np.vstack([1 -decision_binary, decision_binary]).astype(bool)
decision_binary_current = np.roll(decision_binary, 1, axis=1)  # Shift to next trial
decision_binary_alternative = (1 - decision_binary_current).astype(bool)

# SV
current_sv = np.vstack([left_sv, right_sv]).T[decision_binary_current.T]
alternative_sv = np.vstack([left_sv, right_sv]).T[decision_binary_alternative.T]
current_sv[0] = np.nan # No current/alternative on first trial
alternative_sv[0] = np.nan

current_competitors = np.vstack([left_competitors, right_competitors]).T[decision_binary_current.T]
alternative_competitors = np.vstack([left_competitors, right_competitors]).T[decision_binary_alternative.T]
current_competitors[0] = np.nan # No current/alternative on first trial
alternative_competitors[0] = np.nan

sv_diff = current_sv - alternative_sv
competitor_difference = current_competitors - alternative_competitors    

now_or_later = np.array(now_or_later) == False

threat_level = (np.array([re.search('[a-z]+', i).group() for i in labels]) == 'threat').astype(int)
unique_run = np.array([re.search('(?<=-d)[12]', i).group() + re.search('(?<=_Run-)[0-9]', i).group() for i in beta_map_fnames_ordered]).astype(float)
for n, i in enumerate(np.unique(unique_run)):
    unique_run[unique_run == i] = n

# Set up RSMs #

# Covariates #

# Matrix representing the unique run - used to ensure we ignore trials from the same run
unique_run_matrix = distance_matrix(unique_run[now_or_later, np.newaxis], unique_run[now_or_later, np.newaxis]) == 0
unique_run_matrix = Adjacency(unique_run_matrix)

# Session
session_matrix = distance_matrix(np.array(sessions).astype(float)[now_or_later, np.newaxis], np.array(sessions).astype(float)[now_or_later, np.newaxis])
plt.figure(figsize=(8, 8), dpi=100)
sns.heatmap(session_matrix, square=True)
session_matrix = Adjacency(session_matrix, matrix_type='similarity')
session_matrix.data = scale(session_matrix.data)

# Run
run_matrix = distance_matrix(np.array(runs).astype(float)[now_or_later, np.newaxis], np.array(runs).astype(float)[now_or_later, np.newaxis]) == 0
plt.figure(figsize=(8, 8), dpi=100)
sns.heatmap(run_matrix, square=True)
run_matrix = Adjacency(run_matrix, matrix_type='similarity')
run_matrix.data = scale(run_matrix.data)

# RSMs of interest #

# Competitor difference
competitor_difference_matrix = 1 - minmax_scale(distance_matrix(competitor_difference[now_or_later, np.newaxis], competitor_difference[now_or_later, np.newaxis]))
competitor_difference_matrix[np.eye(len(competitor_difference_matrix)).astype(bool)] = np.nan
competitor_difference_matrix = Adjacency(competitor_difference_matrix, matrix_type='similarity')
competitor_difference_matrix.data = scale(competitor_difference_matrix.data)

# Competitors in current patch
current_competitors_matrix = 1 - minmax_scale(distance_matrix(current_competitors[now_or_later, np.newaxis], current_competitors[now_or_later, np.newaxis]))
current_competitors_matrix[np.eye(len(current_competitors_matrix)).astype(bool)] = np.nan
current_competitors_matrix = Adjacency(current_competitors_matrix, matrix_type='similarity')
current_competitors_matrix.data = scale(current_competitors_matrix.data)

# Competitors in alternative patch
alternative_competitors_matrix = 1 - minmax_scale(distance_matrix(alternative_competitors[now_or_later, np.newaxis], alternative_competitors[now_or_later, np.newaxis]))
alternative_competitors_matrix[np.eye(len(alternative_competitors_matrix)).astype(bool)] = np.nan
alternative_competitors_matrix = Adjacency(alternative_competitors_matrix, matrix_type='similarity')
alternative_competitors_matrix.data = scale(alternative_competitors_matrix.data)

# Threat level
threat = threat_level
threat_matrix = 1 - minmax_scale(distance_matrix(threat[now_or_later, np.newaxis], threat[now_or_later, np.newaxis]))
threat_matrix[np.eye(len(threat_matrix)).astype(bool)] = np.nan
threat_matrix = Adjacency(threat_matrix, matrix_type='similarity')
threat_matrix.data = scale(threat_matrix.data)

# Socially adjusted value difference
survival_value_diff = sv_diff
survival_value_diff_matrix = 1 - minmax_scale(distance_matrix(survival_value_diff[now_or_later, np.newaxis], survival_value_diff[now_or_later, np.newaxis]))
survival_value_diff_matrix[np.eye(len(survival_value_diff_matrix)).astype(bool)] = np.nan
survival_value_diff_matrix = Adjacency(survival_value_diff_matrix, matrix_type='similarity')
survival_value_diff_matrix.data = scale(survival_value_diff_matrix.data)

# Current survival value
current_survival_value = np.array(current_sv)
current_survival_value_matrix = 1 - minmax_scale(distance_matrix(current_survival_value[now_or_later, np.newaxis], current_survival_value[now_or_later, np.newaxis]))
current_survival_value_matrix[np.eye(len(current_survival_value_matrix)).astype(bool)] = np.nan
current_survival_value_matrix = Adjacency(current_survival_value_matrix, matrix_type='similarity')
current_survival_value_matrix.data = scale(current_survival_value_matrix.data)

# Alternative patch survival value
alternative_survival_value = np.array(alternative_sv)
alternative_survival_value_matrix = 1 - minmax_scale(distance_matrix(alternative_survival_value[now_or_later, np.newaxis], alternative_survival_value[now_or_later, np.newaxis]))
alternative_survival_value_matrix[np.eye(len(alternative_survival_value_matrix)).astype(bool)] = np.nan
alternative_survival_value_matrix = Adjacency(alternative_survival_value_matrix, matrix_type='similarity')
alternative_survival_value_matrix.data = scale(alternative_survival_value_matrix.data)

# Add an intercept matrix and combine
intercept_matrix = Adjacency(np.ones_like(threat_matrix.data))
intercept_matrix.data = scale(intercept_matrix.data)

theoretical_matrices = Adjacency([intercept_matrix, session_matrix, run_matrix, competitor_difference_matrix, current_competitors_matrix, alternative_competitors_matrix, 
                                  survival_value_diff_matrix, current_survival_value_matrix, alternative_survival_value_matrix, 
                                  threat_matrix])

# Isolate trials where subjects chose the "now" option
beta_data = first_level_betas_nii.get_fdata()
beta_data = beta_data[..., now_or_later]
first_level_betas_nii_now = nb.Nifti1Image(beta_data, first_level_betas_nii.affine)

# Run the searchlight
# This is run using custom searchlight RSA code (in searchlight_functions.py)
print(datetime.datetime.now())
searchlight = SearchLightRSA(
    brain_mask, 
    process_mask_img=brain_mask,  # For quick testing, uncomment code above and use process_mask_img instead of brain_mask
    metric='spearman', 
    radius=6, n_jobs=len(os.sched_getaffinity(0)),
    verbose=0)

searchlight.fit(first_level_betas_nii_now, theoretical_matrices, y_mask=unique_run_matrix) 
print(datetime.datetime.now())

# Get output
searchlight_img = new_img_like(first_level_betas_nii, searchlight.scores_)

# Save
betas_dir = os.path.join(base_dir, 'rsa_revised', 'searchlight_betas')
if not os.path.exists(betas_dir):
    os.makedirs(betas_dir)

nb.save(searchlight_img, os.path.join(betas_dir, 'sub-{0}_searchlight_betas_FSL_revised.nii.gz'.format(subject)))