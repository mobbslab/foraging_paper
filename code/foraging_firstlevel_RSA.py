import os
import pandas as pd
from sklearn.preprocessing import scale
import nibabel as nb
from nilearn.image import smooth_img
from  nipype.interfaces import fsl
import nipype.algorithms.modelgen as model 
from nipype.caching import Memory
from nipype.interfaces.base import Bunch
import itertools
import numpy as np

base_dir = '../../data/derivatives/'
sampling_freq = 30
sessions = ['d1', 'd2']
fwhm = 3
runs = [4, 4]
glm = 'rsa'

# Get slurm run ID
try:
    runID = int(os.environ['SLURM_ARRAY_TASK_ID'])
except:
    runID = 1 # TESTING

# Select subject, session and run to be processed
subs = pd.read_csv('../subject_list.txt', header=None)
sub_sess_run = list(itertools.product(subs[0], ['d1', 'd2'], range(4)))
subject, this_session, this_run = sub_sess_run[runID-1]

n_runs = np.sum(runs)

temp_dir = '../fsl_temp/temp_sub-{0}_sess-{1}_run-{2}'.format(subject, this_session, this_run)

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
    
os.chdir(temp_dir)
    
mem = Memory(base_dir='.')

# LOAD DATA

confounds = {}

for n, session in enumerate(sessions):
    confounds[session] = {}
    for run in range(runs[n]):
        conf = pd.read_csv(os.path.join(base_dir, 'fmriprep/sub-{0}/ses-{1}/func/sub-{0}_ses-{1}_task-mos_run-0{2}_desc-confounds_regressors.tsv'.format(subject, session, run+1)), sep='\t')
        confounds[session][run] = conf[['dvars', 'framewise_displacement', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter']]

events = {}

trial_types = []

for n, session in enumerate(sessions):
    events[session] = {}
    for run in range(runs[n]):
        if glm == 'univariate':
            ev = pd.read_csv(os.path.join(base_dir, 
                                          'fmriprep/sub-{0}/ses-{1}/func/subject-{0}_ses-{1}_task-mos_run-0{2}_events.tsv'.format(subject, 
                                                                                                                         session, 
                                                                                                                         run+1)))[['trial_type', 'onset', 'duration']]
        elif glm == 'rsa':
            ev = pd.read_csv(os.path.join(base_dir, 
                                          'fmriprep/sub-{0}/ses-{1}/func/subject-{0}_ses-{1}_task-mos_run-0{2}_events.tsv'.format(subject, session, 
                                                                                                                         run+1)))[['trial_type', 'onset', 'duration', 'competitor_difference']]
            ev.loc[ev['trial_type'].isin(['blocked_with_decision_safe', 'blocked_with_decision_threat']), 'trial_type'] = ev.loc[ev['trial_type'].isin(['blocked_with_decision_safe', 'blocked_with_decision_threat']), 'trial_type'] + '_' + \
                                                                                              ev.loc[ev['trial_type'].isin(['blocked_with_decision_safe', 'blocked_with_decision_threat']), 'competitor_difference'].astype(str)
            # Estimate first and second day effects separately
#             ev['trial_type'] = ev['trial_type'] + '_' + session + '_run' + str(run)
            
            ev = ev[['trial_type', 'onset', 'duration']]
            trial_types += ev['trial_type'].unique().tolist()
        ev.columns = ['Stim', 'Onset', 'Duration']
        ev[['Onset', 'Duration']] /= 1000
        ev['id'] = range(len(ev))
        
        # CORRECT DRIFT
        ev['trial_number'] = 0
        
        for i in ev[ev['Stim'] == 'decision_end'].index:
            if not i == len(ev) - 1:
                if 'antic' in ev.loc[i+1]['Stim']:
                    ev['trial_number'][i+2:] += 1
                else:
                    ev['trial_number'][i+1:] += 1
        
        # CORRECT DRIFT CAUSED BY TASK CODE DROPPING SAMPLES
        ev['Onset'] += ev['trial_number'] * .26
        ev['Onset'] -= .13
        
        ev = ev.iloc[:, :4]
        
        events[session][run] = ev
        
# Work out where each condition appears
unique_conditions = [i for i in list(set(trial_types)) if 'blocked' in i and 'decision' in i]

condition_locations = {}

for cond in unique_conditions:
    for session in sessions:
        for run in range(runs[n]):
            if cond in events[session][run]['Stim'].unique().tolist():
                if not cond in condition_locations:
                    condition_locations[cond] = []
                condition_locations[cond].append((session, run))

# Get condition counts
conditions = {'Session': [], 'Run': [], 'Condition': [], 'Trial': []}

for session in sessions:
    for run in range(runs[n]):
        run_conds = events[session][run]['Stim']
        run_conds = run_conds[run_conds.str.contains('.0')].tolist()
        conditions['Session'] += [session] * len(run_conds)
        conditions['Run'] += [run] * len(run_conds)
        conditions['Condition'] += run_conds
        conditions['Trial'] += list(range(len(run_conds)))
        
all_conds = list(set(conditions['Condition']))
all_conds = sorted(all_conds)

cond_labels = dict([(i, n) for n, i in enumerate(all_conds)])
conditions['Condition_label'] = [cond_labels[i] for i in conditions['Condition']]
conditions_df = pd.DataFrame(conditions)

# Count number of each condition
c, count = np.unique(conditions_df.Condition, return_counts=True)
condition_counts = dict(zip(c, [str(i) for i in count]))

# Give each trial an ID (for linking to behaviour)
for cond in unique_conditions:
    n_runs_present = len(condition_locations[cond])
    if n_runs_present < 2 and not '4' in cond:
        raise ValueError("Condition {0} present in less than 2 separate runs".format(cond))

    for n, session in enumerate(sessions):
        for run in range(runs[n]):
            events[session][run].loc[events[session][run]['Stim'] == cond, 'Stim'] = \
            events[session][run].loc[events[session][run]['Stim'] == cond, 'Stim'] + '_Ses-{0}_Run-{1}_id-'.format(session, run) + events[session][run].loc[events[session][run]['Stim'] == cond, 'id'].astype(str)
            
for n, session in enumerate(sessions):
    for run in range(runs[n]):
        events[session][run] = events[session][run][[c for c in events[session][run].columns if not c == 'id']]

# Create session info in nipype format
session_info = {}

for n, session in enumerate(sessions):
    session_info[session] = {}
    for run in range(runs[n]):

        trialinfo = events[session][run]
        confoundinfo = confounds[session][run]
        
        # Scale confounds
        confoundinfo[confoundinfo.columns] = scale(confoundinfo[confoundinfo.columns])

        # For some reason the final decision_end period is NaN
        if np.any(trialinfo.isnull()):
            nan_idx = np.where(trialinfo.isnull())[0][0]
            if nan_idx == len(trialinfo) - 1: # Final event
                trialinfo.loc[nan_idx, 'Duration'] = 7  # Ends after 7 seconds
            else:
                trialinfo.loc[nan_idx, 'Duration'] = trialinfo.loc[nan_idx:nan_idx + 1, 'Onset'].diff().values[1]
        
        # Check data
        assert np.any(~trialinfo.isnull()), 'NaNs present in trial info'
        assert np.all(trialinfo[['Onset', 'Duration']] > 0), 'Something is wrong - onsets or durations below zero'
        
        conditions = []
        onsets = []
        durations = []

        for group in trialinfo.groupby('Stim'):
            conditions.append(group[0])
            onsets.append(list(group[1].Onset))
            durations.append(group[1].Duration.tolist())

        subject_info = Bunch(conditions=conditions,
                                onsets=onsets,
                                durations=durations,
                             regressors=[list(confoundinfo.framewise_displacement.fillna(0)),
                                         list(confoundinfo.trans_x),
                                         list(confoundinfo.trans_y),
                                         list(confoundinfo.trans_z),
                                         list(confoundinfo.rot_x),
                                         list(confoundinfo.rot_y),
                                         list(confoundinfo.rot_z),
                                         list(confoundinfo.csf),
                                         list(confoundinfo.white_matter),
                                         list(np.ones_like(confoundinfo.white_matter)) # INTERCEPT
                                        ])

        session_info[session][run] = subject_info

# Get imaging data
base_dir = '/central/groups/mobbslab/toby/foraging/data/derivatives/'

fMRI = []

fMRI = {}

for n, session in enumerate(sessions):
    fMRI[session] = {}
    for run in range(runs[n]):
        fMRI[session][run] = os.path.join(base_dir, 'fmriprep/sub-{0}/ses-{1}/func/sub-{0}_ses-{1}_task-mos_run-0{2}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(subject, session, run+1))

skip = mem.cache(fsl.ExtractROI)
skip_results = skip(in_file=fMRI[this_session][this_run], t_min=0, t_size=-1)

# Set up model
modelspec = model.SpecifyModel(
                            input_units='secs',
                            time_repetition=1,
                            high_pass_filter_cutoff=128,
                            subject_info=session_info[this_session][this_run],
                            functional_runs=skip_results.outputs.roi_file
)

specify_model_results  = modelspec.run()

level1design = fsl.model.Level1Design(interscan_interval=1,
                                    bases = {'dgamma':{'derivs': False}},
                                    session_info = specify_model_results.outputs.session_info,
                                    model_serial_correlations=True)

level1design_results = level1design.run()

# Get first level model
modelgen = mem.cache(fsl.model.FEATModel)
modelgen_results = modelgen(fsf_file=level1design_results.outputs.fsf_files,
                            ev_files=level1design_results.outputs.ev_files)
modelgen_results.outputs

# Get mask
mask = mem.cache(fsl.maths.ApplyMask)
mask_results = mask(in_file=skip_results.outputs.roi_file,
                    mask_file=os.path.join(base_dir, 'fmriprep/sub-{0}/ses-{1}/func/sub-{0}_ses-{1}_task-mos_run-0{2}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(subject, session, run+1)))
mask_results.outputs

# Normalise
img = nb.load(mask_results.outputs.out_file)
img_data = img.get_fdata()
img_data = img_data - img_data.mean(axis=3)[..., np.newaxis]
mean_centered = nb.Nifti1Image(img_data, img.affine)
out_file = mask_results.outputs.out_file.replace('.nii.gz', '_centered.nii.gz')
nb.save(mean_centered, out_file)

# Estimate model
# NOTE - this was run using FSL. For univariate analyses I used a faster method implemented in pure Python.
# See the univariate script for details.
filmgls= mem.cache(fsl.FILMGLS)
filmgls_results = filmgls(in_file=out_file,
                          design_file = modelgen_results.outputs.design_file,
                          tcon_file = modelgen_results.outputs.con_file,
                          fcon_file = modelgen_results.outputs.fcon_file,
                          autocorr_noestimate = False)

# Smooth and rename the data
def smooth_rename(nii, fwhm, out_dir, condition_label=''):
    img = nb.load(nii)
    img = smooth_img(img, fwhm)
    out_fname = '{0}_param_estimates_smoothed.nii.gz'.format(condition_label)
    print("Saving {0} as {1}".format(nii, out_fname))
    nb.save(img, os.path.join(out_dir, out_fname))

# Save outputs
smoothed_dir = os.path.join(base_dir, 'first_level/rsa/fsl_betas_smoothed', 'sub-{0}'.format(subject))

if not os.path.exists(smoothed_dir):
    os.makedirs(smoothed_dir)

for n, cond in enumerate(session_info[this_session][this_run].conditions):
    smooth_rename(filmgls_results.outputs.param_estimates[n], fwhm, smoothed_dir, cond)