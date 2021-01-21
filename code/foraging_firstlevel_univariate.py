# Limit the number of threads used by numpy
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1" 

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import nibabel as nb
import itertools
from lyman import glm, signals
from lyman.utils import matrix_to_image, image_to_matrix
import time
import time
from nipype.interfaces.base import Bunch
import json
from joblib import dump, load
import uuid
import shutil


if __name__ == "__main__":
    
    testing = False
    n_test = 50   

    # Get slurm run ID
    try:
        runID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    except:
        runID = 72

    # Select subject, session and run to be processed
    subs = pd.read_csv('../subject_list.txt', header=None)
    sub_sess_run = list(itertools.product(subs[0], ['d1', 'd2'], range(4)))
    subject, this_session, this_run = sub_sess_run[runID-1]
    
    print("RUNNING SUBJECT {0}, SESSION {1}, RUN {2}".format(subject, this_session, this_run))
    print("Parallel processing with {0} cores".format(len(os.sched_getaffinity(0))))
    

    # Important variabkles
    base_dir = os.path.abspath('../data/derivatives/')
    output_dir = os.path.join(base_dir, 'univariate', 'first_level', 
                             'sub-{0}'.format(subject), 'sess-{0}'.format(this_session), 'run-{0}'.format(this_run))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("OUTPUT DIRECTORY = {0}".format(output_dir))
    
    # Load confounds
    conf = pd.read_csv(os.path.join(base_dir, 
                                    'fmriprep/sub-{0}/ses-{1}/func/sub-{0}_ses-{1}_'
                                    'task-mos_run-0{2}_desc-confounds_regressors.tsv'.format(subject, this_session, this_run+1)), sep='\t')
    confounds= conf[['dvars', 'framewise_displacement', 'trans_x', 'trans_y', 
                     'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter']]
            
            
    # Get events
    trial_types = []

    ev = pd.read_csv(os.path.join(base_dir, 
                                'fmriprep/sub-{0}/ses-{1}/func/sub-{0}_ses-{1}_'
                                'task-mos_run-0{2}_events.tsv'.format(subject, this_session, this_run+1)))[['trial_type', 
                                                                                                            'onset', 
                                                                                                            'duration', 
                                                                                                            'competitor_difference']]
    ev.loc[ev['trial_type'].isin(['blocked_with_decision_safe', 
                                  'blocked_with_decision_threat']), 
                                  'trial_type'] = ev.loc[ev['trial_type'].isin(['blocked_with_decision_safe', 
                                                                                'blocked_with_decision_threat']), 
                                                                                'trial_type'] + '_' + \
                                                                                 ev.loc[ev['trial_type'].isin(['blocked_with_decision_safe', 
                                                                                                                'blocked_with_decision_threat']), 
                                                                                                                'competitor_difference'].astype(str)
    ev = ev[['trial_type', 'onset', 'duration']]
    trial_types += ev['trial_type'].unique().tolist()
    ev.columns = ['Stim', 'Onset', 'Duration']
    ev[['Onset', 'Duration']] /= 1000
    ev['id'] = range(len(ev))
    
    # CORRECT TRIAL NUMBER DRIFT
    ev['trial_number'] = 0
    
    for i in ev[ev['Stim'] == 'decision_end'].index:
        if not i == len(ev) - 1:
            if 'antic' in ev.loc[i+1]['Stim']:
                ev['trial_number'][i+2:] += 1
            else:
                ev['trial_number'][i+1:] += 1
    
    # CORRECT DRIFT CAUSED BY TASK CODE DROPPING SAMPLES
    # Task code occasionally dropped samples which means the behavioural timings and the 
    # imaging go out of sync over the course of the task. Thankfully the amount of drift
    # is consistent on every trial so we can just use this to correct the onsets.
    ev['Onset'] += ev['trial_number'] * .26
    ev['Onset'] -= .13
    
    ev = ev.iloc[:, :4]
    events = ev
    
    # Assign unique IDs to each condition label to line up with behavioural data
    unique_conditions = [i for i in list(set(trial_types)) if 'blocked' in i and 'decision' in i]

    for cond in unique_conditions:
        events.loc[events['Stim'] == cond, 'Stim'] = \
        events.loc[events['Stim'] == cond, 'Stim'] + \
            '_Ses-{0}_Run-{1}_id-'.format(this_session, this_run) + events.loc[events['Stim'] == cond, 'id'].astype(str)
        
    # Get behavioural modelling data
    decision_data = pd.read_csv('../data/decision_data_REVISED.csv')
    decision_data = decision_data[decision_data['subject'] == subject]
    decision_data['decision_current'] = decision_data['left_or_right'].shift(1) # Shift decision to next trial so we know which patch they're currently "in"
    decision_data.loc[decision_data['trial_index'] == 0, 'decision_current'] = np.nan # No current patch for first trial

    # GET SESSION INFO
    # This is compiled to a format nipype can use, although we end up not using nipype
    trialinfo = events
    confoundinfo = confounds
    
    # Get decision variables
    run_decision_data = decision_data[(decision_data['day'] == int(this_session[1:])) & (decision_data['block'] == this_run + 1)]
    
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
    assert np.all(trialinfo[['Onset', 'Duration']] >= 0), 'Something is wrong - onsets or durations below zero'
    
    conditions = []
    onsets = []
    durations = []

    for group in trialinfo.groupby('Stim'):
        conditions.append(group[0])
        onsets.append(list(group[1].Onset))
        durations.append(group[1].Duration.tolist())
        
    # Collapse conditions and onsets for univariate analysis
    decision_ids = [int(re.search('(?<=id-)\d+', i).group()) for i in conditions if 'blocked_with' in i]

    new_conditions = list(set([re.sub('_(safe)?(threat)?_-?\d\.0_Ses-[A-Za-z0-9-_]+', '', i) for i in conditions]))

    new_conditions.append('Competitors_diff_pmod')
    new_conditions.append('Competitors_alternative_pmod')
    new_conditions.append('Competitors_current_pmod')

    new_conditions.append('SV_diff_pmod')
    new_conditions.append('SV_alternative_pmod')
    new_conditions.append('SV_current_pmod')
    
    new_conditions.append('threat')

    new_onsets = dict(zip(new_conditions, [[] for i in range(len(new_conditions))]))
    new_durations = dict(zip(new_conditions, [[] for i in range(len(new_conditions))]))
    amplitudes = dict(zip(new_conditions, [[] for i in range(len(new_conditions))]))


    for n, cond in enumerate(conditions):
        new_cond = re.sub('_(safe)?(threat)?_-?\d\.0_Ses-[A-Za-z0-9-_]+', '', cond)
        new_onsets[new_cond] += onsets[n]
        new_durations[new_cond] += durations[n]
        amplitudes[new_cond] += [1 for i in onsets[n]]

        if 'blocked_with_decision' in new_cond:

            # Get decision variables
            decision_id = int(re.search('(?<=id-)\d+', cond).group())
            trial_decision_data = run_decision_data[run_decision_data['id'] == decision_id]

            if not trial_decision_data['decision_current'].isnull().any():
                current_decision = int(trial_decision_data['decision_current'].values[0])

                current_comp = trial_decision_data[['left_conspecifics_number', 'right_conspecifics_number']].values[0][current_decision]
                alternative_comp = trial_decision_data[['left_conspecifics_number', 'right_conspecifics_number']].values[0][1-current_decision]
                comp_diff = current_comp - alternative_comp

                current_sv = trial_decision_data[['survival_value_left_Z', 'survival_value_right_Z']].values[0][current_decision]
                alternative_sv = trial_decision_data[['survival_value_left_Z', 'survival_value_right_Z']].values[0][1-current_decision]
                sv_diff = current_sv - alternative_sv

                amplitudes['Competitors_diff_pmod'].append(comp_diff)
                amplitudes['Competitors_alternative_pmod'].append(alternative_comp)
                amplitudes['Competitors_current_pmod'].append(current_comp)

                amplitudes['SV_diff_pmod'].append(sv_diff)
                amplitudes['SV_alternative_pmod'].append(alternative_sv)
                amplitudes['SV_current_pmod'].append(current_sv)

                amplitudes['threat'].append(int('threat' in cond))
                
                amplitudes[new_cond].append(1)

                # Add onsets
                for c in ['Competitors_diff_pmod', 'Competitors_alternative_pmod', 'Competitors_current_pmod', 
                        'SV_diff_pmod', 'SV_alternative_pmod', 'SV_current_pmod', 'threat']:
                    new_onsets[c] += onsets[n]
                    new_durations[c] += durations[n]
    
    # Make sure conditions are in the same order to facilitate lining up across runs
    new_conditions = sorted(new_conditions)

    session_info = Bunch(conditions=new_conditions,
                            onsets=[new_onsets[k] for k in new_conditions],
                            durations=[new_durations[k] for k in new_conditions],
                            amplitudes=[list(scale(amplitudes[k]) + 1) for k in new_conditions], # Mean center, mean of 1 (Mumford et al., 2015)
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

    # print(session_info.conditions)        

    # Load fMRI data
    fMRI_data = os.path.join(base_dir, 
                            'fmriprep/sub-{0}/ses-{1}/func/sub-{0}_ses-{1}_task-mos_run-0{2}_'
                            'space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(subject, this_session, this_run+1))

    # Useful function to convert nipype-style session info to a datafrae
    def session_info_to_df(session_info):
        df_info = {'condition': [], 'onset': [], 'duration': [], 'value': []}
        for n, condition_onsets in enumerate(session_info.onsets):
            cond_name = session_info.conditions[n]
            for trial, onset in enumerate(condition_onsets):
                df_info['condition'].append(cond_name)
                df_info['onset'].append(onset)
                df_info['duration'].append(session_info.durations[n][trial])
                df_info['value'].append(session_info.amplitudes[n][trial])
                
        return pd.DataFrame(df_info)
    
    # Load the data
    print("LOADING DATA")
    ts_img = nb.load(fMRI_data)
    mask_img = nb.load(os.path.join(base_dir, 
                                    'fmriprep/sub-{0}/ses-{1}/func/sub-{0}_ses-{1}_'
                                    'task-mos_run-0{2}_space-MNI152NLin2009cAsym_'
                                    'desc-brain_mask.nii.gz'.format(subject, this_session, this_run+1)))

    #### !!!!TESTING!!! #####
    if testing:
        ts_img_data = ts_img.get_fdata()
        ts_img_short = nb.Nifti1Image(ts_img_data[..., :n_test], ts_img.affine, ts_img.header)
        ts_img = ts_img_short
        print(ts_img_short.shape)
    # raise TypeError()
    ##########################
    
    """
    *First-level modelling*
    
    This part uses functions from Lyman (http://www.cns.nyu.edu/~mwaskom/software/lyman/index.html), which
    basically implements FSL's GLM fitting methods in Python and makes them much faster. We have a lot of
    data here so this speedup helps.

    This uses a fork of Lyman that implements parallelisation and optimisation using Numba to make it even faster. 
    The end result is that a model that takes ~10 hours to fit in FSL takes about 45 minutes.

    A lot of this code is modified from this script: https://github.com/mwaskom/lyman/blob/master/lyman/workflows/model.py 

    """

    # Smooth the data
    print("SMOOTHING")
    ts_img_affine = ts_img.affine
    ts_img_header = ts_img.header
    ts_img_dtype = ts_img.get_data_dtype()
    n_tp = ts_img.shape[-1]

    ts_img = signals.smooth_volume(ts_img, 8, mask_img=mask_img, inplace=False)

    data = ts_img.get_fdata()
    mean = data.mean(axis=-1)
    mean_img = nb.Nifti1Image(mean, ts_img_affine, ts_img_header)

    del ts_img

    # Calculate mask
    mask = mask_img.get_fdata()
    mask = (mask > 0)
    mask_img = nb.Nifti1Image(mask.astype(np.uint8), mask_img.affine, mask_img.header)
    n_vox = mask.sum()

    # Set up GLM
    print("SETTING UP GLM")
    
    # Temporally filter the data
    hpf_matrix = glm.highpass_filter_matrix(n_tp, 128, 1)
    data[mask] = np.dot(hpf_matrix, data[mask].T).T
    data[mask] -= data[mask].mean(axis=-1, keepdims=True)
    
    # Temporally filter the nuisance regressors
    for n, i in enumerate(session_info.regressors):
        ### !!!TESTING!! ###########
        if testing:
            session_info.regressors[n] = hpf_matrix.dot(i[:n_test])    
        ############################
        else:
            session_info.regressors[n] = hpf_matrix.dot(i)                           
                                            
    # --- Design matrix construction

    # Build the regressor sub-matrix
    tps = np.arange(0, n_tp * 1, 1)
    confound_cols = [f"confound_{i+1}" for i in range(len(session_info.regressors))]
                                            
    regressors = pd.DataFrame(np.array(session_info.regressors).T, tps, confound_cols)

    # Build the full design matrix
    design = session_info_to_df(session_info)
    
    # #### !!!!TESTING!!! #####
    # if testing:
    #     design = design[design['onset'] < n_test - 5]
    # ##########################

    hrf_model = glm.GammaBasis(time_derivative=False,
                            disp_derivative=False)  
    X = glm.build_design_matrix(design, hrf_model,
                                regressors=regressors,
                                n_tp=n_tp, tr=1,
                                hpf_matrix=hpf_matrix)
    
    # Save the design matrix
    model_file = os.path.join(output_dir, 'sub-{0}_sess-{1}_run-{2}_design_matrix.csv'.format(subject, this_session, this_run))
    X.to_csv(model_file, index=False)

    # --- Model estimation
    data[~mask] = 0  
    
    # Prewhiten the data
    print("PREWHITENING")
    ts_img = nb.Nifti1Image(data, ts_img_affine)
    del data
    WY, WX = glm.prewhiten_image_data(ts_img, mask_img, X.values)

    # np.save('WX', WX)
    # np.save('WY', WY)

    # WX = np.load('WX.npy')
    # WY = np.load('WY.npy')

    # Reshape - seems faster with voxels first
    WX = WX.transpose((2, 0, 1))
    WY = WY.T
    # UID for temp storage
    uid = uuid.uuid4().hex
    folder = '/central/scratchio/tobywise/temp_' + str(uid)
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass

    # Memmap the data so it's not held in memory
    print('Temp directory = {0}'.format(folder))
    WX_filename_memmap = os.path.join(folder, 'X_memmap')
    WY_filename_memmap = os.path.join(folder, 'Y_memmap')
    dump(WX, WX_filename_memmap)
    dump(WY, WY_filename_memmap)
    WX_memmap = load(WX_filename_memmap, mmap_mode='r')
    WY_memmap = load(WY_filename_memmap, mmap_mode='r')

    # Clean things up
    del ts_img
    del WX
    del WY
    
    # FIT THE GLM
    # This uses modified fitting code from Lyman to enable parallel processing
    # Parallel processing is not straightforward here because there is so much data
    # Joblib's Loky backend (which is the most straightforward to use) forks processes by default
    # meaning memory usage is basically copied for every process. The data for each run takes
    # up around 40GB of memory (in addition to the fMRI data, the design matrix has a separate)
    # set of regressors for each voxel, making it n_tp x n_vox x n_regressors. This means that 
    # each process seems to use over 40GB of memory, making things difficult.

    # The solution to this is to memmap the arrays (as done above), so that they are not held in memory,
    # and each process just reads in the bit it needs. However, if we literally do this on each iteration
    # of the fitting procedure (i.e. for each voxel), we end up being limited by I/O speed. Therefore,
    # the way this works is to memmap the data, then within each process read in all the data necessary
    # for the chunk being processed, then write that back to the memmapped array.
    print(h.heap())
    start = time.time()
    B, SS, XtXinv, E = glm.iterative_ols_fit(WY_memmap, WX_memmap, n_jobs=len(os.sched_getaffinity(0)))
    end = time.time()
    print('GLM fitting took {0} seconds'.format(end - start))
    
    try:
        shutil.rmtree(folder)
    except:  # noqa
        print('Could not clean-up automatically.')

    # # Convert outputs to image format
    beta_img = matrix_to_image(B.T, mask_img)
    error_img = matrix_to_image(SS, mask_img)
    XtXinv_flat = XtXinv.reshape(n_vox, -1)
    ols_img = matrix_to_image(XtXinv_flat.T, mask_img)
    resid_img = matrix_to_image(E, mask_img)

    # Save everything
    print("SAVING")
    nb.save(beta_img, os.path.join(output_dir, 'beta.nii.gz'))
    nb.save(error_img, os.path.join(output_dir, 'error.nii.gz'))
    nb.save(ols_img, os.path.join(output_dir, 'ols.nii.gz'))
    nb.save(mask_img, os.path.join(output_dir, 'mask.nii.gz'))
    nb.save(resid_img, os.path.join(output_dir, 'resid.nii.gz'))

    np.save(os.path.join(output_dir, 'B'), B)
    np.save(os.path.join(output_dir, 'SS'), SS)
    np.save(os.path.join(output_dir, 'XtXinv'), XtXinv)
    np.save(os.path.join(output_dir, 'E'), E)

    print("ESTIMATING CONTRASTS")
    # Contrast estimates
    param_names = X.columns
    non_confound_names = [i for i in param_names if not 'confound' in i]

    # Reshape the matrix form data to what the glm functions expect
    # B = B.T
    n_vox, n_ev = B.shape
    # XtXinv = XtXinv.reshape(n_ev, n_ev, n_vox).T

    # Define contrasts
    contrasts = []

    for cond in non_confound_names:
        contrasts.append([cond, [cond], [1]])
        
    contrasts.append(['Competitors_current-alternative', ['Competitors_current_pmod', 'Competitors_alternative_pmod'], [1, -1]])
    contrasts.append(['SV_current-alternative', ['SV_current_pmod', 'SV_alternative_pmod'], [1, -1]])

    for con in contrasts:
        if len(con[-1]) > 1:
            assert np.sum(con[-1]) == 0, 'Contrasts {0} do not sum to zero'.format(con[-1])

    # Save contrast specification to json
    with open(os.path.join(output_dir, 'contrasts.json'), 'w') as f:
        json.dump(contrasts, f)

    # Obtain list of contrast matrices
    C = []
    names = []
    for contrast_spec in contrasts:
        name, params, _ = contrast_spec
        if set(params) <= set(param_names):
            C.append(glm.contrast_matrix(contrast_spec, X))
            names.append(name)

    # Estimate the contrasts, variances, and statistics in each voxel
    G, V, T = glm.iterative_contrast_estimation(B, SS, XtXinv, C)
    contrast_img = matrix_to_image(G.T, mask_img)
    variance_img = matrix_to_image(V.T, mask_img)
    tstat_img = matrix_to_image(T.T, mask_img)

    print("SAVING CONTRAST ESTIMATES")
    # Write out the output files
    nb.save(contrast_img, os.path.join(output_dir, 'contrast.nii.gz'))
    nb.save(variance_img, os.path.join(output_dir, 'variance.nii.gz'))
    nb.save(tstat_img, os.path.join(output_dir, 'tstat.nii.gz'))

    name_file = os.path.join(output_dir, 'sub-{0}_sess-{1}_run-{2}_contrast.txt'.format(subject, this_session, this_run))
    np.savetxt(name_file, names, "%s")

    print("DONE")

