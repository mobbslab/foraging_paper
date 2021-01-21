import os
import pandas as pd
import numpy as np
import nibabel as nb
from lyman import glm
from lyman.utils import matrix_to_image
import time
import json
import warnings

def load_json(fname):
    with open(fname, 'r') as f:
        out = json.load(f)
    return out

if __name__ == "__main__":
    
    # Get slurm run ID
    # runID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    runID = 1

    # Select subject, session and run to be processed
    subs = pd.read_csv('../subject_list.txt', header=None)[0].tolist()
    print(subs)

    base_dir = os.path.abspath('../data/derivatives/')

    second_level_contrast_files = {}

    for n, subject in enumerate(subs):
        
        print("RUNNING SUBJECT {0} | {1} / {2}".format(subject, n+1, len(subs)))

        # Important variabkles
        output_dir = os.path.join(base_dir, 'univariate', 'second_level', 
                                'sub-{0}'.format(subject))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("OUTPUT DIRECTORY = {0}".format(output_dir))

        contrast_files = []
        variance_files = []
        name_files = []
        contrast_spec_files = []

        sessions = ['d1', 'd2']
        runs = range(4)

        print("LOADING FIRST LEVEL DATA")
        for sess in sessions:
            for run in runs:
                first_level_output_dir = os.path.join(base_dir, 'univariate', 'first_level', 
                                        'sub-{0}'.format(subject), 'sess-{0}'.format(sess), 'run-{0}'.format(run))
                contrast_files.append(os.path.join(first_level_output_dir, 'contrast.nii.gz'))
                variance_files.append(os.path.join(first_level_output_dir, 'variance.nii.gz'))
                contrast_spec_files.append(os.path.join(first_level_output_dir, 'contrasts.json'))
                name_files.append(os.path.join(first_level_output_dir, 'sub-{0}_sess-{1}_run-{2}_contrast.txt'.format(subject, sess, run)))

        
        # Load the parameter and variance data for each run/contrast.
        con_images = [nb.load(f) for f in contrast_files]
        var_images = [nb.load(f) for f in variance_files]
        name_lists = [np.loadtxt(f, str).tolist() for f in name_files]

        # Get contrast specs
        contrasts = [load_json(f) for f in contrast_spec_files]

        # Figure out the idx of each contrast in the nifti images
        # Some contrasts are missing from certain runs - e.g. sometimes there's no shock
        contrast_names = []
        run_contrast_idx = []
        for run in contrasts:
            contrast_idx = {}
            for n, contrast in enumerate(run):
                name, _, _ = contrast
                contrast_names.append(name)
                contrast_idx[name] = n
            run_contrast_idx.append(contrast_idx)

        contrast_names = list(set(contrast_names))


        # Loop over contrasts
        for i, name in enumerate(contrast_names):

            print("ESTIMATING CONTRAST {0}, {1}".format(i+1, name))

            con_frames = []
            var_frames = []

            # Find the contrast image
            for run in range(len(contrasts)):
                contrast_idx = run_contrast_idx[run]
                if not name in contrast_idx:
                    warnings.warn('Contrast name {0} not present in run {1}'.format(name, run))
                else:
                    idx = contrast_idx[name]
                    con_frames.append(con_images[run].get_fdata()[..., idx])
                    var_frames.append(var_images[run].get_fdata()[..., idx])

            con_data = np.stack(con_frames, axis=-1)
            var_data = np.stack(var_frames, axis=-1)

            # Define a mask as voxels with nonzero variance in each run
            # and extract voxel data as arrays
            mask = (var_data > 0).all(axis=-1)
            mask_img = nb.Nifti1Image(mask.astype(np.int8), con_images[0].affine, con_images[0].header)
            con = con_data[mask]
            var = var_data[mask]

            # Compute the higher-level fixed effects parameters
            con_ffx, var_ffx, t_ffx = glm.contrast_fixed_effects(con, var)

            # Convert to image volume format
            con_img = matrix_to_image(con_ffx.T, mask_img)
            var_img = matrix_to_image(var_ffx.T, mask_img)
            t_img = matrix_to_image(t_ffx.T, mask_img)

            print(con_img.shape)

            # Write out output images
            if not os.path.exists(os.path.join(output_dir, name)):
                os.makedirs(os.path.join(output_dir, name))
            con_img.to_filename(os.path.join(output_dir, name, "second_level_contrast.nii.gz"))
            var_img.to_filename(os.path.join(output_dir, name, "second_level_variance.nii.gz"))
            t_img.to_filename(os.path.join(output_dir, name, "second_level_tstat.nii.gz"))
            mask_img.to_filename(os.path.join(output_dir, name, "second_level_mask.nii.gz"))

            if not name in second_level_contrast_files:
                second_level_contrast_files[name] = []
            second_level_contrast_files[name].append(os.path.join(output_dir, name, "second_level_contrast.nii.gz"))

    # Combine to 4D image
    output_dir = os.path.join(base_dir, 'univariate', 'second_level', '4D')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name, images in second_level_contrast_files.items():
        img_4d = nb.concat_images(images)
        out_fname = os.path.join(output_dir, '{0}_4D.nii.gz'.format(name))
        img_4d.to_filename(out_fname)


    print("DONE")

