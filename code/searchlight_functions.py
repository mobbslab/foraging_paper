##########################################
# FUNCTIONS FOR SEARCHLIGHT RSA ANALYSES #
##########################################

# This code implements RSA within a moveable searchlight by adapting the nilearn searchlight class.
# This is extensively optimised using Numba and certain elements can be run in parallel using joblib.
# This implementation is NOT designed to be flexible however, for example it only implements Spearman 
# correlation as a measure of similarity.

from numba import njit
import numpy as np
from nilearn._utils.niimg_conversions import check_niimg_4d, check_niimg_3d
from sklearn import neighbors
from nilearn.image.resampling import coord_transform
import joblib
from nilearn import image
import warnings
from sklearn.externals.joblib import Parallel, delayed, cpu_count
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from nilearn import masking
from nilearn.image.resampling import coord_transform
from nilearn._utils import check_niimg_4d

np.seterr(divide='ignore', invalid='ignore')


@njit
def get_tri(arr):
    tri_idx = np.triu_indices_from(arr, k=1)
    out = np.zeros(len(tri_idx[0]), arr.dtype) 
    for n, (i,j) in enumerate(zip(tri_idx[0], tri_idx[1])): 
        out[n] = arr[i,j] 
    return out

@njit
def scale_data(X):
    return (X - np.nanmean(X)) / (np.nanstd(X) + 1e-20)

@njit
def ols(x, y, y_mask):
    for i in range(y.shape[1]):
        y_mask[np.isnan(y[:, i])] = True
    y_mask[np.isnan(x[:, 0])] = True
    x = x[~y_mask]
    y = y[~y_mask]
        
    coefs = np.dot(np.linalg.pinv(np.dot(x.T,x)),np.dot(x.T,y))

    return coefs

@njit
def rankdata(a):

    arr = np.ravel(np.asarray(a))
    sorter = np.argsort(arr, kind='quicksort')

    inv = np.empty(sorter.size, dtype=np.int16)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    arr = arr[sorter]
    obs = np.hstack((np.array([True]), arr[1:] != arr[:-1]))
    dense = obs.cumsum()[inv]

    # cumulative counts of each unique value
    count = np.hstack((np.nonzero(obs)[0], np.array([len(obs)])))

    # average method
    return .5 * (count[dense] + count[dense - 1] + 1)

@njit
def pearson_corr(data1, data2):
    
    M = data1.size

    sum1 = 0.
    sum2 = 0.
    for i in range(M):
        sum1 += data1[i]
        sum2 += data2[i]
    mean1 = sum1 / M
    mean2 = sum2 / M

    var_sum1 = 0.
    var_sum2 = 0.
    cross_sum = 0.
    for i in range(M):
        var_sum1 += (data1[i] - mean1) ** 2
        var_sum2 += (data2[i] - mean2) ** 2
        cross_sum += (data1[i] * data2[i])

    std1 = (var_sum1 / M) ** .5
    std2 = (var_sum2 / M) ** .5
    cross_mean = cross_sum / M
    
    std1 = std1 + 1e-8
    std2 = std2 + 1e-8
    
    out = (cross_mean - mean1 * mean2) / (std1 * std2)

    return out

@njit
def pairwise_corr(X):
    n, m = X.shape
    out = np.zeros((n, n))
    ranks = np.zeros((n, m))
    for i in range(n):
        ranks[i, :] = rankdata(X[i, :])
    i_idx, j_idx = np.tril_indices_from(out)
    idx = zip(i_idx, j_idx)
    
    # ASSUMES SYMMETRY
    for i, j in idx:
        corr = pearson_corr(ranks[i, :], ranks[j, :])
        out[i, j] = corr
        out[j, i] = corr
    return out

@njit
def numba_rsa_iterator(X, y, y_mask, list_rows):

    par_scores = np.ones((len(list_rows), y.shape[1])) * -999

    for i in range(len(list_rows)):
        
        row = list_rows[i]
        # Get RDM/RSM
        new_x = np.zeros((X.shape[0], len(row)))
        for n, r in enumerate(row):
            new_x[:, n] = X[:, r] + 1e-8 # Avoid division by zero
        pwd = pairwise_corr(new_x)

        sim_data = get_tri(pwd)
        sim_data[np.where(y_mask)] = np.nan
        sim_data = scale_data(sim_data)
        sim_data_reshaped = np.zeros((sim_data.shape[0], 1))
        sim_data_reshaped[:, 0] = sim_data
        if np.any(sim_data_reshaped != 0):
            coefs = ols(sim_data_reshaped, y, y_mask)
            par_scores[np.array(i), :] = coefs
        
    return par_scores

@njit
def coord_transform_numba(x, y, z, affine):
    shape = np.asarray(x).shape
    coords = np.vstack((np.atleast_1d(np.array([x])),
                   np.atleast_1d(np.array([y])),
                   np.atleast_1d(np.array([z])),
                   np.ones_like(np.atleast_1d(np.array([z])))))
    x, y, z, _ = np.dot(affine, coords)
    
    return x.item(), y.item(), z.item()

@njit
def seed_nearest(seeds, affine, mask_coords_):
    nearests_list = []
    for sx, sy, sz in seeds:
        transformed_coords = np.array(coord_transform_numba(sx, sy, sz, np.linalg.inv(affine)))
        nearest = np.round(transformed_coords, 0, np.zeros_like(transformed_coords))
        nearest = (int(nearest[0]), int(nearest[1]), int(nearest[2]))
        try:
            nearests_list.append(mask_coords_.index(nearest))
        except:
            nearests_list.append(None)
    return nearests_list

def apply_mask_and_get_affinity(seeds, niimg, radius, allow_overlap, n_jobs=1,
                                 mask_img=None):
    import time
    start = time.time()
    
    seeds = list(seeds)
    affine = niimg.affine

    # Compute world coordinates of all in-mask voxels.
    mask_img = check_niimg_3d(mask_img)
    mask_img = image.resample_img(mask_img, target_affine=affine,
                                    target_shape=niimg.shape[:3],
                                    interpolation='nearest')
    mask, _ = masking._load_mask_img(mask_img)
    mask_coords = list(zip(*np.where(mask != 0)))

    X = masking._apply_mask_fmri(niimg, mask_img)
    
    # For each seed, get coordinates of nearest voxel
    nearests = joblib.Parallel(n_jobs=n_jobs)(
    joblib.delayed(seed_nearest)(
        seed_chunk, affine, mask_coords)
    for thread_id, seed_chunk in enumerate(np.array_split(seeds, n_jobs)))
    nearests = [i for j in nearests for i in j]
    
    mask_coords = np.asarray(list(zip(*mask_coords)))
    mask_coords = coord_transform(mask_coords[0], mask_coords[1],
                                  mask_coords[2], affine)
    mask_coords = np.asarray(mask_coords).T

    clf = neighbors.NearestNeighbors(radius=radius)
    A = clf.fit(mask_coords).radius_neighbors_graph(seeds)
    A = A.tolil()
    for i, nearest in enumerate(nearests):
        if nearest is None:
            continue
        A[i, nearest] = True
    # Include the voxel containing the seed itself if not masked
    mask_coords = mask_coords.astype(int).tolist()
    for i, seed in enumerate(seeds):
        try:
            A[i, mask_coords.index(seed)] = True
        except ValueError:
            # seed is not in the mask
            pass

    if not allow_overlap:
        if np.any(A.sum(axis=0) >= 2):
            raise ValueError('Overlap detected between spheres')
            
    return X, A


def search_light_rsa(X, y, A, y_mask=None, n_jobs=-1, verbose=0):

    group_iter = GroupIterator(A.shape[0], n_jobs)
    with warnings.catch_warnings():  # might not converge
        warnings.simplefilter('ignore', ConvergenceWarning)
        scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_group_iter_search_light_rsa)(
                A.rows[list_i], 
                X, y, y_mask)
            for thread_id, list_i in enumerate(group_iter))
    return np.concatenate(scores)


class GroupIterator(object):

    def __init__(self, n_features, n_jobs=1):
        self.n_features = n_features
        if n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs

    def __iter__(self):
        split = np.array_split(np.arange(self.n_features), self.n_jobs)
        for list_i in split:
            yield list_i

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def _group_iter_search_light_rsa(list_rows,X, y, y_mask):
    from numba.typed import List

    y_mask_data = y_mask.data
    y_data = y.data.T
    
    # Make list of rows into a nice format
    list_rows_list = list(list_rows)
    list_rows_typedlist = List()
    for i in list_rows_list:
        list_rows_typedlist.append(i)
    
    par_scores = numba_rsa_iterator(X, y_data, y_mask_data, list_rows_typedlist)
    
    return par_scores


##############################################################################
# Class for search_light #####################################################
##############################################################################
class SearchLightRSA(BaseEstimator):

    def __init__(self, mask_img, process_mask_img=None, radius=2.,
                 metric='correlation',
                 n_jobs=1, scoring=None, cv=None, 
                 verbose=0):
        from nltools.data import Adjacency
        self.mask_img = mask_img
        self.process_mask_img = process_mask_img
        self.radius = radius
        self.metric = metric
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose

    def fit(self, imgs, y, y_mask=None, groups=None, X=None, A=None):

        # check if image is 4D
        imgs = check_niimg_4d(imgs)

        # Get the seeds
        process_mask_img = self.process_mask_img
        if self.process_mask_img is None:
            process_mask_img = self.mask_img

        # Compute world coordinates of the seeds
        process_mask, process_mask_affine = masking._load_mask_img(
            process_mask_img)
        process_mask_coords = np.where(process_mask != 0)
        process_mask_coords = coord_transform(
            process_mask_coords[0], process_mask_coords[1],
            process_mask_coords[2], process_mask_affine)
        process_mask_coords = np.asarray(process_mask_coords).T

        import time
        
        start = time.time()
        print("GETTING SEARCHLIGHT SPHERES")
        
        X, A = apply_mask_and_get_affinity(
            process_mask_coords, imgs, self.radius, True,
            mask_img=self.mask_img, n_jobs=self.n_jobs)
        
        self.X = X
        self.A = A
        self.y = y
        self.process_mask = process_mask
        
        elapsed = time.time() - start
        print(elapsed)
                
        print("FITTING")
        scores = search_light_rsa(X, y, None, A, y_mask,self.n_jobs,
                              self.verbose)

        scores_3D = np.zeros((process_mask.shape) + (len(y), )) 
        for i in range(len(y)):
            scores_3D[process_mask, i] = scores[:, i]
        self.scores_ = scores_3D
        return self
