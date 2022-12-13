import pyriemann
import numpy as np

def _check_dimensions(X, Cref):
    n_channels_1, n_channels_2 = X.shape[-2:]
    n_channels_3, n_channels_4 = Cref.shape
    if not (n_channels_1 == n_channels_2 == n_channels_3 == n_channels_4):
        raise ValueError("Inputs have incompatible dimensions.")
        
        
def upper(X):
 
    n_channels = X.shape[-1]
    idx = np.triu_indices_from(np.empty((n_channels, n_channels)))
    coeffs = (np.sqrt(2) * np.triu(np.ones((n_channels, n_channels)), 1) +
              np.eye(n_channels))[idx]
    T = coeffs * X[..., idx[0], idx[1]]
    return T

def log_map_riemann(X, Cref, C12=False):

    _check_dimensions(X, Cref)
    Cm12 = pyriemann.utils.base.invsqrtm(Cref)
    X_new = pyriemann.utils.base.logm(Cm12 @ X @ Cm12)
    if C12:
        C12 = pyriemann.utils.base.sqrtm(Cref)
        X_new = C12 @ X_new @ C12
    return X_new

def get_riemann_features(X_tensor):
    X_ = np.zeros((len(X_tensor),3, 3))
    for i in range(len(X_tensor)):
            X_[i,:,:] = (X_tensor[i,:,:]@X_tensor[i,:,:].T)/(X_tensor[i,:,:].shape[1])
            
    S_s = log_map_riemann(X_, mean.mean_riemann(X_))
    return upper(S_s)