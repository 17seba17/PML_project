import numpy as np


def fit_ols(tg_anom_train, fgmt_train):
    """
    tg_anom = alpha + beta * fgmt
    """
    X_mat = np.column_stack([np.ones_like(fgmt_train), fgmt_train])  # (N_time, 2)
    
    params, _, _, _ = np.linalg.lstsq(X_mat, tg_anom_train, rcond=None)
    
    alpha = params[0, :]
    beta = params[1, :]
    
    return alpha, beta

