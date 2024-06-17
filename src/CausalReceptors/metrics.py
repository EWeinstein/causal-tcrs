"""Metrics for evaluating models.

These are essentially the sklearn metrics, only made robust to missing or degenerate data.

"""
import numpy as np
import torch

from sklearn import metrics as skl_metrics
from sklearn.feature_selection import r_regression
from scipy.stats import pearsonr, PermutationMethod


def roc_auc_score(y_true, y_score, check_missing=True):
    if check_missing:
        # Take non-missing values.
        present_indx = y_true > -1
        y_true = y_true[present_indx]
        y_score = y_score[present_indx]
    if len(y_true) == 0:
        return np.nan
    elif (torch.allclose(y_true, torch.tensor(1, dtype=y_true.dtype)) or
          torch.allclose(y_true, torch.tensor(0, dtype=y_true.dtype))):
        return np.nan
    else:
        return skl_metrics.roc_auc_score(y_true, y_score)


def accuracy(y_true, y_score, check_missing=True, one=torch.tensor(1.)):
    if check_missing:
        # Take non-missing values.
        present_indx = y_true > -1
        y_true = y_true[present_indx]
        y_score = y_score[present_indx]
    y_predict = 0.5 * (1 + torch.sign(y_score))
    if len(y_true) == 0:
        return torch.nan
    else:
        # For fast gpu computation, avoid use of isclose (assert statements involve cpu transfer).
        return (-(y_true - y_predict).abs() + one).mean(dim=-1)



def r2_score(y_true, y_pred, base_var=None):
    if len(y_true) == 0:
        return np.nan
    elif base_var is not None:
        # In some cases the baseline variance can be computed from a larger sample,
        # which helps stabilize the estimate of the R2 score.
        return 1. - ((y_true - y_pred)**2).mean()/base_var
    else:
        return skl_metrics.r2_score(y_true, y_pred)


def r_pvalue(x, y):
    return pearsonr(x, y, alternative='greater', method=PermutationMethod()).pvalue


def explained_variance_score(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan
    else:
        return skl_metrics.explained_variance_score(y_true, y_pred)


def pearson_score(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan
    else:
        return np.corrcoef(y_pred, y_true)[0, 1]


def average_precision_score(y_true, y_score, check_missing=True):
    if check_missing:
        # Take non-missing values.
        present_indx = y_true > -1
        y_true = y_true[present_indx]
        y_score = y_score[present_indx]
    if len(y_true) == 0:
        return np.nan
    elif (torch.allclose(y_true, torch.tensor(1, dtype=y_true.dtype)) or
          torch.allclose(y_true, torch.tensor(0, dtype=y_true.dtype))):
        return np.nan
    else:
        return skl_metrics.average_precision_score(y_true, y_score)