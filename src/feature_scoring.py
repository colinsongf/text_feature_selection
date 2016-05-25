# -*- coding: utf-8 -*-

from numpy import inf
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import binarize
from scipy.sparse import issparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot


def _entropy(P):
    """
    Function that calculates the entropy of a random variable measure in bits.

    H(X) = \sum_{x \in X} p(X=x) * log_2(1/p(X=x))

    Parameters
    ---------
    P: list
        A list containing the probability for each posible state of the r.v.
    """

    #TODO remove the "+ 1e-20" inside the log2 computation
    # it's just a hack to avoid to compute log2(0)
    ent = -1.0 * np.sum(P * np.log2(P+1e-20), axis=0)
    return ent

def ig(X, y):
    """
    This method calculates the information gain for two random variables I(X, Y).
    """

    # binarization: from counts to presence/abscence
    binarize(X, threshold=0.0, copy=False)

    # una columna por cada clase
    Y = LabelBinarizer().fit_transform(y)
    if Y.shape[1] == 1: # binary problem case
        Y = np.append(1-Y, Y, axis=1)

    Y_prob = (np.sum(Y, axis=0, dtype=np.float64) / len(Y)).reshape(-1, 1)

    # calculate the class entropy H(Y)
    class_entropy = _entropy(Y_prob)

    X_y_count = safe_sparse_dot(Y.T, X)
    # TODO XXX FIXME ver si estoy calculando bien esta probabilidad
    X_y_prob = \
        X_y_count / np.sum(X_y_count, axis=0, dtype=np.float64)

    # calculate the conditional entropy of the class given the feature H(y|f_i)
    cond_entropy = _entropy(X_y_prob) # TODO XXX FIXME ver si estoy calculando bien la entropia condicional
    print "class:", class_entropy
    print "cond_entropy:", cond_entropy

    infogain = class_entropy - cond_entropy

    return infogain, None

def _z_score(X):
    """
    Computes the standard Normal distribution’s inverse cumulative probability function
    """
    _validate_bounded_values(X, 0, 1)
    return norm.ppf(X)

def _validate_bounded_values(X, lower, upper):
    if np.any(X <= lower) or np.any(X >= upper):
        raise ValueError("Invalid value for parameter X. Valid values are in range (0, 1).")

# lower and upper bound defaults were taken from this paper:
# http://www.jmlr.org/papers/volume3/forman03a/forman03a.pdf
def bounded(a, lower=0.0005, upper=0.9995):
    a[a <= lower] = lower
    a[a >= upper] = upper
    return a

def bns(X, y):
    """
    Implements the bi-normal separation scoring.
    """

    # binarization: from counts to presence/abscence
    binarize(X, threshold=0.0, copy=False)

    # one column per class
    Y = LabelBinarizer().fit_transform(y)
    if Y.shape[1] == 1: # binary problem case
        Y = np.append(1-Y, Y, axis=1)

    pos = np.sum(Y, axis=0)
    neg = Y.shape[0] - pos

    tp = safe_sparse_dot(X.T, Y)
    fp = np.sum(tp, axis=1).reshape(-1, 1) - tp

    tpr = bounded(tp/pos.astype(float))
    fpr = bounded(fp/neg.astype(float))

    bns = np.abs(_z_score(tpr) - _z_score(fpr))

    return bns[:,1], None
