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

# otra implementacion de entropy (sacada del source de scikit learn)
    # def entropy(samples):
    # n_samples = len(samples)
    # entropy = 0.
    #
    # for count in bincount(samples):
    #     p = 1. * count / n_samples
    #     if p > 0:
    #         entropy -= p * np.log2(p)
    #
    # return entropy

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
    # it's just a hack to avoid compute log2(0)
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

    # calculate class entropy H(Y)
    class_entropy = _entropy(Y_prob)

    X_y_count = safe_sparse_dot(Y.T, X)
    # TODO XXX FIXME ver si estoy calculando bien esta probabilidad
    X_y_prob = \
        X_y_count / np.sum(X_y_count, axis=0, dtype=np.float64)

    # calculate class entropy given feature H(y|f_i)
    cond_entropy = _entropy(X_y_prob) # TODO XXX FIXME ver si estoy calculando bien la entropia condicional
    print "class:", class_entropy
    print "cond_entropy:", cond_entropy

    infogain = class_entropy - cond_entropy

    return infogain, None

def infogain_score(X, y):

    def get_t1(fc, c, f):
        t = np.log2(fc/(c * f))
        t[~np.isfinite(t)] = 0
        return np.multiply(fc, t)

    def get_t2(fc, c, f):
        t = np.log2((1-f-c+fc)/((1-c)*(1-f)))
        t[~np.isfinite(t)] = 0
        return np.multiply((1-f-c+fc), t)

    def get_t3(c, f, class_count, observed, total):
        nfc = (class_count - observed)/total
        t = np.log2(nfc/(c*(1-f)))
        t[~np.isfinite(t)] = 0
        return np.multiply(nfc, t)

    def get_t4(c, f, feature_count, observed, total):
        fnc = (feature_count - observed)/total
        t = np.log2(fnc/((1-c)*f))
        t[~np.isfinite(t)] = 0
        return np.multiply(fnc, t)

    X = check_array(X, accept_sparse='csr')
    if np.any((X.data if issparse(X) else X) < 0):
        raise ValueError("Input X must be non-negative.")

    Y = LabelBinarizer().fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    # counts
    observed = safe_sparse_dot(Y.T, X)          # n_classes * n_features
    total = observed.sum(axis=0).reshape(1, -1).sum()
    feature_count = X.sum(axis=0).reshape(1, -1)
    class_count = (X.sum(axis=1).reshape(1, -1) * Y).T

    # probs
    f = feature_count / feature_count.sum()
    c = class_count / float(class_count.sum())
    fc = observed / total

    # the feature score is averaged over classes
    scores = (get_t1(fc, c, f) +
            get_t2(fc, c, f) +
            get_t3(c, f, class_count, observed, total) +
            get_t4(c, f, feature_count, observed, total)).mean(axis=0)

    scores = np.asarray(scores).reshape(-1)

    return scores, None

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
