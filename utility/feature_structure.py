from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from scipy.sparse import *
from utility import construct_W
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist, squareform
import numpy as np
from joblib import Parallel, delayed
import dcor

eps = np.spacing(1)


def estimateReg(distx, k):
    """
    用于计算local自适应结构学习中mu的值
    :param x:
    :param k:
    :return:
    """
    n_sample = distx.shape[0]
    idx = np.argsort(distx)
    distx1 = np.sort(distx)
    a = np.zeros((n_sample, n_sample))
    rr = np.zeros((n_sample, 1))
    for i in range(n_sample):
        di = distx1[i, 1:k + 2]
        rr[i] = 0.5 * (k * di[k] - sum(di[:k]))
        id_ = idx[i, 1:k + 2]
        a[i, id_] = (di[k] - di) / (k * di[k] - sum(di[:k]) + eps)
    r = np.mean(rr)
    return r, a


def compute_distance_correlation(i, j, X):
    if i == j:
        return 1.0  # diagonal elements are 1 (self-correlation)
    else:
        x_i, x_j = X[:, i], X[:, j]
        return dcor.distance_correlation(x_i, x_j)


def feature_structure(X, mode, top_percent):
    A = np.zeros((X.shape[1], X.shape[1]))
    if mode == 'cosine':
        A = cosine_similarity(X.T)
        # A = A * A
    elif mode == 'pearson':
        A = np.corrcoef(X, rowvar=False)
        A = A * A
    elif mode == 'gaussian':
        A = similar_matrix(X.T, 6, 1).todense()
        A = np.array(A)
    elif mode == 'can':
        Dist_x = pairwise_distances(X.T) ** 2
        _, A = estimateReg(Dist_x, 5)
        A = (A + A.T) / 2
    elif mode == 'discorr':
        d = X.shape[1]
        A = np.array(Parallel(n_jobs=-1)(delayed(compute_distance_correlation)
                                         (i, j, X) for i in range(d) for j in range(d))).reshape(d, d)
    if top_percent:
        threshold = np.percentile(A, 100 - top_percent)
        A_new = np.where(A > threshold, A, 0)
        L_A_new = np.diag(A_new.sum(1)) - A_new
        return A_new, L_A_new
    else:
        return A, np.diag(A.sum(1)) - A


def similar_matrix(x, k, t_c):
    """
    :param t_c: scale for para t
    :param x: N D
    :param k:
    :return:
    """
    # compute pairwise euclidean distances
    n_samples, n_features = x.shape
    D = pairwise_distances(x)
    D **= 2
    # sort the distance matrix D in ascending order
    dump = np.sort(D, axis=1)
    idx = np.argsort(D, axis=1)
    # 0值:沿着每一列索引值向下执行方法(axis=0代表往跨行)分别对每一列
    # 1值:沿着每一行(axis=1代表跨列) 分别对每一行
    idx_new = idx[:, 0:k + 1]
    dump_new = dump[:, 0:k + 1]
    # compute the pairwise heat kernel distances
    # t = np.percentile(D.flatten(), 20)  # 20210816 tkde13
    t = np.mean(D)
    t = t_c * t
    dump_heat_kernel = np.exp(-dump_new / (2 * t))
    G = np.zeros((n_samples * (k + 1), 3))
    G[:, 0] = np.tile(np.arange(n_samples), (k + 1, 1)).reshape(-1)  # 第一个参数为Y轴扩大倍数，第二个为X轴扩大倍数
    G[:, 1] = np.ravel(idx_new, order='F')  # 按列顺序重塑 n_samples*(k+1)
    G[:, 2] = np.ravel(dump_heat_kernel, order='F')
    # build the sparse affinity matrix W
    W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
    bigger = np.transpose(W) > W
    W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
    # np.transpose(W).multiply(bigger)不等于np.multiply(W,bigger)
    return W


def feature_structure_intra_class(X, y, n_class, mode, top_percent):
    L_M = []
    M = []
    for i in range(n_class):
        A, LA = feature_structure(X[y == i + 1], mode, top_percent)
        L_M.append(LA)
        M.append(A)
        # if top_percent:
        #     threshold = np.percentile(A, 100 - top_percent)
        #     A_new = np.where(A > threshold, A, 0)
        #     L_A_new = np.diag(A_new.sum(1)) - A_new
        #     L_M.append(L_A_new)
        #     M.append(A_new)
        # else:
        #     L_M.append(LA)
        #     M.append(A)
    return L_M, M


def distcorr(X, Y):
    """ Compute the distance correlation function

    # >>> a = [1,2,3,4,5]
    # >>> b = np.array([1,2,9,4,4])
    # >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor
