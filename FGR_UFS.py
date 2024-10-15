# -*- coding: utf-8 -*-
# @Time    : 7/28/2024 9:36 AM
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : FIX_FG_model2.py


import numpy as np
from sklearn.metrics import pairwise_distances

from utility.init_func import orthogonal_
from utility.subfunc import eig_lastk, local_structure_learning


def FGR_UFS(X, S, M, beta, gamma, alpha, k, r, l, kn, random_state, hard_thre):
    np.random.seed(random_state)
    lambda_w = 1e10  # A^TA=I的约束
    lambda_f = alpha
    n, d = X.shape

    Ls = np.diag(S.sum(0)) - S

    W = np.random.rand(d, k)
    W = orthogonal_(W, random_state)

    # A = scipy.linalg.orth(A)
    np.abs(W, out=W)
    W[W < 1e-6] = 1e-6

    P = np.random.rand(k, r)
    P = orthogonal_(P, random_state)

    Y, _ = eig_lastk(Ls, r)

    F = np.zeros((n, k))  # init F

    gamm_beta = gamma / 2
    Z = np.zeros((k, k))
    D = np.zeros((k, k))
    B = np.zeros((k, k))

    max_iter = 30
    obj = np.zeros(max_iter)
    for iter_ in range(max_iter):
        # update S
        XAB = X @ W @ P
        dist_XAB = pairwise_distances(XAB) ** 2
        dist_y = pairwise_distances(Y) ** 2
        dist_f = lambda_f * dist_y
        S = local_structure_learning(kn, alpha, dist_XAB, dist_f, 1)
        Ls = np.diag(S.sum(0)) - S

        # update W
        XLX = X.T @ Ls @ X
        W_down = beta * XLX @ W @ P @ P.T + M @ W @ W.T @ M @ W + lambda_w * W @ W.T @ W
        temp = np.divide(lambda_w * W + M @ W @ B, W_down)
        W = W * np.array(temp)

        temp = np.diag(np.sqrt(np.diag(1 / (np.dot(W.transpose(), W) + 1e-16))))
        W = np.dot(W, temp)

        # hard thre
        if hard_thre:
            row_norms = np.linalg.norm(W, axis=1)
            W[W < 0.001] = 0
            top_k_indices = np.argsort(row_norms)[-k:]
            top_k = np.sort(top_k_indices)
            W_new = np.zeros_like(W)
            W_new[top_k, :] = W[top_k, :]
            W = W_new

        Zk = Z.copy()
        Z = W.T @ M @ W

        # update B
        Bk = B.copy()
        B = Z - gamm_beta * (np.diag(D).reshape(-1, 1) - D)
        B = np.maximum(0, (B + B.T) / 2)
        np.fill_diagonal(B, 0)
        LB = np.diag(B @ np.ones((k, 1)).ravel()) - B

        # update D
        V, _ = eig_lastk(LB, l)
        D = V @ V.T

        # update P
        P, _ = eig_lastk(W.T @ XLX @ W, r)

        # update Y
        Y_old = Y
        Y, e_val = eig_lastk(Ls, r)

        diffZ = np.max(np.abs(Z - Zk))
        diffB = np.max(np.abs(B - Bk))
        stopC = max(diffZ, diffB)
        if stopC < 1e-3:
            break

        fn1 = np.sum(e_val[:r])
        fn2 = np.sum(e_val[:r + 1])
        if fn1 > 10e-10:
            lambda_f = 2 * lambda_f
        elif fn2 < 10e-10:
            lambda_f = lambda_f / 2
            Y = Y_old
        else:
            break

    return W
