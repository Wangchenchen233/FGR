import numpy as np
from utility.init_func import orthogonal_
from utility.subfunc import eig_lastk


def FGR(X, M, gamma, k, r, l, random_state=None, hard_thre=False):
    np.random.seed(random_state)
    lambda_w = 1e16  # A^TA=I的约束
    n, d = X.shape

    # init W
    W = np.random.rand(d, k)
    W = orthogonal_(W, random_state)
    np.abs(W, out=W)
    W[W < 1e-6] = 1e-6

    P = np.zeros((k, r))

    gamm_beta = gamma / 2

    Z = np.zeros((k, k))
    D = np.zeros((k, k))
    B = np.zeros((k, k))

    max_iter = 50
    obj = np.zeros(max_iter)
    for iter_ in range(max_iter):
        # update W
        WW = W @ W.T
        MW = M @ W
        W_down = M @ WW @ MW + lambda_w * WW @ W
        temp = (MW @ B + lambda_w * W) / (W_down + 1e-16)
        W *= temp

        temp = np.diag(np.sqrt(np.diag(1 / (np.dot(W.transpose(), W) + 1e-16))))
        W = np.dot(W, temp)

        # hard thre
        if hard_thre:
            row_norms = np.linalg.norm(W, axis=1)
            W[W < 0.001] = 0
            top_70_indices = np.argsort(row_norms)[-100:]
            top_70 = np.sort(top_70_indices)
            W_new = np.zeros_like(W)
            W_new[top_70, :] = W[top_70, :]
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

        diffZ = np.max(np.abs(Z - Zk))
        diffB = np.max(np.abs(B - Bk))
        stopC = max(diffZ, diffB)
        if stopC < 1e-3:
            break

    return W, Z, B
