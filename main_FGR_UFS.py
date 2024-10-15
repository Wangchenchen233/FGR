import os
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from FGR import FGR
from FGR_UFS import FGR_UFS
from utility.feature_structure import feature_structure
from utility.subfunc import estimateReg
from utility.unsupervised_evaluation import cluster_evaluation2
from utility.data_load import dataset_pro, construct_label_matrix
import pandas as pd


def process_result_mean_var_column(results_array):
    array_len = results_array.shape[1]
    even_columns = results_array[:, ::2]

    max_values = even_columns.max(axis=0)
    max_indices = even_columns.argmax(axis=0)

    odd_columns_indices = np.arange(1, array_len, 2)
    odd_columns_values = results_array[max_indices, odd_columns_indices]

    result = np.empty(array_len, dtype=results_array.dtype)

    result[::2] = max_values

    result[1::2] = odd_columns_values

    return result


def process_result_mean_var_raw(results_array):
    means = results_array[:, ::2]
    variances = results_array[:, 1::2]

    max_means = []
    corresponding_variances = []

    for i in range(results_array.shape[0]):
        row_means = means[i]
        row_variances = variances[i]

        max_mean_index = np.argmax(row_means)
        max_mean = row_means[max_mean_index]
        corresponding_variance = row_variances[max_mean_index]

        max_means.append(max_mean)
        corresponding_variances.append(corresponding_variance)

    max_means = np.array(max_means)
    corresponding_variances = np.array(corresponding_variances)
    result = np.hstack([results_array, max_means[:, np.newaxis], corresponding_variances[:, np.newaxis]])

    return result


def process_result_mean_max(result_array):
    even_columns = result_array[:, ::2]
    means = even_columns.mean(axis=1)
    variances = even_columns.std(axis=1)
    result = np.hstack([result_array, means[:, np.newaxis], variances[:, np.newaxis]])
    return result


if __name__ == '__main__':
    Data_names = ['lung_discrete', 'urban', 'Isolet', 'mfeat', 'Yale', 'COIL20', 'SRBCT', 'LUNG', 'lymphoma', 'DLBCL',
                  'GLIOMA', 'TOX_171', 'pixraw10P', 'orlraws10P', 'CLL_SUB_111']

    file_path = './results_FGR_UFS/'
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    for data_name in ['lung_discrete']:
        X, y, n_Classes = dataset_pro(data_name, 'scale')
        Y = construct_label_matrix(y)

        Dist_x = pairwise_distances(X) ** 2
        Local_reg, S = estimateReg(Dist_x, 10)
        S = (S + S.T) / 2

        top_percent = None
        M, _ = feature_structure(X, 'gaussian', top_percent)

        Fea_nums = np.arange(1, 11, 1) * n_Classes

        n_run = 10

        paras = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        grid_search = [(beta, eta) for beta in paras for eta in paras]
        # grid_search = [gamma for gamma in paras] # for FGR

        results_fea_all_nmi = np.zeros((10, n_run * 2))
        results_fea_all_acc = np.zeros((10, n_run * 2))
        results_fea_all_ne = np.zeros((10, n_run * 2))
        i_fea = 0
        for fea_num in Fea_nums:
            nmi_results_random = []  # 每次调参最优值和方差
            acc_results_random = []
            ne_results_random = []

            for i_run in np.arange(1, n_run + 1, 1):  # 重复实验
                nmi_result = np.zeros((len(grid_search), 2))
                acc_result = np.zeros((len(grid_search), 2))
                ne_result = np.zeros((len(grid_search), 2))

                pbar = tqdm(grid_search, desc=f'{data_name}, Fea_num={fea_num}, Run={i_run}')
                for i_para, para in enumerate(pbar):
                    beta, gamma = para
                    W = FGR_UFS(X, S, M, beta, gamma, alpha=Local_reg, k=fea_num, r=n_Classes, l=n_Classes,
                                kn=10, random_state=None, hard_thre=False)

                    # gamma = para
                    # W, Z, B = FGR(X, M, gamma, fea_num, r=n_Classes, l=n_Classes, random_state=None, hard_thre=True)

                    idx = np.argsort(W.sum(1), 0)[::-1]

                    nmi_para, acc_para, ne_para = cluster_evaluation2(X, y, n_Classes, idx, 20, [fea_num])

                    nmi_result[i_para, :] = nmi_para
                    acc_result[i_para, :] = acc_para
                    ne_result[i_para, :] = ne_para

                nmi_results_random.append(process_result_mean_var_column(nmi_result))
                acc_results_random.append(process_result_mean_var_column(acc_result))
                ne_results_random.append(process_result_mean_var_column(ne_result))

            results_fea_all_nmi[i_fea, :] = np.concatenate(nmi_results_random)
            results_fea_all_acc[i_fea, :] = np.concatenate(acc_results_random)
            results_fea_all_ne[i_fea, :] = np.concatenate(ne_results_random)
            i_fea += 1

        results_fea_all_nmi2 = process_result_mean_max(results_fea_all_nmi)
        results_all_nmi = process_result_mean_var_raw(results_fea_all_nmi2)
        df = pd.DataFrame(results_all_nmi)
        df.to_excel(file_path + data_name + '-nmi.xlsx', index=False)

        results_fea_all_acc2 = process_result_mean_max(results_fea_all_acc)
        results_all_acc = process_result_mean_var_raw(results_fea_all_acc2)
        df = pd.DataFrame(results_all_acc)
        df.to_excel(file_path + data_name + '-acc.xlsx', index=False)

        results_fea_all_ne2 = process_result_mean_max(results_fea_all_ne)
        results_all_ne = process_result_mean_var_raw(results_fea_all_ne2)
        df = pd.DataFrame(results_all_ne)
        df.to_excel(file_path + data_name + '-ne.xlsx', index=False)
