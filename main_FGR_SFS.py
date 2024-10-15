import os.path
import numpy as np
from tqdm import tqdm

from FGR import FGR
from utility.classification_verify import classification_verify

from FGR_SFS import FGR_SFS
from utility.feature_structure import feature_structure
from utility.data_load import dataset_pro, construct_label_matrix
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    Data_names = ['lung_discrete', 'urban', 'Isolet', 'mfeat', 'Yale', 'COIL20', 'SRBCT', 'LUNG', 'lymphoma', 'DLBCL',
                  'GLIOMA', 'TOX_171', 'pixraw10P', 'orlraws10P', 'CLL_SUB_111']

    file_path = './results_FGR_SFS/'
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    for dname in ['lung_discrete']:
        X, y, n_Classes = dataset_pro(dname, 'scale')
        Y = construct_label_matrix(y)

        top_percent = None
        M, _ = feature_structure(X, 'gaussian', top_percent)

        Fea_nums = np.arange(1, 11, 1) * n_Classes

        n_run = 1
        n_split = 20

        n_model = 16
        model_names = []

        paras = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        grid_search = [(gamma, lambda_lsr) for gamma in paras for lambda_lsr in paras]
        # grid_search = [gamma for gamma in paras] # for FGR

        results_fea_all_acc = []
        results_fea_all_f1 = []
        for i_fea, fea_num in enumerate(Fea_nums):
            results_repeat_acc = np.zeros((n_run, n_model))
            results_repeat_f1 = np.zeros((n_run, n_model))
            for i_run in range(n_run):
                results_paras_acc = np.zeros((len(grid_search), n_model))
                results_paras_f1 = np.zeros((len(grid_search), n_model))
                pbar = tqdm(grid_search, desc=f'{dname}, Fea_num={fea_num}, Run={i_run + 1}')
                for i_para, para in enumerate(pbar):
                    gamma, lambda_lsr = para
                    # gamma = para # for FGR
                    accuracies_para = np.zeros((n_split, n_model))
                    f1_score_para = np.zeros((n_split, n_model))
                    for i_split in range(n_split):

                        X_train, X_test, y_train, y_test, Y_train, Y_test = train_test_split(X, y, Y, test_size=0.2,
                                                                                             random_state=i_split)

                        Mt, _ = feature_structure(X_train, 'gaussian', top_percent)
                        # FGR-SFS
                        W = FGR_SFS(X_train, Y_train, Mt, gamma, lambda_lsr, fea_num, r=n_Classes, l=n_Classes,
                                    random_state=None, hard_thre=False)

                        # FGR
                        # W, Z, B = FGR(X_train, Mt, gamma, fea_num, r=n_Classes, l=n_Classes, random_state=None, hard_thre=True)


                        idx = np.argsort(W.sum(1), 0)[::-1]

                        X_sub = X_train[:, idx[0:fea_num]]
                        X_sub_train = X_train[:, idx[0:fea_num]]
                        X_sub_test = X_test[:, idx[0:fea_num]]

                        result_accs, result_f1s = classification_verify(X_sub_train, X_sub_test, y_train, y_test)
                        for i, (key, value) in enumerate(result_accs.items()):
                            accuracies_para[i_split, i] = value
                        for i, (key, value) in enumerate(result_f1s.items()):
                            f1_score_para[i_split, i] = value
                        if not model_names:
                            model_names = list(result_accs.keys())

                    # 计算20次的均值
                    results_paras_acc[i_para, :] = np.mean(accuracies_para, axis=0)
                    results_paras_f1[i_para, :] = np.mean(f1_score_para, axis=0)

                results_repeat_acc[i_run, :] = np.max(results_paras_acc, 0)
                results_repeat_f1[i_run, :] = np.max(results_paras_f1, 0)
                print(np.max(results_repeat_acc[i_run, :]), np.max(results_repeat_f1[i_run, :]))
            results_fea_all_acc.append(results_repeat_acc)
            results_fea_all_f1.append(results_repeat_f1)

        writer = pd.ExcelWriter(file_path + dname + "_acc.xlsx")
        for i, results_paras in enumerate(results_fea_all_acc):
            results_paras_df = pd.DataFrame(results_paras, columns=model_names)
            results_paras_df.to_excel(writer, sheet_name=f'Fea_num={Fea_nums[i]}', index=False)
        writer.save()
        writer = pd.ExcelWriter(file_path + dname + "_f1.xlsx")
        for i, results_paras in enumerate(results_fea_all_f1):
            results_paras_df = pd.DataFrame(results_paras, columns=model_names)
            results_paras_df.to_excel(writer, sheet_name=f'Fea_num={Fea_nums[i]}', index=False)
        writer.save()
