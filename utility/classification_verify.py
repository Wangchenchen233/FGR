from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score


def evaluate_classifier(model, X_sub_train, X_sub_test, y_train, y_test):
    model.fit(X_sub_train, y_train)
    y_pred = model.predict(X_sub_test)
    accuracy = accuracy_score(y_test, y_pred)
    # multiclass_auc = roc_auc_score(y_test, y_pred, multi_class="ovr", average="macro")
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, f1


def classification_verify(X_sub_train, X_sub_test, y_train, y_test):
    Cs = [0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1]

    accuracies = {}
    f1_scores = {}

    for C in Cs:
        for gamma in gammas:
            svm_model = SVC(C=C, kernel='rbf', gamma=gamma)
            svm_model.fit(X_sub_train, y_train)
            y_pred = svm_model.predict(X_sub_test)
            accuracy_key = f"SVM_C={C}_gamma={gamma}"
            accuracies[accuracy_key] = accuracy_score(y_test, y_pred)
            f1_scores[accuracy_key] = f1_score(y_test, y_pred, average='weighted')

    for k in [1, 3, 5, 7, 9]:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_sub_train, y_train)
        knn_pred = knn_model.predict(X_sub_test)
        accuracy_key = f"KNN_K={k}"
        accuracies[accuracy_key] = accuracy_score(y_test, knn_pred)
        f1_scores[accuracy_key] = f1_score(y_test, knn_pred, average='weighted')

    rf_model = RandomForestClassifier()
    rf_model.fit(X_sub_train, y_train)
    rf_pred = rf_model.predict(X_sub_test)
    accuracies["Random Forest"] = accuracy_score(y_test, rf_pred)
    f1_scores["Random Forest"] = f1_score(y_test, rf_pred, average='weighted')

    nb_model = GaussianNB()
    nb_model.fit(X_sub_train, y_train)
    y_pred = nb_model.predict(X_sub_test)
    accuracies["Naive Bayes"] = accuracy_score(y_test, y_pred)
    f1_scores["Naive Bayes"] = f1_score(y_test, y_pred, average='weighted')
    return accuracies, f1_scores
