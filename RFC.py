import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV
from scipy import interp
from MLPipe.measures import Measures

class RFC_Pipeline:

    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None, itera=None, cv=None, mean_tprr=None, path_results=None, scoring='f1', run=False):
        self.measure = Measures(run)
        if run:
            run(x_train, y_train, x_test, y_test, itera, cv, mean_tprr, path_results, scoring='f1')
        else:
            self.name = 'NONE'
            self.clf = 0

    def run_grid(self, x_train, y_train, x_test, y_test, itera, cv, mean_tprr, path_results="", scoring='f1'):
        self.run = True
        self.name = 'RFC'
        self.measure.run = True
        
        Paramsrfc = self.RandomGridSearchRFC(
            x_train, y_train, cv, path_results, scoring)
        print("Done Grid Search")
        self.clf = self.TestRFC(x_train, y_train, x_test, y_test, Paramsrfc['n_estimators'], Paramsrfc['max_depth'], Paramsrfc['max_features'],
                                                            Paramsrfc['criterion'], Paramsrfc['min_samples_split'], Paramsrfc['min_samples_leaf'], itera)
        print("Done testing - RFC")

    def RandomGridSearchRFC(self, X, Y, splits, path_results, scoring):
        """
        This function looks for the best set o parameters for RFC method
        Input: 
            X: training set
            Y: labels of training set
            splits: cross validation splits, used to make sure the parameters are stable
        Output:
            clf.best_params_: dictionary with the parameters, to use: param_svm['kernel']
        """
        start = time.time()

        tuned_parameters = {
            'n_estimators': [200, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
            # precomputed,'poly', 'sigmoid'
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 4, 6, 8],
            'min_samples_leaf': [2, 4, 6, 8, 10]
        }

        rfc = RandomForestClassifier(oob_score=True)

        print("RFC Grid Search")
        clf = RandomizedSearchCV(rfc, tuned_parameters, cv=splits,
                                 scoring=scoring, n_jobs=-2)

        clf.fit(X, Y)
        # print("Score",clf.best_score_)
        elapsed = round(time.time() - start, 4)
        print("Total time to process: ", elapsed)

        with open(path_results + "parameters_rfc.txt", "a") as file:
            for item in clf.best_params_:
                file.write(" %s %s" % (item, clf.best_params_[item]))
            file.write(";%s\n" % (elapsed))
        return(clf.best_params_)

    def TestRFC(self, X_train, Y_train, X_test, Y_test, n_estim, max_depth, max_feat, crit, m_s_split, m_s_leaf, itera):
        """
        This function trains and tests the RFC method on the dataset and returns training and testing AUC and the ROC values
        Input:
            X_train: training set
            Y_train: training set labels
            X_test: testing set
            Y_test: testing set
            n_estim:  number of trees
            max_feat: number of features when looking for best split
            crit: criterion for quality of split measure
            itera: iteration of cross validation, used to write down the models 
        Output: 
            result: array with training [0] and testing error[1]    
            fpr_svm: false positive rate, ROc values
            tpr_svm: true positive rate, ROc values
            clf: trained classifier, can be used to combine with others, like superlearner
            probas: probabilities from predict function
            brier: brier score
        """

        clf_rfc = RandomForestClassifier(max_features=max_feat, n_estimators=n_estim, max_depth=max_depth, min_samples_split=m_s_split,
                                         min_samples_leaf=m_s_leaf, oob_score=True, criterion=crit)

        clf_rfc.fit(X_train, Y_train)
        # print(clf.feature_importances_)
        preds = clf_rfc.predict(X_test)
        probas = clf_rfc.predict_proba(X_test)[:, 1]

        fpr_rf, tpr_rf, _ = roc_curve(Y_test, probas)
        # print(feat_imp[itera, :])
        # print(clf_rfc.feature_importances_)
        self.measure.feat_imp.append(clf_rfc.feature_importances_)
        # name=('Models/RFC'+str(itera)+'.pkl')
        # joblib.dump(clf,name)

        self.measure.calculate(Y_test, preds, probas)

        return clf_rfc
