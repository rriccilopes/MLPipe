import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV
from scipy import interp
from MLPipe.measures import Measures

class SVM_Pipeline:
    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None, itera=None, cv=None, mean_tprr=None, path_results=None, scoring='f1', run=False):
        self.measure = Measures(run)
        if run:
            self.run(x_train, y_train, x_test, y_test, itera, cv, mean_tprr, path_results, scoring='f1')
        else:
            self.name = 'NONE'
            self.clf = 0

    def run_grid(self, x_train, y_train, x_test, y_test, itera, cv, mean_tprr, path_results, scoring='f1'):
        self.run = True
        self.name = 'SVM'
        self.measure.run = True

        Paramssvm = self.RandomGridSearchSVM(x_train, y_train, cv, path_results, scoring)
        print("Done Grid Search")
        self.clf = self.TestSVM(x_train, y_train, x_test, y_test, Paramssvm['kernel'],
                                                            Paramssvm['C'], Paramssvm['gamma'], Paramssvm['degree'], itera)
        print("Done testing - SVM")


    def RandomGridSearchSVM(self, X, Y, splits, path_results, scoring):
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
            'C': [0.1, 0.01, 0.001, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'degree': [1, 2, 3, 4, 5, 6],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
            #'tol':  [1, 0.1, 0.01, 0.001, 0.0001],
        }

        print("SVM Grid Search")
        clf = RandomizedSearchCV(SVC(), tuned_parameters, cv=splits,
                                 scoring=scoring, n_jobs=-2)

       # clf =  GridSearchCV(SVC(), tuned_parameters, cv=splits,
       #                scoring='%s' % scoring,n_jobs=-1)
        clf.fit(X, Y)

        elapsed = round(time.time() - start, 4)
        print("Total time to process: ", elapsed)

        with open(path_results + "parameters_svm.txt", "a") as file:
            for item in clf.best_params_:
                file.write(" %s %s" % (item, clf.best_params_[item]))
            file.write(";%s\n" % (elapsed))
        return(clf.best_params_)

    def TestSVM(self, X_train, Y_train, X_test, Y_test, kernel, C, gamma, deg, itera):
        """
        This function trains and tests the SVM method on the dataset and returns training and testing AUC and the ROC values
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

        clf_svm = SVC(C=C, kernel=kernel, gamma=gamma,
                      degree=deg, probability=True)

        clf_svm.fit(X_train, Y_train)
        preds = clf_svm.predict(X_test)
        decisions = clf_svm.decision_function(X_test)
        probas = (decisions - decisions.min()) / (decisions.max() - decisions.min())
        #probas=clf.predict_proba(X_test)[:, 1]

        self.measure.calculate(Y_test, preds, probas)

        # name=('Models/RFC'+str(itera)+'.pkl')
        # joblib.dump(clf,name)

        return clf_svm
