import numpy as np
import time
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV
from scipy import interp
from MLPipe.measures import Measures

class SL_Pipeline:
    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None, l=None, mean_tprr=None, class_rfc=None, class_svm=None, class_lr=None, class_nn=None, run=False):
        self.measure = Measures(run)
        if run:
            self.run = True
            self.run(x_train, y_train, x_test, y_test, l, mean_tprr, class_rfc, class_svm, class_lr, class_nn)

        else:
            self.name = 'NONE'
            self.clf = 0

    def run_grid(self, x_train, y_train, x_test, y_test, l, mean_tprr, class_rfc, class_svm, class_lr, class_nn):
        if run:
            self.name = 'SL'
            self.measure.run = True

            list_clfs = [class_rfc, class_svm, class_lr, class_nn]
            list_clfs = [x.clf for x in list_clfs if x.clf != 0]
            names = [class_rfc.name, class_svm.name,
                     class_lr.name, class_nn.name]
            names = [x for x in names if x != 'NONE']
            self.clf = self.TrainAndTestSL(x_train, y_train, x_test, y_test, l, list_clfs, names)
            print("Done testing - SL")

    def TrainAndTestSL(self, X_train, Y_train, X_test, Y_test, itera, class_list, libnames):

        sl_clf = sl.SuperLearner(class_list, libnames, loss="nloglik")
        #sl_clf=sl.SuperLearner(lib, libnames, loss="L2")

        sl_clf.fit(X_train, Y_train)

        probas = sl_clf.predict(X_test)
        preds = probas > 0.5
        fpr_rf, tpr_rf, _ = roc_curve(Y_test, probas)

        self.measure.calculate(Y_test, preds, probas)

        return sl_clf
