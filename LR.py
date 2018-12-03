import numpy as np
import time
from scipy import interp
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression, ElasticNetCV
from sklearn.feature_selection import RFECV, SelectFromModel
from MLPipe.measures import Measures

class LR_Pipeline:

    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None, itera=None, cv=None, mean_tprr=None, select_feats_logit=False, T=0.1, method='rfc', run=False):
        self.measure = Measures(run)
        if run:
            self.run(x_train, y_train, x_test, y_test, itera, cv, mean_tprr, select_feats_logit=False, T=0.1, method='rfc')

        else:
            self.name = 'NONE'
            self.clf = 0

    def run_grid(self, x_train, y_train, x_test, y_test, itera, cv, mean_tprr, select_feats_logit=False, T=0.1, method='rfc'):
        
        self.run = True
        self.name = 'LR'
        self.measure.run = True

        feats = np.ones(x_train.shape[1])
        if select_feats_logit:
            feats = self.Feature_Selection(x_train, y_train, T, method, cv)
            print("Features Selected", sum(feats))
            x_train = x_train[:, feats]
            x_test = x_test[:, feats]

        self.clf = self.TestLogistic(x_train, y_train, x_test, y_test, itera, feats)
        print("Done testing - LR")

    def Feature_Selection(self, X, y, T, method, cv):
        """
        This functions returns only the features selected by the method using the threshold selected.
        We advise to run this function with several thresholds and look for the best, put this function inside a loop and see how it goes
        Suggestions for the range of t, thresholds = np.linspace(0.00001, 0.1, num=10)
        Input: 
            X=training set
            y=training labels
            T=threshold selected
            which method= 'rfc', 'lasso', 'elastik'
            cv= number of cross validation iterations
        Output:
        Boolean array with the selected features,with this you can X=X[feats] to select only the relevant features
        """
        alphagrid = np.linspace(0.001, 0.99, num=cv)

        clf = {
            'rfc': RandomForestClassifier(),
            'lasso': LassoCV(),  # alphas=alphagrid),
            'elastik': ElasticNetCV(alphas=alphagrid),
            'backward': RFECV(LogisticRegression(), cv=cv, n_jobs=-2)

        }[method]
        if method == 'backward':
            clf = clf.fit(X, y)
            feats = clf.support_
        else:
            clf.fit(X, y)
            sfm = SelectFromModel(clf)  # , threshold=T)
            print(X.shape)
            sfm.fit(X, y)
            feats = sfm.get_support()

        return(feats)

    def TestLogistic(self, X_train, Y_train, X_test, Y_test, itera, feats):

        clf = LogisticRegression(C=100000, solver="liblinear")

        clf.fit(X_train, Y_train)

        preds = clf.predict(X_test)
        probas = clf.predict_proba(X_test)[:, 1]

        odds = np.exp(clf.coef_)
        feats = np.array(feats, dtype='float64')
        pos = 0
        for i in range(0, feats.shape[0]):
            if feats[i] == 1:
                feats[i] = odds[0, pos]
                # print(odds[0,pos])
                pos = pos + 1
        # print(feats)
        self.measure.feat_imp.append(feats)
        # print("classes", clf.classes_)
        # name=('Models/RFC'+str(itera)+'.pkl')
        # joblib.dump(clf,name)

        self.measure.calculate(Y_test, preds, probas)

        return clf
