from time import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LassoCV, LogisticRegression, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel
from MLPipe.measures import Measures
from MLPipe.pipeline import Pipeline
import numpy as np

class LR(Pipeline):
    def run_grid(self, clf, grid_param, X, Y, cv, scoring, path_results=None, n_jobs=-2, verbose=1, select_feats_logit=False, T=0.1, method='rfc):
        # Measuring time elapsed
        start = time()

        if verbose:
            print("\nStarting feature selection: %s" % self.name)

        feats = np.ones(X.shape[1])
        if select_feats_logit:
            feats = self.Feature_Selection(X, Y, T, method, cv)
            print("Features Selected", sum(feats))
            X = X[:, feats]

        elapsed = round(time() - start, 4)
        if verbose:
            print("Total time to process: ", elapsed)

        # If have path for results, save the best param and time elapsed
        if path_results:
            with open(path_results + "parameters_%s.txt" % self.name, "a") as file:
                for item in feats:
                    file.write(" %s " % (item))
                file.write(";%s\n" % (elapsed))

        return feats

    def run_test(self, clf, X_train, Y_train, X_test, Y_test):
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)

        try:
            decisions = clf.decision_function(X_test)
            probas = (decisions - decisions.min()) / (decisions.max() - decisions.min())      
        except AttributeError:
            probas = clf.predict_proba(X_test)[:, 1]

        # probas=clf.predict_proba(X_test)[:, 1]
        
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
        self.measure.calculate(Y_test, y_pred, probas)

        # name=('Models/RFC'+str(itera)+'.pkl')
        # joblib.dump(clf,name)

        return clf

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
