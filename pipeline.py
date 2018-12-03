import numpy as np
from time import time
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pickle
from MLPipe.measures import Measures

class Pipeline():
    def __str__(self):
        return "name: " + str(self.name) +\
                "\nmodel_list: " + str(self.model_list) +\
                "\nmeasure: " + str(self.measure) +\
                "\nclf: " + str(self.clf) +\
                "\ngrid_param: " + str(self.grid_param) +\
                "\nX_train: " + str(self.X_train.shape) +\
                "\tY_train: " + str(self.Y_train.shape) +\
                "\tX_test: " + str(self.X_test.shape) +\
                "\tY_test: " + str(self.Y_test.shape) +\
                "\nitera: " + str(self.itera) +\
                "\tcv: " + str(self.cv) +\
                "\tpath_results: " + str(self.path_results) +\
                "\tscoring: " + str(self.scoring) +\
                "\nn_jobs: " + str(self.n_jobs) +\
                "\tverbose: " + str(self.verbose) +\
                "\trun: " + str(self.run)

    def __init__(self, clf, grid_param=None, X_train=None, Y_train=None, X_test=None, Y_test=None, itera=1, cv=None, path_results=None, scoring='f1', measures=None, n_jobs=-2, verbose=1, name="", fit_params={}, save=False, run=False):
        # Get classifier name
        self.name = str(clf).split("(")[0] + name

        # Initialize model list
        self.model_list = []

        # Initialize measures
        if not measures:
            self.measure = Measures()
        else:
            self.measure = measures

        # Set parameters
        self.clf = clf
        self.grid_param = grid_param
        self.fit_params = fit_params
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.itera = itera
        self.cv = cv
        self.path_results = path_results
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.run = run
        
        # If set to run
        if run:
            self.run_pipeline(save=save)

    def save_results(self):
        if self.path_results:
            path_save = self.path_results
        else:
            path_save = ""
        name = path_save + "models_%s.pkl" % self.name

        filehandler = open(name, 'wb')
        pickle.dump(self, filehandler)

    def run_pipeline(self, save=False):

        # Run gridsearch and training 'itera' times
            for _ in range(self.itera):
                clf_tuned = self.run_grid(self.clf, self.grid_param, self.X_train, self.Y_train, self.cv, self.scoring, self.path_results, self.n_jobs, self.verbose, self.fit_params)
                model = self.run_test(clf_tuned, self.X_train, self.Y_train, self.X_test, self.Y_test)
                self.model_list.append(model)

            if save:
                self.save_results()

    def run_grid(self, clf, grid_param, X, Y, cv, scoring, path_results=None, n_jobs=-2, verbose=1, fit_params={}):
        print(scoring, path_results, n_jobs, verbose, fit_params)
        # Measuring time elapsed
        start = time()

        if verbose:
            print("\nStarting grid search: %s" % self.name)

        if grid_param:
            # Initialize and run randomized gridsearch
            grid = RandomizedSearchCV(clf, grid_param, cv=cv, n_iter=100, fit_params=fit_params,
                                    scoring=scoring, n_jobs=n_jobs, verbose=verbose)

            # grid = GridSearchCV(clf, grid_param, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose, fit_params=fit_params)
        else:
            grid = clf
        grid.fit(X, Y)

        elapsed = round(time() - start, 4)
        if verbose:
            print("Total time to process: ", elapsed)

        best_estimator = grid
        optimized = hasattr(grid, 'best_params_')
        if optimized:
            best_estimator = grid.best_estimator_

        # If have path for results, save the best param and time elapsed
        if path_results:
            with open(path_results + "parameters_%s.txt" % self.name, "a") as file:
                if optimized:
                    for item in grid.best_params_:
                        file.write(" %s %s" % (item, grid.best_params_[item]))
                file.write(";%s\n" % (elapsed))

        return best_estimator

    def run_test(self, clf, X_train, Y_train, X_test, Y_test):
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)

        try:
            decisions = clf.decision_function(X_test)
            probas = (decisions - decisions.min()) / (decisions.max() - decisions.min())      
        except AttributeError:
            probas = clf.predict_proba(X_test)[:, 1]

        self.measure.calculate(Y_test, y_pred, probas)

        return clf
