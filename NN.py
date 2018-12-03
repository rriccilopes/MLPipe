import numpy as np
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV
from scipy import interp
from MLPipe.measures import Measures

class NN_Pipeline:

    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None, itera=None, cv=None, mean_tprr=None, path_results=None, scoring='f1', run=False):
        self.measure = Measures(run)
        if run:
            self.run(x_train, y_train, x_test, y_test, itera, cv, mean_tprr, path_results, scoring='f1')
        else:
            self.name = 'NONE'
            self.clf = 0

    def run_grid(self, x_train, y_train, x_test, y_test, itera, cv, mean_tprr, path_results, scoring='f1'):
        self.run = True
        self.name = 'NN'
        self.measure.run = True

        Paramsnn = self.RandomGridSearchNN(
            x_train, y_train, cv, path_results, scoring)
        print("Done Grid Search")
        self.clf = self.TestNN(x_train, y_train, x_test, y_test, Paramsnn.get('activation'), Paramsnn.get('hidden_layer_sizes'), Paramsnn.get('alpha'), Paramsnn.get('batch_size'),
                                                            Paramsnn.get('learning_rate_init'), Paramsnn.get('solver'), itera)
        print("Done testing - NN")


    def RandomGridSearchNN(self, X, Y, splits, path_results, scoring='f1'):
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
            'activation': ['relu', 'logistic', 'tanh'],
            'hidden_layer_sizes': [[80, 160, 80], [78, 156, 78], [88, 176, 88], [80, 160]],
            'alpha': [0.1, 0.01, 0.001, 0.0001],
            'batch_size': [32, 64],
            'learning_rate_init': [0.01, 0.001],
            'solver': ["adam"]
        }

        print("NN Grid Search")
        mlp = MLPClassifier(max_iter=5000)
        clf = RandomizedSearchCV(
            mlp, tuned_parameters, cv=splits, scoring=scoring, n_jobs=-1)

        clf.fit(X, Y)

        elapsed = round(time.time() - start, 4)
        print("Total time to process: ", elapsed)

        with open(path_results + "parameters_nn.txt", "a") as file:
            for item in clf.best_params_:
                file.write(" %s %s" % (item, clf.best_params_[item]))
            file.write(";%s\n" % (elapsed))

        return(clf.best_params_)

    def TestNN(self, X_train, Y_train, X_test, Y_test, act, hid, alpha, batch, learn, solver, itera):
        """
        This function trains and tests the SVM method on the dataset and returns training and testing AUC and the ROC values
        Input:
            X_train: training set
            Y_train: training set labels
            X_test: testing set
            Y_test: testing set
            act:  activation function for the hidden layers
            hid: size of hidden layers, array like [10,10]
            alpha: regularization parameters
            batch: minibatch size
            learn: learning rate
            solver: Adam or SGD
            itera: iteration of cross validation, used to write down the models 
        Output: 
            result: array with training [0] and testing error[1]
            fpr_svm: false positive rate, ROc values
            tpr_svm: true positive rate, ROc values
            clf: trained classifier, can be used to combine with others, like superlearner
            probas: probabilities from predict function
            brier: brier score
        """

        clf_nn = MLPClassifier(solver=solver, activation=act, hidden_layer_sizes=hid, alpha=alpha,
                               batch_size=batch, learning_rate_init=learn, max_iter=5000)

        clf_nn = clf_nn.fit(X_train, Y_train)
        preds = clf_nn.predict(X_test)
        probas = clf_nn.predict_proba(X_test)[:, 1]

        # name=('Models/RFC'+str(itera)+'.pkl')
        # joblib.dump(clf,name)

        self.measure.calculate(Y_test, preds, probas)

        return clf_nn
