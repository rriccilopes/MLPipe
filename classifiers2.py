def get_SVC():
    from sklearn.svm import SVC

    grid_param = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [1, 2, 3, 4, 5, 6],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'class_weight': ['balanced', {0: 1, 1: 5}, {0: 1, 1: 7}, {0: 1, 1: 10}, {0: 1, 1: 13}, {0: 1, 1: 15}]
        }
    return SVC(probability=True), grid_param

def get_NN():
    from sklearn.neural_network import MLPClassifier

    grid_param = {
        'activation': ['relu', 'logistic', 'tanh'],
        'hidden_layer_sizes': [[4, 4], [4, 8, 4], [50, 25], [50, 25, 10], [70, 40, 20], [70, 30], [50, 30, 20, 10]],
        'alpha': [0.1, 0.01, 0.001, 0.0001],
        'batch_size': [8, 16, 32, 64],
        'learning_rate_init': [0.01, 0.001],
        'solver': ['adam', 'lbfgs']
    }
    return MLPClassifier(), grid_param

def get_RFC():
    from sklearn.ensemble import RandomForestClassifier

    grid_param = {
        'n_estimators': [10, 20, 50, 100, 400, 800, 1200, 1600, 2000],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_depth': [None, 3, 5, 7, 9, 11, 13, 20, 50],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 4, 6, 8, 10, 20],
        'min_samples_leaf': [1, 3, 5, 7, 9],
        'class_weight': ['balanced', {0: 1, 1: 5}, {0: 1, 1: 7}, {0: 1, 1: 10}, {0: 1, 1: 13}, {0: 1, 1: 15}]
        }
    return RandomForestClassifier(), grid_param

def get_LR():
    from sklearn.linear_model import LogisticRegression

    grid_param = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
        'class_weight': ['balanced', {0: 1, 1: 5}, {0: 1, 1: 7}, {0: 1, 1: 10}, {0: 1, 1: 13}, {0: 1, 1: 15}]
    }
    return LogisticRegression(), grid_param

def get_xgboost():
    from xgboost import XGBClassifier
    
    grid_param = {
        'min_child_weight': [1, 5, 10, 15],
        'gamma': [0, 1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [4, 7, 10, 15],
        'scale_pos_weight': [5, 7, 10, 13, 15],
        'max_delta_step': [0, 1, 5, 10],
        'n_estimators': [30, 50, 100, 200],
    }
    return XGBClassifier(), grid_param