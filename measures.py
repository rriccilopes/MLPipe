"""
This code is for the EEg project, it contains feature extraction and data pre-processing

The filter fit in python is different from matlab, it is giving me different values from marjolein
@author: laramos
"""
import numpy as np
from sklearn.metrics import confusion_matrix, brier_score_loss, roc_curve, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, auc
from scipy import interp
import matplotlib.pyplot as plt

class Measures:

    def __init__(self, run=False):
        self.run = run
        self.auc = []
        self.brier = []
        self.sensitivity = []
        self.specifity = []
        self.mean_tpr = 0.0
        self.mean_fpr = np.linspace(0, 1, 100)
        self.cm = []
        self.accuracy = []
        self.f1 = []
        self.precision = []
        self.recall = []
        self.fpr = []
        self.tpr = []
        self.tprs = []
        self.frac_pos = []
        self.feat_imp = []

    def calculate(self, y_true, y_pred=None, y_proba=None, class_threshold=0.5):
        if y_pred is None and y_proba is None:
            raise Exception("You must provide 'y_pred' or 'y_proba'")
            
        if y_pred is None and y_proba is not None:
            y_pred = y_proba > class_threshold
        
        self.cm.append(confusion_matrix(y_true, y_pred))
        self.accuracy.append(round(accuracy_score(y_true, y_pred), 2))
        self.f1.append(round(f1_score(y_true, y_pred), 2))
        self.precision.append(round(precision_score(y_true, y_pred), 2))
        self.recall.append(round(recall_score(y_true, y_pred), 2))
        self.sensitivity.append(self.cm[-1][0, 0] / (self.cm[-1][0, 0] + self.cm[-1][1, 0]))
        self.specifity.append(self.cm[-1][1, 1] / (self.cm[-1][1, 1] + self.cm[-1][0, 1]))

        if y_proba is not None:
            self.brier.append(brier_score_loss(y_true, y_proba))
            self.auc.append(round(roc_auc_score(y_true, y_proba), 2))
            
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            self.fpr.append(fpr)
            self.tpr.append(tpr)
            self.tprs.append(interp(self.mean_fpr, fpr, tpr))
            self.tprs[-1][0] = 0.0
            self.mean_tpr = np.mean(self.tprs, axis=0)
            self.mean_tpr[-1] = 1.0
    
    def plot_auc(self):
        
        # Plot each curve
        i = 1
        for fpr, tpr in zip(self.fpr, self.tpr):
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
            i +=1
        
        # Plot mean curve
        mean_auc = auc(self.mean_fpr, self.mean_tpr)
        std_auc = np.std(self.auc)
        plt.plot(self.mean_fpr, self.mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

        # Details
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.show()

    @staticmethod
    def plot_multiple_auc(model_list):
        fpr = np.linspace(0, 1, 100)
        for model in model_list:
            tpr = model.measure.mean_tpr
            roc_auc = auc(fpr, tpr)
            std_auc = np.std(model.measure.auc)
            plt.plot(fpr, tpr, lw=1, label='ROC %s (AUC = %0.2f $\pm$ %0.2f)' % (model.name, roc_auc, std_auc))

        # Details
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.show()