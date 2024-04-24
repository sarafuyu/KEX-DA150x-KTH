#%% md
# Northstar Prediction Estimation
#%% md
## Test for SVM
#%%
# Load the important packages
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

#Build the model
# TODO: try probability=True
svm = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None)

# Trained the model
svm.fit(X, y)