import pandas as pd
from sklearn.svm import SVC
#from sklearn.model_selection import train_test_split
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from sklearn.metrics import accuracy_score
import pickle

trn_X1 = pd.read_pickle("trn_X.pickle")
val_X1 = pd.read_pickle("val_X.pickle")
trn_y1 = pd.read_pickle("trn_y.pickle")
val_y1 = pd.read_pickle("val_y.pickle")

trn_X = trn_X1.values
val_X = val_X1.values
trn_y = trn_y1.values
val_y = val_y1.values
clf = SVC()
print(clf.fit(trn_X, trn_y))
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)
predict_y = clf.predict(val_X)
##  Mean accuracy on test data and labels
print("score = ",clf.score(val_X,val_y))
#('score = ', 0.8117977528089888)
print(pd.crosstab(val_y, predict_y, rownames=['Actual Authors'], colnames=['Predicted Authors']))
#Predicted Authors     0    1    2
#Actual Authors                   
#0                  1369   80  151
#1                   200  816   86
#2                   181   39  994