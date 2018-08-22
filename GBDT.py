import pandas as pd 
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from sklearn.ensemble import GradientBoostingClassifier
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

clf = GradientBoostingClassifier()
print(clf.fit(trn_X, trn_y))
#GradientBoostingClassifier(criterion='friedman_mse', init=None,
#              learning_rate=0.1, loss='deviance', max_depth=3,
#              max_features=None, max_leaf_nodes=None,
#              min_impurity_decrease=0.0, min_impurity_split=None,
#              min_samples_leaf=1, min_samples_split=2,
#              min_weight_fraction_leaf=0.0, n_estimators=100,
#              presort='auto', random_state=None, subsample=1.0, verbose=0,
#              warm_start=False)
predict_y = clf.predict(val_X)
##  Mean accuracy on test data and labels
print("score = ",clf.score(val_X,val_y))
#('score = ', 0.84346271705822262)
print(pd.crosstab(val_y, predict_y, rownames=['Actual Authors'], colnames=['Predicted Authors']))
#Predicted Authors     0    1     2
#Actual Authors                    
#0                  1380   84   136
#1                   129  910    63
#2                   146   55  1013
print(list(zip(trn_X, clf.feature_importances_)))
