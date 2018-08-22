import pandas as pd 
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from collections import OrderedDict


import pickle

trn_X = pd.read_pickle("trn_X.pickle")
val_X = pd.read_pickle("val_X.pickle")
trn_y = pd.read_pickle("trn_y.pickle")
val_y = pd.read_pickle("val_y.pickle")

# Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#from sklearn.datasets import make_classification
#from sklearn.model_selection import train_test_split

### RandomForestClassifier
error_rate = []
clf = RandomForestClassifier(random_state=0, max_depth = 12, oob_score = True)
min_estimators = 20
max_estimators = 100
for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(trn_X, trn_y)
        oob_error = 1 - clf.oob_score_
        print oob_error, clf.oob_score
        error_rate.append((i, oob_error))
#clf = RandomForestClassifier(random_state=0, max_depth = 12)
#print('RandomForestClassifier')
#print(clf.fit(trn_X, trn_y))
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#            oob_score=False, random_state=0, verbose=0, warm_start=False)
#for elem1, elem2 in error_rate:
xs, ys = zip(*error_rate)
plt.plot(xs, ys)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()
pred_author = clf.predict(val_X)
##  Mean accuracy on test data and labels
print("Score",clf.score(val_X, val_y))
#('Score', 0.8314606741573034)

# Create a confusion matrix
print(pd.crosstab(val_y, pred_author, rownames=['Actual Authors'], colnames=['Predicted Authors']))
cm = confusion_matrix(val_y, pred_author)

print(cm)

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
pyplot.grid()
plt.show()


#Predicted Authors     0    1    2
#Actual Authors                   
#0                  1368   87  145
#1                   147  890   65
#2                   162   54  998
# View Feature Importance
# View a list of the features and their importance scores
print(list(zip(trn_X, clf.feature_importances_)))
x = zip(trn_X, clf.feature_importances_)
print(x.sort(key=lambda x: x[1], reverse = True))
