import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.metrics import classification_report
# plot learning curve
from numpy import loadtxt
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
from sklearn.metrics import accuracy_score
import pickle

trn_X = pd.read_pickle("trn_X.pickle")
val_X = pd.read_pickle("val_X.pickle")
trn_y = pd.read_pickle("trn_y.pickle")
val_y = pd.read_pickle("val_y.pickle")
val_df = pd.read_pickle("val_df1.pickle")
trn_X.head()

clf = XGBClassifier(max_depth=6)
# eval_set = [(val_X, val_y)]
eval_set = [(trn_X, trn_y), (val_X, val_y)]

# clf.fit(trn_X, trn_y, eval_metric="error", eval_set=eval_set, verbose=True)
clf.fit(trn_X, trn_y, eval_metric=["merror", "mlogloss"], eval_set=eval_set, verbose=True)


print(clf)
# print(clf.fit(trn_X, trn_y))
#XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
#       n_jobs=1, nthread=None, objective='multi:softprob', random_state=0,
#       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#       silent=True, subsample=1)
predict_y = clf.predict(val_X)
predictions = [round(value) for value in predict_y]
# evaluate predictions
accuracy = accuracy_score(val_y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# retrieve performance metrics
results = clf.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.grid()
pyplot.show()
# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.grid()
pyplot.show()

##  Mean accuracy on test data and labels
# print("score = ",accuracy_score(val_y,predict_y))
# Max_depth = 3:
#score =  0.876659856997
# Max_depth = 4:
#score =  0.882277834525
# Max_depth = 5:
#score =  0.885597548519
# Max_depth = 6:
# score =  0.886108273749
# Max_depth = 10:
# score = 0.879468845761
print(pd.crosstab(val_y, predict_y, rownames=['Actual Authors'], colnames=['Predicted Authors']))
target_names = ['EAP', 'HPL', 'MWS']
print(classification_report(val_y, predict_y, target_names=target_names))
# Accuracy: 88.61%
# Predicted Authors     0    1     2
# Actual Authors                    
# 0                  1456   51    93
# 1                   100  953    49
# 2                   103   50  1061