from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_excel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score

df = read_excel() #sample data
X = df.values[:, 0:18]
y = df.values[:, 18]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

rf = RandomForestClassifier(random_state=0,n_estimators=10, criterion='gini', max_depth=None,min_samples_split=2, min_samples_leaf=1,max_features=None, oob_score=False)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(rf_auc)
tree_predict = cross_val_predict(rf, X_test, y_test, cv=10, method='predict_proba')
print(precision_score(y_test,tree_predict))
print(accuracy_score(y_test, tree_predict))
def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestClassifier(
            random_state=0,
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999), # float
            max_depth=int(max_depth)
            ),
    X, y, cv=10,scoring='roc_auc'
    ).mean()
    return val
rf_bo = BayesianOptimization(
        rf_cv,
        {'n_estimators': (10, 250),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999),
        'max_depth': (5, 15)},
        random_state=0
    )
rf_bo.maximize()
print(rf_bo.max)
rf1 = RandomForestClassifier(n_estimators=52, max_depth=9.2, min_samples_split=17,
                             max_features=0.14,oob_score=False, random_state=0)
df1 = read_excel() #sample data
X1 = df.values[:, 0:16]
y1 = df.values[:, 16]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, random_state=0)
rf1.fit(X1_train,y1_train)
fpr1, tpr1, thresholds1 = roc_curve(y,
                                 rf.predict_proba(X_train)[:, 1])
plt.plot(fpr1, tpr1, label="AUC(Unoptimized)= ")# AUC value
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc=4)
plt.savefig(r'E:\ROC.png', dpi=500, bbox_inches='tight') #image storage

data = read_excel() #all points data
predict_data = rf1.predict_proba(data.values[:, 0:16])[:, 1]

df2 = pd.DataFrame(predict_data)
df2.to_excel() #output data
