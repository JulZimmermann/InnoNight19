import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

testdata = pd.read_csv("https://raw.githubusercontent.com/ChristophRaab/MLH/master/dataset.csv")

(testdata.head())

X = testdata

X["job"] = pd.factorize(X.job)[0]
X["marital"] = pd.factorize(X.marital)[0]
X["education"] = pd.factorize(X.education)[0]
X["default"] = pd.factorize(X.default)[0]
X["housing"] = pd.factorize(X.housing)[0]
X["loan"] = pd.factorize(X.loan)[0]
X["contact"] = pd.factorize(X.contact)[0]
X["month"] = pd.factorize(X.month)[0]
X["day"] = pd.factorize(X.day)[0]
X["poutcome"] = pd.factorize(X.poutcome)[0]
X["deposit"] = pd.factorize(X.deposit)[0]

y = X["deposit"]

del X['deposit']
(X.head())

#splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#defining rf as the RandomForestClassifier
clf = RandomForestClassifier()

clf.fit(X_train, y_train)

pred_test = clf.predict(X_test)
print("Mean absolute error:", metrics.mean_absolute_error(y_test, pred_test))
print("Accuracy:", metrics.accuracy_score(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
