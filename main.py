import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from DataFactorizer import DataFactorizer

df = DataFactorizer()

#load testdata from CSV
testdata = pd.read_csv("https://raw.githubusercontent.com/ChristophRaab/MLH/master/dataset.csv")

df.factorizeDataSave(testdata)

# create X and y
X = testdata
y = X["deposit"]

#deleting y from X
del X['deposit']

#splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#defining rf as the RandomForestClassifier
clf = RandomForestClassifier()

clf.fit(X_train, y_train)

pred_test = clf.predict(X_test)
print("Mean absolute error:", metrics.mean_absolute_error(y_test, pred_test))
print("Accuracy:", metrics.accuracy_score(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))

