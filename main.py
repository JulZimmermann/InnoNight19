import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from DataDumyCreator import DataDumyCreator
from DataFactorizer import DataFactorizer

df = DataFactorizer()
dc = DataDumyCreator()

#load testdata from CSV
testdata = pd.read_csv("https://raw.githubusercontent.com/ChristophRaab/MLH/master/dataset.csv")

#df.factorizeDataSave(testdata)
testdata = dc.createDummies(testdata, "deposit", "yes")

# create X and y
X = testdata
y = X["deposit"]

#deleting y from X
del X['deposit']

#splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#defining rf as the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000)

clf.fit(X_train, y_train)

pred_test = clf.predict(X_test)

print("Mean absolute error:", metrics.mean_absolute_error(y_test, pred_test))
print("Accuracy:", metrics.accuracy_score(y_test, pred_test))
print("F1: ", metrics.f1_score(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
