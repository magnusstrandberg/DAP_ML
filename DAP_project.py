import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

#Read the samples
X = np.asarray(pd.read_csv("Data.csv"))
print(X)

#Read the classes
Y = np.ravel(np.asarray(pd.read_csv("train_labels.csv")))
print(Y)

#Multiclass learning using OvO
clf = OneVsOneClassifier(LinearSVC(random_state=0))
print(clf.fit(X,Y).predict(X))

