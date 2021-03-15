import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
import itertools


#Read the samples
X = np.asarray(pd.read_csv("Data.csv"))
#print(X)

#Read the classes
Y = np.ravel(np.asarray(pd.read_csv("train_labels.csv")))
#print(Y)

test = np.asarray(pd.read_csv("Test.csv"))

#Multiclass learning using OvO and linearSVC
#OvOclf = OneVsOneClassifier(LinearSVC(random_state=0))
#a = np.reshape(np.asarray(list(range(1,len(test)+1))),(6544,1))
#b = np.reshape(np.asarray(OvOclf.fit(X,Y).predict(test)),(6544,1))
#OvOclf.fit(X,Y).predict(X)
#print('Multiclass, OvO, linear decision function: ' +str(clf.decision_function(X)))
#print('Multiclass, OvO, linear score: '+str(OvOclf.score(X,Y)))

#logloss
clf = CalibratedClassifierCV(LinearSVC(random_state=0,class_weight='balanced',C=3.5,max_iter=2000),cv=5)
a = np.reshape(np.asarray(list(range(1,len(test)+1))),(6544,1))
b = np.reshape(np.asarray(clf.fit(X,Y).predict(test)),(6544,1))
c = np.reshape(np.asarray(list(range(1,len(test)+1))),(6544,1))
d = np.reshape(np.asarray(clf.fit(X,Y).predict_proba(test)),(6544,10))
print(d)
clf.fit(X,Y).predict(X)
e = clf.predict_proba(X)
print('Multiclass, OvO, linear logloss for train set: ' +str(clf.predict_proba(test)))
print('logloss: ' + str(log_loss(Y, e)))
print('Score: '+str(clf.score(X,Y)))


#Multiclass learning using OvR and linearSVC
#clf = OneVsRestClassifier(LinearSVC(random_state=0))
#a = np.reshape(np.asarray(list(range(1,len(test)+1))),(6544,1))
#b = np.reshape(np.asarray(clf.fit(X,Y).predict(test)),(6544,1))
#clf.fit(X,Y).predict(X)
#print('Multiclass, OvR, linear decision function: ' +str(clf.decision_function(X)))
#print('Score: '+str(clf.score(X,Y)))

#Multiclass learning using OvO and polynomial SVC
#clf = OneVsOneClassifier(LinearSVC(random_state=0,class_weight='balanced'))
#a = np.reshape(np.asarray(list(range(1,len(test)+1))),(6544,1))
#b = np.reshape(np.asarray(clf.fit(X,Y).predict(test)),(6544,1))
#clf.fit(X,Y).predict(X)
#print('Multiclass, OvO, poly decision function: ' +str(clf.decision_function(X)))
#print('Multiclass, OvO, weighted poly score: '+str(clf.score(X,Y)))


#Deep tree
#clf = DecisionTreeClassifier(random_state=0)
#a = np.reshape(np.asarray(list(range(1,len(test)+1))),(6544,1))
#b = np.reshape(np.asarray(clf.fit(X,Y).predict(test)),(6544,1))
#print('Deep tree: '+str(clf.score(X,Y)))
#print('Cross val:'+str(cross_val_score(clf,X,Y,cv=5)))



#Support Vector Machine with degree 1
#clf = svm.SVC(gamma='scale',kernel='poly',degree=2,decision_function_shape='ovo')
#clf.fit(X,Y)
#clf.predict(test)
#print('logloss:' +str(clf.predict_proba()))
#print(clf.score(X,Y))
#print(cross_val_score(clf,X,Y,cv=5))


#result = np.hstack((a,np.asarray(clf.fit(X,Y).predict(test))))
#result = np.concatenate((a,np.asarray(clf.fit(X,Y).predict(test))), 1)


result = np.append(a,b,axis=1)
#print(result)
#np.savetxt("reslut.csv",np.array(clf.fit(X,Y).predict(test)).astype(int),delimiter=",")
with open("accuracy.csv", "wb") as f:
    f.write(b'Sample_id,Sample_label\n')
    np.savetxt(f, result.astype(int), fmt='%i', delimiter=",")

result2 = np.append(c,d.astype(float),axis=1)
#print(result)
#np.savetxt("reslut.csv",np.array(clf.fit(X,Y).predict(test)).astype(int),delimiter=",")
with open("logloss.csv", "wb") as f:
    f.write(b'Sample_id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9,Class_10\n')
    np.savetxt(f, result2, delimiter=",")


# Split the data into a training set and a test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

#Multiclass learning using OvO and linearSVC
OvOclf = OneVsOneClassifier(LinearSVC(random_state=0))
a = np.reshape(np.asarray(list(range(1,len(test)+1))),(6544,1))
b = np.reshape(np.asarray(OvOclf.fit(X_train,Y_train).predict(test)),(6544,1))
OvOclf.fit(X,Y).predict(X)
y_pred = OvOclf.fit(X_train,Y_train).predict(X_test)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cnf_matrix = confusion_matrix(Y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,normalize=True, classes= [1,2,3,4,5,6,7,8,9,10],
                      title='Confusion matrix, with normalization')

plt.show()
