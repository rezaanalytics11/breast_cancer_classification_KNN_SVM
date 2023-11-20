import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

x=data.data
y=data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#KNN Classification

def knn_(x1,y1,x2,y2):
 model0=KNeighborsClassifier(n_neighbors=5)#accuracy:0.92
 model0.fit(x_train,y_train)
 y_pred0=model0.predict(x_test)
 score0=model0.score(x_test,y_test)
 print(model0.predict([[7.760e+00, 2.454e+01, 4.792e+01, 1.810e+02 ,5.263e-02 ,4.362e-02 ,0.000e+00,
 0.000e+00 ,1.587e-01, 5.884e-02, 3.857e-01, 1.428e+00 ,2.548e+00 ,1.915e+01,
 7.189e-03 ,4.660e-03 ,0.000e+00 ,0.000e+00 ,2.676e-02, 2.783e-03, 9.456e+00,
 3.037e+01 ,5.916e+01 ,2.686e+02 ,8.996e-02 ,6.444e-02, 0.000e+00 ,0.000e+00,
 2.871e-01, 7.039e-02]]))
 print(score0)

#SVM Classification
def svm_(x1,y1,x2,y2):
 model1=svm.SVC(kernel='linear')#accuracy:0.97
 model1=svm.SVC(kernel='linear',C=2)#accuracy:0.976
 #model1=svm.SVC(kernel='poly')#accuracy:0.88
 #model1=svm.SVC(kernel='sigmoid')#accuracy:0.4
 #model1=svm.SVC(kernel='rbf')#accuracy:0.91
 model1.fit(x_train,y_train)
 y_pred1=model1.predict(x_test)
 score1=accuracy_score(y_test,y_pred1)

 print(score1)


knn_(x_train,x_test,y_train,y_test)

svm_(x_train,x_test,y_train,y_test)