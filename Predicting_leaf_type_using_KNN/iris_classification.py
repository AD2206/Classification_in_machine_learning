import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('iris.csv')
X=dataset.iloc[:,:-1].values

y = dataset['species'].map({'setosa':0,'versicolor':1,'virginica':2}).values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)
sepal_length=float(input("Enter the length of sepal:"))
sepal_width=float(input("Enter the width of sepal:"))
petal_length=float(input("Enter the length of petal:"))
petal_width=float(input("Enter the width of the petal:"))
print(classifier.predict(sc.transform([[sepal_length,sepal_width,petal_length,petal_width]]))) 
prediction = classifier.predict(sc.transform([[sepal_length,sepal_width,petal_length,petal_width]]))

label_map = {
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
}

print("Predicted flower:", label_map[prediction[0]])
#predicting the test set results
y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
#making the confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score 
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))
