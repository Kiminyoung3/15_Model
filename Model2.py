import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

header = ['preg', 'plas', 'pres', 'skin', 'est', 'mass', 'pedi', 'age', 'class']
data= pd.read_csv("./data/pima-indians-diabetes.data.csv")

array = data.values
X=array[:, 0:8]
Y=array[:,8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X=scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(rescaled_X, Y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(Y_test, y_pred)
print(acc)

