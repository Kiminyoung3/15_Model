import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

header = ['preg', 'plas', 'pres', 'skin', 'est', 'mass', 'pedi', 'age', 'class']
data= pd.read_csv("./data/pima-indians-diabetes.data.csv")

array = data.values
X=array[:, 0:8]
Y=array[:,8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X=scaler.fit_transform(X)

model = LogisticRegression()

fold = KFold(n_splits=10, shuffle=True)

acc=cross_val_score(model, rescaled_X, Y, cv=fold, scoring='accuracy')
print(acc)

print(sum(acc)/len(acc))

