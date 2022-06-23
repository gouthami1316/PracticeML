import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("mobile_price_range_data (2).csv")
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.dtypes)

x = df.iloc[:, :-1]
y = df.iloc[:, -1]
from sklearn.model_selection import train_test_split
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25)
print(x_tr.shape)
print(x_te.shape)
print(y_tr.shape)
print(y_te.shape)


from sklearn.neighbors import KNeighborsClassifier

ml = KNeighborsClassifier(n_neighbors=100)
ml.fit(x_tr, y_tr)

ypred = ml.predict(x_te)
print(ypred)
print("Training score", ml.score(x_tr, y_tr))
print("Testing score", ml.score(x_te, y_te))


from sklearn.metrics import confusion_matrix, classification_report

cm_ml = confusion_matrix(y_te, ypred)
print(cm_ml)
print(classification_report(y_te, ypred))

