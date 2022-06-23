import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("mobile_price_range_data (2).csv")
print(df.head())

x = df.iloc[:, :-1]
y = df.iloc[:, -1]
print(x.head())
print(y.head())

from sklearn.model_selection import train_test_split

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25)
print(x_tr.shape)
print(x_te.shape)
print(y_tr.shape)
print(y_te.shape)

from sklearn.linear_model import LogisticRegression

ml = LogisticRegression()
ml.fit(x_tr, y_tr)
ypred = ml.predict(x_te)
print(ypred)

res = pd.DataFrame({'y_te': y_te, 'y_pred': ypred})
print(res.head())

m = ml.coef_
c = ml.intercept_
print(m)
print(c)


def logistc(x, m, c):
    sig = 1 / (1 + np.exp(-(m * x + c)))
    print(sig)

from sklearn.metrics import confusion_matrix, classification_report
cm_m1 = confusion_matrix(y_te,ypred)
print(cm_m1)
print(classification_report(y_te,ypred))