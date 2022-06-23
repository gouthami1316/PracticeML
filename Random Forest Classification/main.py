import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

from sklearn.ensemble import RandomForestClassifier

ml = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=10, min_samples_split=50)
ml.fit(x_tr, y_tr)
print("Training Score - ", ml.score(x_tr, y_tr))
print("Testing Score - ", ml.score(x_te, y_te))

ypred = ml.predict(x_te)
print(ypred)

from sklearn.metrics import confusion_matrix, classification_report

cm_ml = confusion_matrix(y_te, ypred)
print(cm_ml)
print(classification_report(y_te, ypred))

from sklearn import tree
fn = x_tr.columns
print(fn)
cn = ['0', '1', '2', '3']
print(cn)

plt.figure(figsize=(20, 15))
tree.plot_tree(ml.estimators_[0], feature_names=fn, class_names=cn, filled=True)
plt.show()

