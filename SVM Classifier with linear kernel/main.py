import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets
df = {}
df = np.loadtxt("mobile_price_range_data (2).csv",delimiter=',',dtype=int)
da = pd.read_csv("mobile_price_range_data (2).csv")
print(df)
x = df[:, [0,6]]
y = df[:, -1]
print(y)
print(x[:5], end="\n\n\n")
print(y[:5])

print(x.shape)
print(y.shape)

from sklearn.svm import SVC


ml = SVC(kernel="linear")
ml.fit(x, y)
plot_decision_regions(x, y, clf=ml)
plt.xlabel("battery_power")
plt.ylabel("int_memory")
plt.title("SVM with kernel Linear")
plt.show()
