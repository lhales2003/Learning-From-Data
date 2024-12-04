import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = preprocessing.main()

regression = LinearRegression()
regression.fit(X_train, y_train)
score = regression.score(X_test, y_test)
print(score)

y_pred = np.expm1(regression.predict(X_test))
df_test = X_test.copy()
df_test["price"] = y_test
df_test["pred"] = y_pred

fig, (ax1, ax2) = plt.subplots(1, 2,  figsize=(10, 5))

ax1.scatter(df_test["age"], np.expm1(df_test["price"]), color="blue")
ax2.scatter(df_test["odometer"], np.expm1(df_test["price"]), color="green")
ax1.plot(df_test["age"], df_test["pred"], color="red")
ax2.plot(df_test["odometer"], df_test["pred"], color="red")

ax1.set_xlabel("Age")
ax1.set_ylabel("Price")
ax2.set_xlabel("Mileage")
ax2.set_ylabel("Price")

plt.show()