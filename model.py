import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

X_train, X_test, y_train, y_test = preprocessing.main()
lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
print("Coeffecients:", lin_reg.coef_)
print("Intercept:", lin_reg.intercept_)
print("R2:", lin_reg.score(X_test, y_test))
y_pred = np.expm1(lin_reg.predict(X_test))

df_test = X_test.copy()
df_test["price"] = y_test
df_test["pred"] = y_pred

fig, (ax1, ax2) = plt.subplots(1, 2,  figsize=(10, 5))

ax1.scatter(df_test["age"], df_test["price"], color="blue")
ax2.scatter(df_test["odometer"], df_test["price"], color="green")
ax1.plot(df_test["age"], df_test["pred"], color="red")
ax2.plot(df_test["odometer"], df_test["pred"], color="red")

ax1.set_xlabel("Year")
ax1.set_ylabel("Price")
ax2.set_xlabel("Mileage")
ax2.set_ylabel("Price")

plt.tight_layout()
plt.show()