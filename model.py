import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    q1 = q1 - (1.5 * iqr)
    q3 = q3 + (1.5 * iqr)
    
    return df.loc[(df[col] > q1) & (df[col] < q3)]

vehicles = pd.read_csv("vehicles.csv")

df = pd.DataFrame(vehicles)

print(df.shape)
print(df.head())
print(df.info())

print(df.isnull().sum())

df = df.drop(columns=["id", "url", "region", "region_url", "manufacturer", "model", "condition", "fuel", "title_status", "transmission", "VIN", "drive", "size", "type", "paint_color", "image_url", "description", "county", "state", "lat", "long", "posting_date"])
df = df.dropna()

# no_of_regions = df["region"].value_counts()
# print(no_of_regions)
# df["region"] = df["region"].where(df["region"].isin(df["region"].value_counts()[:50].index.tolist()), "other")

# no_of_manufacturers = df["manufacturer"].value_counts()
# print(no_of_manufacturers)
# df["manufacturer"] = df["manufacturer"].where(df["manufacturer"].isin(df["manufacturer"].value_counts()[:20].index.tolist()), "other")

df = remove_outliers(df, "odometer")
df = remove_outliers(df, "price")

label_encoder = LabelEncoder()
df["cylinders"] = label_encoder.fit_transform(df["cylinders"])

X = df.drop(columns="price")
X = X.dropna()
y = np.log1p(df["price"])
y = y[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
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

ax1.scatter(df_test["year"], df_test["price"], color="blue")
ax2.scatter(df_test["odometer"], df_test["price"], color="green")
ax1.plot(df_test["year"], df_test["pred"], color="red")
ax2.plot(df_test["odometer"], df_test["pred"], color="red")

ax1.set_xlabel("Year")
ax1.set_ylabel("Price")
ax2.set_xlabel("Mileage")
ax2.set_ylabel("Price")

plt.tight_layout()
plt.show()