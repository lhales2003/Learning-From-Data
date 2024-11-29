import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    q1 = q1 - (1.5 * iqr)
    q3 = q3 + (1.5 * iqr)
    
    return df.loc[(df[col] > q1) & (df[col] < q3)]

vehicles = pd.read_csv("vehicles.csv")
df = pd.DataFrame(vehicles)

print(df.head())
print(df.shape)
print(df.info())

df = df.drop(columns=["id", "url", "region", "region_url", "model", "VIN", "paint_color", "image_url", "description", "county", "lat", "long", "posting_date"])
df = df.dropna()

print(df.shape)
print(df.info())
print(df.isnull().sum())

print(df.dtypes)

df["manufacturer"] = df["manufacturer"].astype("category")
df["condition"] = df["condition"].astype("category")
df["cylinders"] = df["cylinders"].astype("category")
df["fuel"] = df["fuel"].astype("category")
df["title_status"] = df["title_status"].astype("category")
df["transmission"] = df["transmission"].astype("category")
df["drive"] = df["drive"].astype("category")
df["size"] = df["size"].astype("category")
df["type"] = df["type"].astype("category")
df["state"] = df["state"].astype("category")

# df["age"] = 2024 - df["year"]
# df = df.drop(columns=["year"])

print(df.dtypes)

print(df.describe().T)

df = remove_outliers(df, "odometer")
df = remove_outliers(df, "price")
df = df.loc[(df["price"] > 500)]

unique_counts = df.nunique()
print(unique_counts)

print(df.describe().T)

print(df.isnull().sum())
# one_hot_encoder = OneHotEncoder(sparse_output=False)

# one_hot_encoded = one_hot_encoder.fit_transform(df[["manufacturer", "condition", "cylinders", "fuel", "title_status", "transmission", "drive", "size", "type", "state"]])
# one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(["manufacturer", "condition", "cylinders", "fuel", "title_status", "transmission", "drive", "size", "type", "state"]))
# df = pd.concat([df, one_hot_df], axis=1).drop(columns=["manufacturer", "condition", "cylinders", "fuel", "title_status", "transmission", "drive", "size", "type", "state"])
label_encoder = LabelEncoder()

# one_hot_encoded = one_hot_encoder.fit_transform(df[["transmission"]])
# one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(["transmission"]))
# df = pd.concat([df, one_hot_df], axis=1).drop(columns=["transmission"])


# df["manufacturer"] = label_encoder.fit_transform(df["manufacturer"])
# # df["condition"] = label_encoder.fit_transform(df["condition"])
# # df["cylinders"] = label_encoder.fit_transform(df["cylinders"])
# # df["fuel"] = label_encoder.fit_transform(df["fuel"])
# # df["title_status"] = label_encoder.fit_transform(df["title_status"])
# # df["transmission"] = label_encoder.fit_transform(df["transmission"])
# # df["drive"] = label_encoder.fit_transform(df["drive"])
# # df["size"] = label_encoder.fit_transform(df["size"])
# df["type"] = label_encoder.fit_transform(df["type"])
# df["state"] = label_encoder.fit_transform(df["state"])
df = pd.get_dummies(df)


# df = pd.get_dummies(df)
# print(df.columns)


print(df.isnull().sum())
print(df.head())
print(df.shape)

X = df.drop(columns =["price"])
y = np.log1p(df["price"])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regression = Ridge(alpha=1)
regression.fit(X_train, y_train)
score = regression.score(X_test, y_test)
print(score)

y_pred = np.expm1(regression.predict(X_test))
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