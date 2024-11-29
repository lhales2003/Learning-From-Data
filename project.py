import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    q1 = q1 - (1.5 * iqr)
    q3 = q3 + (1.5 * iqr)
    
    return df.loc[(df[col] > q1) & (df[col] < q3)]

vehicles = pd.read_csv("vehicles.csv")
df = pd.DataFrame(vehicles)



df = df.drop(columns=["id", "url", "region", "region_url", "manufacturer", "model", "fuel", "transmission", "VIN", "size", "paint_color", "image_url", "description", "county", "lat", "long", "posting_date"])
df = df.dropna(subset=["year", "odometer"])
df = remove_outliers(df, "odometer")
df = remove_outliers(df, "price")

df = df[pd.to_numeric(df["cylinders"].str.replace("cylinders", ""), errors='coerce').notna()]
df["cylinders"] = df["cylinders"].str.replace("cylinders", "").astype(float)

encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(df[["state", "title_status", "drive", "condition", "type"]])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(["state", "title_status", "drive", "condition", "type"]))

df = pd.concat([df, one_hot_df], axis=1)
df = df.drop(columns=["state", "title_status", "drive", "condition", "type"], axis=1)

print(df.info())
 
X = df.drop(columns=["price"])
X = X.dropna()
y = df["price"]
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
print("Coeffecients:", lin_reg.coef_)
print("Intercept:", lin_reg.intercept_)
print("R2:", lin_reg.score(X_test, y_test))
y_pred = lin_reg.predict(X_test)

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