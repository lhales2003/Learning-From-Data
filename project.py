import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
vehicles = pd.read_csv("vehicles.csv")

df = pd.DataFrame(vehicles)

df_cleaned = df.loc[(df["price"] >= 1000) & (df["price"] <= 5000)]

lin_reg = LinearRegression()

lin_reg.fit(df_cleaned[["year", "odometer"]], df_cleaned["price"])
y_pred = lin_reg.predict(df_cleaned[["year", "odometer"]])

df_cleaned["pred"] = y_pred

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.scatter(df_cleaned["year"], df_cleaned["price"], color="blue")
ax2.scatter(df_cleaned["odometer"], df_cleaned["price"], color="green")

plt.tight_layout()
plt.show()