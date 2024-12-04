import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans

vehicles = pd.read_csv("vehicles.csv")
df = pd.DataFrame(vehicles)
    
label_encoder = LabelEncoder()
# find the average price of a car per state
price_per_state = df.groupby("state", as_index=False)["price"].mean()
print(price_per_state)

# encode the states to prepare them for clustering
price_per_state["encoded_state"] = label_encoder.fit_transform(price_per_state["state"])
clustering_data = price_per_state[["encoded_state", "price"]]

inertia_values = []

cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(clustering_data)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertia_values)
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()