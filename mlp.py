import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, classification_report, confusion_matrix

X_train, X_test, y_train, y_test = preprocessing.main()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPRegressor(hidden_layer_sizes=(64, 32))

mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(r2)