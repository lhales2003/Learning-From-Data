import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = preprocessing.main()
param_grid = { 
	'n_estimators': [25, 50, 100, 150, 200],  
	'max_depth': [15, 20, 25], 
	'max_leaf_nodes': [6, 9, 12], 
} 

grid_search = GridSearchCV(RandomForestRegressor(), param_grid=param_grid)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_)