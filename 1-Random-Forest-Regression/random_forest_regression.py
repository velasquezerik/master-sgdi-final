"""
Asignatura:
    Sistemas de Gestión de Datos y de la Información.
Presentación Final:
    Random Forest Regression
Autor(es):
    Erik Manuel Velásquez Rodríguez
"""
# Importing the libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
print(dataset)
X = dataset.iloc[:, 1:2].values
print(X)
y = dataset.iloc[:, 2].values
print(y)

# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict([[6.5]])
print(y_pred)
