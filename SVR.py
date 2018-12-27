import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

# Plotting the SVR result
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x,y,color = 'red')
plt.plot(x, regressor.predict(x))
plt.title('Salary estimator using SVR')
plt.xlabel('Position level')
plt.ylabel('Salary in USD')
plt.show()