import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#importing data
ad = pd.read_csv("C:/Users/Shubham/.spyder-py3/datasets_14872_228180_Admission_Predict.csv")
ad.head(10)
#finding missing Value
ad.isnull().sum()
dc = ad.describe()
corr = ad.corr()
sns.pairplot(ad)
# Indpendent and dependent variables
X = ad.iloc[:,1:8]
y = ad.iloc[:,8]
# spliting and fiting model 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, train_size=80, random_state=0)
from sklearn.ensemble import RandomForestRegressor
fcl = RandomForestRegressor(n_estimators=110, random_state=0)
fcl.fit(X_train, y_train)
y_pred=fcl.predict(X_test)
from sklearn import metrics
print('MSE: ',metrics.mean_squared_error(y_test, y_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
plt.scatter(y_test,y_pred)
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=9)
knn.fit( X_train , y_train)
y_pred1=knn.predict(X_test)
print('MSE: ',metrics.mean_squared_error(y_test, y_pred1))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))
from sklearn.tree import DecisionTreeRegressor
dc=DecisionTreeRegressor()
dc.fit(X_train, y_train)
y_pred2 =dc.predict(X_test)
print('MSE: ',metrics.mean_squared_error(y_test, y_pred2))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
plt.scatter(y_test,y_pred2)
plt.scatter(y_test,y_pred1)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred3=lr.predict(X_test)
print('MSE: ',metrics.mean_squared_error(y_test, y_pred3))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, y_pred3)))
plt.scatter(y_test,y_pred3)
from sklearn.ensemble import VotingRegressor
voting_clf_S = VotingRegressor([('lr', lr), ('rf', fcl), ('KNN', knn)])
voting_clf_S.fit(X_train, y_train)
pred =voting_clf_S.predict(X_test)
print('MSE: ',metrics.mean_squared_error(y_test, pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, pred)))
plt.scatter(y_test,pred)
