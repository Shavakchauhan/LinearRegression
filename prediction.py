import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

LE=LabelEncoder()
x[:,3]=LE.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

LR=LinearRegression()
LR.fit(x_train,y_train)
y_pred=LR.predict(x_test)
print('predicted',y_pred)
print('actual',y_test)

