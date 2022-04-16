import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



df=pd.read_csv(r"D:\TEK UP 2eme\trait analytique avec python\house-prices.csv")
print(df.head)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), ["Brick","Neighborhood"])], remainder='passthrough')
data = columnTransformer.fit_transform(df)
column_name = columnTransformer.get_feature_names()
df =  pd.DataFrame(data, columns= column_name)
df.head(2)





x=df.drop(["Price","SqFt"],axis=1)
y=df["Price"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

regr = RandomForestRegressor(n_estimators = 200, max_features = 'sqrt',
                             max_depth=5, random_state=18)
regr.fit(x_train, y_train)
pred =regr.predict(x_test)

from sklearn.metrics import r2_score
print( "R2 =  ",r2_score(y_test, pred))

import math
from sklearn.metrics import mean_squared_error
print("rmse : ", math.sqrt(mean_squared_error(y_test, pred)))

