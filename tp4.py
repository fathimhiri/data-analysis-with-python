import pandas as pd
import numpy as np



data=pd.read_csv(r"D:\TEK UP 2eme\trait analytique avec python\house-prices.csv")
data

# size=data.SqFt.values  // array (line) (array is ligne, whereas series is a vector , colonne)

size=data.SqFt  # series
size

y = data.Price
#q2

#transform series to array vector(2nd methode)
x1=size.values.reshape(-1,1)
y1=y.values.reshape(-1,1) # array colonne vecoteur
y2=y.values # array 1 D line


#q3
from sklearn.linear_model  import LinearRegression
model = LinearRegression()
model.fit(x1,y1)

a = model.coef_
b=model.intercept_

label= model.predict(x1)

#q5
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y1,label)
import math
print('RMSE', math.sqrt(mse))

pd.DataFrame(y1).describe()

#♣ on remarque que lerreur rms est proche de la valeur min
#de y1 (prices) donc on a une grande erruere

from sklearn.metrics import explained_variance_score
ev= explained_variance_score(y1,label)
print("explained variance score ", ev)

import matplotlib.pyplot as plt
plt.figure(1)
plt.scatter(x1,y1,color="g")
plt.plot(x1,label,'r')
plt.title("linear regression")
plt.xlabel("size")
plt.ylabel("price")
print(model.predict([[1000]]))


#------------------------------------------exercice 2
import pandas as pd
import numpy as np

dataset=pd.read_csv(r"D:\TEK UP 2eme\trait analytique avec python\house-prices.csv")


price=dataset['Price']
Y=np.array(price).reshape(-1,1)


Data=dataset.drop(['Price','Brick','Neighborhood','Home'],axis=1)

 
from sklearn.preprocessing import LabelEncoder
X1=dataset['Brick']
le = LabelEncoder()
X1New = le.fit_transform(X1) 
X1New = X1New.reshape(-1,1)

 

X2=dataset['Neighborhood']
le1 = LabelEncoder()
X2New = le1.fit_transform(X2) 
X2New = X2New.reshape(-1,1)

 

X=np.concatenate((Data,X1New,X2New),axis=1)







from sklearn.linear_model  import LinearRegression
model = LinearRegression()
model.fit(X,Y)
a = model.coef_
b=model.intercept_
#q5
label1= model.predict(X)

#rmse
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,label1)
import math
print('RMSE', math.sqrt(mse))

pd.DataFrame(Y).describe()



from sklearn.metrics import explained_variance_score
ev= explained_variance_score(Y,label1)
print("explained variance score ", ev)
# => ev =0.85 donc modele performant mais n'est pas excelet car < à 0.9

#q7
# we compare ev for simpe model and multiple model of the lienar regression
# we found 0.3 for the siple and 0.85 for the multiple regression
#♥ so multiple is way mor ebetter
























