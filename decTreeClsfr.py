import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

df=pd.read_csv(r"D:\TEK UP 2eme\trait analytique avec python\iris.csv")
df.shape
x=df.drop("variety",axis=1)
y=df["variety"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


model1= DecisionTreeClassifier(random_state=0,criterion="gini")
model1.fit(X_train, y_train)
print( "score with gini = ", model1.score( X_test,y_test))

print("------------------------------------")

model0= DecisionTreeClassifier(random_state=0,criterion="entropy")
model0.fit(X_train, y_train)
print( "score with entropy= ", model0.score( X_test,y_test))

pred = model0.predict(X_test)
cm = confusion_matrix(y_test, pred, labels=model0.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model0.classes_)
disp.plot()

plt.show()



