# -*- coding: utf-8 -*-
"""tp nlp text classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ekwCVN2A72tF1x6P8s0Xe1rMHnBaUkKa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow 

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

df=pd.read_csv(r"D:\TEK UP 2eme\trait analytique avec python\spam.csv",encoding='latin-1',delimiter=',')

df
df.info()

"""On remarque que les 3 colonnes 'Unnamed: 2','Unnamed: 3' et 'Unnamed: 4' ont la majorité des valeurs NAN : missing values, donc on va les supprimer ."""

df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

df.info()

df

"""on transforme la colonne v1 en type numerique à l'aide de labelencoder: elle sera notre target"""

le = LabelEncoder()

df['v1'] = le.fit_transform(df['v1'])


x=df.v2
y=df.v1



tok = Tokenizer(num_words=1000)
tok.fit_on_texts(x)
seq = tok.texts_to_sequences(x)

seq_ma=sequence.pad_sequences(seq,maxlen=150,padding='post')



seqDF=pd.DataFrame(seq_ma)


# we make 2 data absed on target because we have unlabeled data set
#here so we gonna use monoclass mode

y=pd.DataFrame(y)
y1=y.loc[y['v1']==1]
y0=y.loc[y['v1']==0]-1


df1=seqDF.loc[y1['v1'].index]
df0=seqDF.loc[y0['v1'].index]




X_train1, X_test1, y_train1, y_test1 = train_test_split(df1, y1, test_size=0.33, random_state=42)

from sklearn import svm
#train on class1 only
model=svm.OneClassSVM(kernel='rbf',gamma=0.1,nu=0.1)
model.fit(X_train1,y_train1)


X_train0, X_test0, y_train0, y_test0 = train_test_split(df0, y0, test_size=0.33, random_state=42)

x_test=np.concatenate([X_test1,X_test0],axis=0)
y_test=np.concatenate([y_test1,y_test0],axis=0)
#pred on both
pred=model.predict(x_test)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, pred)
print("accuracy: ", acc)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, pred)
print(cm)