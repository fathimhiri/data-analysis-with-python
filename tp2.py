

import pandas as pd
from sklearn.model_selection import train_test_split
data=pd.read_csv(r"D:\TEK UP 2eme\trait analytique avec python\diabetes.csv")
data
data.info()
data.describe()
label=data['Outcome']
df=data.drop(['Outcome'],axis=1)
xtrain, xtest, ytrain, ytest = train_test_split(df,label,test_size=0.33, random_state=42)
label.value_counts()

from sklearn import svm
m = svm.SVC(kernel='linear', C=1)
m.fit(xtrain,ytrain)
pr=m.predict(xtest)



from sklearn.metrics import accuracy_score
acc =accuracy_score(ytest,pr )
print(acc)


pr1=m.predict(xtrain)
acc_train=accuracy_score(ytrain, pr1)
print(acc_train)


from sklearn.metrics import classification_report
print(classification_report(ytest,pr ))
#inbalance data


from sklearn.metrics import confusion_matrix
CM = confusion_matrix(ytest,pr)
print(CM)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
# CM1=pd.DataFrame(CM)
# print(CM1)
sns.heatmap(pd.DataFrame(CM), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')





from sklearn.metrics import roc_curve, auc
fp, tp, thresholds = roc_curve(ytest, pr, pos_label=1)
print(fp,tp)
AUC=auc(fp, tp)*100
print(AUC)

 

"""roc curve"""
import matplotlib.pyplot as plt
plt.plot(fp, tp, color='blue',label = 'AUC = %0.2f' % AUC)
plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % AUC)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#complexité
model=svm.SVC(kernel="linear",C=1)
import time
debut = time.time()
model.fit(xtrain,ytrain)
fin=time.time() - debut

print(fin)