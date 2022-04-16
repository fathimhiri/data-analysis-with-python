import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

date =pd.date_range('1/1/2017',periods=2000 , freq='D')
type(date)
val1=np.random.randn(2000) #distr normal de mean = 0 and std = 1
#val1 = np.random.normal(3,np.sqrt(2),2000)

#rand vs randn vs normal
#Q3

series1 = pd.Series(data=val1,index=date)

plt.title("Courbe représentative des val en fc de date")
plt.ylabel('temp')
plt.xlabel("date")
#plt.plot(date,val1)

plt.figure(1)
plt.plot(series1)
#series1.plot()

cumule1 = series1.cumsum()
plt.figure(2)
plt.title("c1")
#plt.ylim([0,100])
plt.plot(cumule1)

# partie II

val2  = np.random.normal(3,np.sqrt(2),(2000,4))
type(val2)
df2=pd.DataFrame(data=val2,index = date , columns =['F1','F2','F3','F4'])



#q3 q4
cumul2=df2.cumsum()


cumul2.plot()
# cumul.plot ilaya les legendes des courbe etc, mais si plot(cumul3 on trouve pas legende)


val3 =np.random.randn(2000,4)
df=pd.DataFrame(val3,date,columns =['F1','F2','F3','F4'])
c3=df.cumsum()

c3.plot()

# partir III



mat=np.random.randn(2000,2)
df3=pd.DataFrame(data=mat,columns=['feature2','Feature3'])
cumule3=df3.cumsum()


df3['Feature1']=pd.Series(list(range(len(cumule3))))
df3.plot('Feature1','feature2')
df3.plot('Feature1','Feature3')




#part 4



df4=pd.DataFrame(np.random.randn(5,4),columns=['Fet1','Fet2','Fet3','Fet4'])
df4.plot.bar()

df4.hist(color='r',bins=20,alpha=0.5)

# part5

#m=np.random.normal(3,2,(100,4))
m=2*np.random.randn(100,4)+3
df5=pd.DataFrame(m,columns=['Fetu1','Fetu2','Fetu3','Fetu4'])

df5.plot('Fetu3','Fetu4',color="darkgreen")
df5.plot('Fetu1','Fetu2',color="darkblue")

plt.figure(10)
plt.plot(df5['Fetu3'],df5['Fetu4'],"+g")
plt.plot(df5['Fetu1'],df5['Fetu2'],"*b")
plt.legend(["Fetu4", "Fetu2"])


#oubien
fig1 = df5.plot.scatter(x=['Fetu3'],y=['Fetu4'],color="darkblue",label="group1")
#puiz
df5.plot.scatter(x=['Fetu1'],y=['Fetu2'],color="darkgreen",label="group2",ax=fig1)
series=pd.Series(3*np.random.rand(4),index=['a','b','c','d'] , name='series')
series.plot.pie(figsize=(6,6))





