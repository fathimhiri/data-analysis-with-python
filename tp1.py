import matplotlib.pyplot as plt
plt.figure(1)
Nb=[10,20,30,40,50,60,70,80,90,100]
T=[2,4,6,8,10,12,14,16,18,20]

plt.plot(T,Nb)
plt.xlim([2,16])
plt.ylim([10,80])


#q3
plt.title("Courbe représentative du pièces conformes fabriquées en fonction du temps")
plt.ylabel('piece conforme')
plt.xlabel("temp")

#q5
Nb1=[10,20,30,40,50,60]
Nb2=[30,40,50,60,70,80]
Nb3=[40,50,60,70,80,90]
T1=[10,12,14,16,18,20]
plt.figure(2)
plt.plot(T1,Nb1, '--r',linewidth=7)
plt.plot(T1,Nb2,'*b',linewidth=12)
plt.plot(T1,Nb3, '+g',linewidth=10)
plt.text(15,40,"Nb optimal")


plt.figure(3)
plt.subplot(121)
plt.plot(T1,Nb1, '--r',linewidth=7)
plt.plot(T1,Nb2,'*b',linewidth=12)
plt.plot(T1,Nb3, '+g',linewidth=10)
plt.subplot(122)
plt.plot(T1,Nb1, '--r')
plt.plot(T1,Nb2,'*b')
plt.plot(T1,Nb3, '+g')


plt.figure(4)
plt.subplot(221)
plt.plot(T1,Nb1, '--r',linewidth=7)
plt.plot(T1,Nb2,'*b',linewidth=12)
plt.plot(T1,Nb3, '+g',linewidth=10)
plt.subplot(224)
plt.plot(T1,Nb1, '--r')
plt.plot(T1,Nb2,'*b')
plt.plot(T1,Nb3, '+g')

plt.figure(5)
plt.subplot(211)
plt.plot(T1,Nb1, '--r',linewidth=7)
plt.plot(T1,Nb2,'*b',linewidth=12)
plt.plot(T1,Nb3, '+g',linewidth=10)
plt.subplot(212)
plt.plot(T1,Nb1, '--r')
plt.plot(T1,Nb2,'*b')
plt.plot(T1,Nb3, '+g')


#q11
plt.figure
explode=[0.1,0.2,0.5,0.13]
cat=['cat1','cat2','cat3','cat4']
N =[5000, 26000 , 21400, 12000]
plt.pie(N, labels = cat,explode=explode)



