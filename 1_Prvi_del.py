# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 23:09:49 2021

@author: Hmelj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import ndtri

pot = "kibergrad.csv"
data = pd.read_csv(pot)
lastnosti = data.columns
#print(data.head())

dohodki = data.iloc[:,[3]] #stolpec dohodkov


q1 = np.percentile(dohodki, 25) #Q1
q3 = np.percentile(dohodki, 75) #Q3 

povprecje = int(np.mean(dohodki))
std = int(np.std(dohodki, ddof=1))

m = len(dohodki)
print("q1 je: ", q1)

print("q3 je: ", q3)

sirina = 2*(q3-q1)/np.cbrt(m)




iqr = q3 - q1
spodnja_meja = q1 - 3/2*iqr
zgornja_meja = q3 + 3/2*iqr
print("Spodnja meja osamelcev je: ", spodnja_meja)
print("Zgornja meja osamelcev je: ", zgornja_meja)
#ni osamelcev

print("sirina je: ", sirina) #zaokrozimo na 400
sirina = 2000
print("popravljena sirina je: ", sirina)
zacetek =  (int(dohodki.min())//sirina)*sirina
konec = (int(dohodki.max())//sirina+1)*sirina

print("std je: ", std)
print("povprecje je: ", povprecje)


print("zacetek je: ", zacetek)
print("konec je: ", konec)

dohodki = dohodki.values.tolist()

flat_list = []
for sublist in dohodki:
    for item in sublist:
        flat_list.append(item)
dohodki = flat_list


"""histogram dohodkov"""
plt.figure()
plt.hist(dohodki, bins=int(((konec - zacetek)//sirina )), range=(zacetek, konec))
plt.title("Histogram dohodkov")
plt.show()

"""primerjava histograma z normalno porazdelitvijo"""
x = np.linspace(zacetek, konec, m)
plt.figure()
plt.hist(dohodki, bins=int(((konec - zacetek)//sirina )), range=(zacetek, konec), density=True) 
plt.plot(x, stats.norm.pdf(x, povprecje, std))
plt.title("Primerjava z normalno porazdelitvijo")
plt.show()



"""komulativni histogram"""
st = int(((konec - zacetek)//sirina ))
res = stats.cumfreq(dohodki, numbins=st, defaultreallimits=(zacetek, konec)) #sesteje komulativno po intervalih
res = res[0] 
res = np.insert(res, 0, 0., axis=0)
res = res/res[-1] #normiramo

x = np.linspace(zacetek,konec, st +1)
y_cdf = stats.norm.cdf(x, povprecje, std)

plt.figure()
plt.plot(x, res)
plt.title("Komulativna porazdelitvena funkcija")
plt.show()


"""primerjava z normalno komulativno"""
plt.figure()
plt.plot(x, res)
plt.plot(x, stats.norm.cdf(x, povprecje, std))
plt.title("Primerjava z normalno komulativno funkcijo")
plt.show()




"""Q-Q"""

urejeno = np.sort(dohodki) #uredimo po vrsti

#normalno porazdelitev razdelimo na n+1 delov. 
delcki = np.arange(1,m+1)/(m+1) #range ne vkluci zadnjega

#izracunamo teoreticne vrednosti porazdelitve
teoreticne_vrednosti = ndtri(delcki)


#normaliziramo nase vrednosti
norm_podatki = (urejeno - povprecje)/std
"""q-q"""
plt.figure()
plt.plot(norm_podatki, teoreticne_vrednosti) #scatter da v tocke
plt.title("Q-Q grafikon")
plt.show()

plt.figure()
plt.plot(norm_podatki, teoreticne_vrednosti) #scatter da v tocke
plt.plot(teoreticne_vrednosti, teoreticne_vrednosti) #primerjava z normalno porazdelitvijo
plt.title("Q-Q primerjava z normalno porazdelitvijo")
plt.show()