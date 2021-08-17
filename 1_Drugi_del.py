# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 21:40:42 2021

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




















vzorci_povprecja = []
n = 400 #elikost vzorca
m = 1000 #st vzorcev
for i in range(m):
    vzorec = dohodki.sample(n=400)
    vzorec_mean = vzorec.describe().loc['mean'][0] #povprecje vzorca
    vzorci_povprecja.append(vzorec_mean)

    
 
q1 = np.percentile(vzorci_povprecja, 25) #Q1
q3 = np.percentile(vzorci_povprecja, 75) #Q3 

print("q1 je: ", q1)

print("q3 je: ", q3)

sirina = 2*(q3-q1)/np.cbrt(m)

zacetek =  (min(vzorci_povprecja)//400)*400
konec = (max(vzorci_povprecja)//400+1)*400
print("sirina je: ", sirina) #zaokrozimo na 400

sirina = 400

"""prvi graf"""
plt.figure()
plt.hist(vzorci_povprecja, bins=int(((konec - zacetek)//sirina )), range=(zacetek, konec))
plt.title("Histogram vzorčnih povprečij")
plt.show()


print("Popravljena sirina je: ", sirina)
povprecje = np.mean(vzorci_povprecja)
print("Povprecje vzorcnih pvprecij je: ", povprecje)

std = np.std(dohodki, ddof=1) 

vzorec = dohodki.sample(n=400) #??????????????????????????????????????????????????????????????????????????????????
std = np.std(vzorec, ddof=1)


povprecje_vseh = np.mean(dohodki)
print("std je: ", std)
print("povprecje vseh je: ", povprecje_vseh)

N = len(dohodki) #st vseh dohodkov
SE = int(np.sqrt((N - n)/(N)* (std*std / n)))
print("Standardna napaka za vorec velikosti 400 je: ", SE)
"""histogram z dodano nomrlano porazdelitvijo"""
x = np.linspace(zacetek, konec, n)
plt.figure()
plt.hist(vzorci_povprecja, bins=int(((konec - zacetek)//sirina )), range=(zacetek, konec), density=True) 
plt.plot(x, stats.norm.pdf(x, povprecje, SE))
plt.title("Primerjava z normalno porazdelitvijo")
plt.show()



#f)
"""komulativni histogram"""
st = int(((konec - zacetek)//sirina ))
res = stats.cumfreq(vzorci_povprecja, numbins=st, defaultreallimits=(zacetek, konec)) #sesteje komulativno po intervalih
res = res[0] 
res = np.insert(res, 0, 0., axis=0)
res = res/res[-1] #normiramo

x = np.linspace(zacetek,konec, st +1)
y_cdf = stats.norm.cdf(x, povprecje, SE)

plt.figure()
plt.plot(x, res)
plt.title("Komulativna porazdelitvena funkcija")
plt.show()

"""primerjava z normalno komulativno"""
plt.figure()
plt.plot(x, res)
plt.plot(x, stats.norm.cdf(x, povprecje, SE))
plt.title("Primerjava z normalno komulativno funkcijo")
plt.show()


"""Q-Q"""

urejeno = np.sort(vzorci_povprecja) #uredimo po vrsti

#normalno porazdelitev razdelimo na n+1 delov. 
delcki = np.arange(1,m+1)/(m+1) #range ne vkluci zadnjega

#izracunamo teoreticne vrednosti porazdelitve
teoreticne_vrednosti = ndtri(delcki)


#normaliziramo nase vrednosti
norm_podatki = (urejeno - povprecje)/SE
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

















