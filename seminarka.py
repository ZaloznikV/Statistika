# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 17:59:06 2021

@author: Vito Zaloznik

Prva naloga seminarske naloge pri predmetu Statistika na FMF.


a)Narišite histogram dohodkov vseh družin v Kibergradu. Pri tem dohodke razdelite v enako široke razrede. 
Širino posameznega razreda določite v skladu s Freedman–Diaconisovim pravilom,
Kjer sta q1/4 in q3/4 prvi in tretji kvartil, n pa je število enot. To vrednost nato
smiselno zaokrožite na število oblike k · 10^r, kjer je k ∈ {1, 2, 5} in r ∈ Z.

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

n = dohodki.size #tevilo vrstic oz podatkov
q1 = dohodki.describe().loc['25%'][0] #1. kvartil

q3 = dohodki.describe().loc['75%'][0] #3. kvartil

zacetek = int(dohodki.describe().loc['min'][0]) #zacetek histograma
konec = int(dohodki.describe().loc['max'][0])

sirina = 2*(q3 - q1)/np.cbrt(n)
sirina = int(sirina//1000*1000)

mu = dohodki.describe().loc["mean"][0]
std = dohodki.describe().loc["std"][0] 

zacetek = (zacetek // sirina) * sirina #da se zacnejo na celo "lepo" stevilo
konec = (konec // sirina + 1) * sirina


x = np.linspace(zacetek, konec, (konec - zacetek)//sirina+1)



k = 0
y = []
while (k< (konec - zacetek)//sirina ): #st intervalov dolzine sirina
    a= int(dohodki[(dohodki >= zacetek+ k*sirina) & ( dohodki < zacetek + (k+1)*sirina )].count())
    y.append(a)
    k = k + 1



# for i in range(x.size-2):
#     if (x[i+2] - x[i+1]) != (x[i+1] - x[i]):
#         print(i)
    
dohodki_np = dohodki.to_numpy()
#normirano
plt.figure()
plt.hist(dohodki_np, bins=((konec - zacetek)//sirina ), range=(zacetek, konec), density=True)
plt.plot(x, stats.norm.pdf(x, mu, std))
plt.show()


"""
Narišite kumulativno porazdelitveno funkcijo porazdelitve dohodkov družin v
Kibergradu in primerjajte s kumulativno porazdelitveno funkcijo ustrezne normalne porazdelitve.
"""

komulativno = np.cumsum(y)

X = []
for i in range(x.size-1):
    sredina = (x[i+1] + x[i])/2
    X.append(sredina)

y_cdf = stats.norm.cdf(X, mu, std)

komulativno = komulativno/komulativno[-1] #normiramo podatke 

plt.figure()   
plt.plot(X, komulativno)
plt.plot(X, y_cdf)
plt.show()

"""
d) Narišite še primerjalni kvantilni (Q–Q) grafikon, ki porazdelitev dohodkov
družin v Kibergradu primerja z normalno porazdelitvijo (
"""

dohodki_np = np.hstack(dohodki_np)
urejen = np.sort(dohodki_np, axis=None ) #uredimo array 

#normalno porazdelitev razdelimo na n+1 delov. 

delcki = np.arange(1,n+1)/(n+1) #range ne vkluci zadnjega

#izracunamo teoreticne vrednosti porazdelitve
teoreticne_vrednosti = ndtri(delcki)


#normaliziramo nase vrednosti
norm_podatki = (urejen - mu)/std

#narisemo graf 
plt.figure()
plt.plot(norm_podatki, teoreticne_vrednosti) #scatter da v tocke
plt.plot(teoreticne_vrednosti, teoreticne_vrednosti) #primerjava z normalno porazdelitvijo
plt.show()



"""
Vzemite 1000 enostavnih slučajnih vzorcev velikosti 400 in narišite histogram
vzorčnih povprečij dohodkov družin.
"""

for i in range(1000):
    vzorec = dohodki.sample(n=400)
    vzorec_mean = dohodki.describe().loc['mean'][0] #povprecje vzorca
    #naredimo nov array in dodajamo notri