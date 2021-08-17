# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 16:21:41 2021

@author: Hmeljaro
"""

import pandas as pd
import numpy as np
import itertools
from itertools import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("Pulz.csv")

data.columns
my_data = data.iloc[:,[0,1,3,6]]

kombinacije = [] #vse mnozice podatkov ki jih bomo uporabili v modelu
stolpci = [0,1, 2, 3]
for L in range(0, len(stolpci)+1):
    for subset in itertools.combinations(stolpci, L):
        kombinacije.append(list(subset))
del kombinacije[0] #izbrisemo prazno mnozico


n_modelov = len(kombinacije)
AIC = []
RSS = []
reg = LinearRegression()
Y = data.PULZ1 #ocenjujemo ta podatek
n = len(Y)
#za vsako podmnozico vhodnih parametrov bomo evaluirali model
for kombinacija in kombinacije:
    m = len(kombinacija) #st parametrov
    reg.fit(my_data.iloc[: , kombinacija], Y) #regresija
    #print(reg.coef_) #koeficienti regresije
    
    Y_ocenjeno = reg.predict(my_data.iloc[: , kombinacija])
    mse = mean_squared_error(Y, Y_ocenjeno)
    # rss = N* MSE
    
    rss = n * mse 
    RSS.append(rss)
    aic = 2 * m + n * np.log(rss) 
    AIC.append(aic) #seznam AIC-jev za posamezni model
    
    
min_index = AIC.index(min(AIC)) #index z parametri ki dajo najmanjši AIC
optimalni_podatki = my_data.iloc[:,kombinacije[6]]
rss_od_optimalnih = RSS[min_index]


optimalni_AIC = min(AIC)
print("Najmanjši AIC je: ", optimalni_AIC, "ki ima RSS: ", rss_od_optimalnih)
optimalni_podatki = my_data.iloc[:,kombinacije[6]]
print("Najboljši podatki so: ", optimalni_podatki)

kajenje = data.iloc[:,[4]]
alkohol = data.iloc[:,[5]]

dodatno_kadi = pd.concat([optimalni_podatki, kajenje], axis=1)


dodatno_alkohol = pd.concat([optimalni_podatki, alkohol], axis=1)


#dimenzije vseh podatkov enake tako d lahko m in n parameter ostaneta za vse
m = dodatno_alkohol.shape[1] #st parametrov
n = dodatno_alkohol.shape[0] #velikost vzorca

print("Velikost vozrca je: ", n)

AIC_2 = [optimalni_AIC]
dic = {}
dic[optimalni_AIC] = optimalni_podatki



Y = data.PULZ1 #ocenjujemo ta podatek  
reg.fit(dodatno_alkohol, Y) #regresija
Y_ocenjeno = reg.predict(dodatno_alkohol)
mse = mean_squared_error(Y, Y_ocenjeno)
    # rss = N* MSE
rss = n * mse 
print("RSS dodatno alkohol je: ", rss)


reg.fit(dodatno_kadi, Y) #regresija
Y_ocenjeno = reg.predict(dodatno_kadi)
mse = mean_squared_error(Y, Y_ocenjeno)
    # rss = N* MSE
rss = n * mse 

print("RSS dodatno kadi je: ", rss)



