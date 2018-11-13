# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:34:37 2018

@author: Varun
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
folder = "E:/Acads/9th sem/ZS/GOT_ZS_LATEST/GOT_ZS_LATEST/"
df = pd.read_csv(folder+"train.csv")
df.dropna(inplace = True)
core = df.corr()
lastcol = core.iloc[:,-1]
impcol = lastcol[np.abs(lastcol)>0.1]
df = df[impcol.index]

Xtr, Xts, ytr, yts = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size = 0.3)

reg = LinearRegression()
reg.fit(Xtr,ytr)
ypred = reg.predict(Xts)
MAE = sum(np.abs(ypred-yts))/len(yts)

dft = pd.read_csv(folder + "test.csv") 
solid = dft.iloc[:,0:1]
dft = dft[impcol.index[:-1]]
pred = reg.predict(dft)

solid["bestSoldierPerc"] = pred
solid.to_csv(folder+"output.csv", index= False)


