# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 10:09:36 2023

@author: Jason.Roth
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression as lr
import scipy as sp

var = 'runoff_in'


df = pd.read_csv('test_data.csv')

df['date'] = pd.to_datetime(df['date'])

df = df.dropna()

ct_dat = df[df['site_indicator']==0]

tx_dat = df[df['site_indicator']==1]


## trim down to paired events
## get list of paired event dates for treatment and control staid
## determine where 
for d in tx_dat['date']:
    if ct_dat[ct_dat.date==d].shape[0]<1:
        tx_dat = tx_dat.drop(tx_dat[tx_dat.date==d].index)

for d in ct_dat['date']:
    if tx_dat[tx_dat.date==d].shape[0]<1:
        ct_dat = ct_dat.drop(ct_dat[ct_dat.date==d].index) 
        
fig = plt.figure()
ax = fig.add_subplot()

ct_bl = ct_dat[ct_dat['phase_indicator']==0]['runoff_in'].values
tx_bl = tx_dat[tx_dat['phase_indicator']==0]['runoff_in'].values

ct_tx = ct_dat[ct_dat['phase_indicator']==1]['runoff_in'].values
tx_tx = tx_dat[tx_dat['phase_indicator']==1]['runoff_in'].values

ax.scatter(ct_bl, tx_bl)

ax.scatter(ct_tx, tx_tx)

bl_reg = sp.stats.linregress(ct_bl, tx_bl,)

tx_reg = sp.stats.linregress(ct_tx, tx_tx)

x =[ct_bl.min(),]
ax.plot([ct_bl.min(), ct_bl.max()],
         [ct_bl.min()*bl_reg.slope+bl_reg.intercept, 
         ct_bl.max()*bl_reg.slope+bl_reg.intercept])

ax.plot([ct_tx.min(), ct_tx.max()],
         [ct_tx.min()*tx_reg.slope+tx_reg.intercept, 
         ct_tx.max()*tx_reg.slope+tx_reg.intercept])

sk_reg_bl = lr(fit_intercept=True)

sk_reg_bl.fit(ct_bl, tx_bl)
r2 = sk_reg_bl.score(ct_bl[:,np.newaxis, 0], tx_bl)
## use regression model to predict over the domain of the axes
xt_x = [[ct_bl.min()], [ct_bl.max()]]
xt_y = sk_reg_bl.predict(xt_x)


plt.show()