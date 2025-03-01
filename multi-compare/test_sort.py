# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:47:48 2023

@author: Jason.Roth
@email: jason.roth@usda.gov
@position:Env.Engr, WQQT
@agency: USDA-NRCS

Description:

"""
import pandas as pd
import os
from scipy import stats
import numpy as np

def get_sig_difs(df, a, sd_val=1, nsd_val=0):
    """
    @author: Jason.Roth
    
    Description:
    assigns significant or non-significant designator to a dataframe 
    of p-values produced from a multiple-comparason analysis such as 
    Tukey HSD or Dunns

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    ## convert the 
    
    
    for c in df.columns:
        for i in df.index:
            ## convert all values above the diagnal to null
            if c > i:
                df.loc[i,c] = ""
            else:
                if df.loc[i,c] > a:
                    df.loc[i,c] = 1
                else:    
                    df.loc[i,c] = 0
    return df

def get_comp_grps(df):
    """
    assigns a group based on statistical differences amongst cohort groups
    
    Steps
        1.) 
        2.) count number of "nsd" across columns
    
    Parameters
    ----------
    df_pval : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ## labels for distinct groups
    grp_lab = ['a', 'b', 'c', 'd', 'e']

    ## make a copy of the dataframe to hold individual groupings per column
    df_grps = df.copy()
    
    ## set all values to null
    df_grps.loc[:,:]=''
    
    ## make a list to keep counts in
    grp_num = -1
    prv_cnt = 0
    
    for c in df_grps.columns:
        if sum(df.loc[:,c]==1) >= prv_cnt:
            grp_num += 1
            prv_cnt = sum(df.loc[:,c]==1)
            df_grps.loc[df.loc[:,c]==1,c] = grp_lab[grp_num]
            
    for c in df_grps.columns[1:]:
        df_grps.loc[:,0]+=df_grps.loc[:,c]
            
    return df_grps[0]

cwd=os.getcwd()

alpha = 0.05

## define names as we did in the ancova script
orig_nam = ['cb', 'tb', 'ct', 'tt']

## add data import and integrity check here
orig_dat = [[1, 2, 2, 4],
             [4, 6, 8, 10],
             [20, 21, 25, 24],
             [36, 34.2, 31.2, 35]]

orig_mu = [sum(i)/len(i) for i in orig_dat]


# get the means 
sort_mu = [sum(i)/len(i) for i in orig_dat]

## sort the data by means high to low. To use the ARS spreadsheet method.
sort_mu.sort(reverse=True)

sort_dat = []

sort_nam = []

## resort each data set from greatest to least mean
for mu in sort_mu:
    sort_dat.append(orig_dat[orig_mu.index(mu)])
    
## add parametric check and perform transform if needed/possible. 
## Run an anova if norma, if not normal run kruskal-wallace,

## run anova or k-w
r = stats.f_oneway(*sort_dat)

if r[1] < alpha:
    
    ## determine what groups are statistically significantly different.
    g = stats.tukey_hsd(*sort_dat)
    
    mc_pvals = pd.DataFrame(g.pvalue)
    
    mc_difs = get_sig_difs(mc_pvals.copy(), alpha)  
    
    mc_grps = get_comp_grps(mc_difs.copy())
    
    mc_grps = pd.Series(mc_grps.values,
                        [orig_nam[orig_mu.index(i)] for i in sort_mu])
    
    grps = pd.DataFrame([mc_grps[orig_nam].tolist()], 
                           columns=orig_nam)


## custom sort algo don't delete
#while len(x) > 0:
#    for i in range(len(x)):
#        if i == 0:
#            m = x[0]
#        else:
#            if x[i]<m:
#                m = x[i]
#    x.remove(m)
#    y.append(m)
    
