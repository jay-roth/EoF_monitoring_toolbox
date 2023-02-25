# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 10:00:37 2023

@author: Jason.Roth
"""


import os
import pandas as pd
import numpy as np
from datetime import datetime as dt
import scipy as sp


######## INPUT VARIABLES ######################################################
## metric: EMC, LOAD, YIELD
metric = 'YIELD'

## name of file containing paired events of control and treatment site data
data_file ="AR_focus_data_{0}_final.csv".format(metric)

## minimum event dqi to plot
min_evt_dqi = 1

min_par_dqi = 1

#### names of columns for use in logic and labeling
## site indicator column, control or treatment: n>0 is treatment
site_ind_col= 'site_indicator'

## phase indicator column, baseline or treatment: n>0 treatment
phas_ind_col = 'phase_indicator'

## date of event
date_col = 'date'

## project name column
proj_col = 'project_title'

## columns of observations to make paired plots for
par_cols = ['peak_discharge_cfs', 'runoff_in', 'NH4_mg.l', 'nitrate_nitrite_mg.l', 
             'TN_mg.l', 'dissolvedp_mg.l', 'TP_mg.l', 'TSS_mg.l']

dqi_cols = ['runoff_dqi', 'runoff_dqi', 'NH4_dqi', 'nitrate_nitrite_dqi', 
             'TN_dqi', 'dissolvedP_dqi', 'TP_dqi', 'TSS_dqi']

## min value of data to plot, just to make sure nan and zeros are out.
min_val = 1e-6

## use paired observation or all observations
paired_obs = True

######## BEGIN CODE ###########################################################
## get the current working dir
cwd = os.getcwd()

## import the data into a data 
#encoding='windows-1252'
df = pd.read_csv(os.path.join(cwd, data_file))

## convert date strings in "date" column to date time values
df['date'] = pd.to_datetime(df['date'])

df = df[df.event_dqi>=min_evt_dqi]

## get a list of unique projects
prjs = df[proj_col].unique()

prjs.sort()

stat_cols = ['prj','cstaid', 'xstaid', 'param', 'ctrl_bl',
             'ctrl_xt', 'xmnt_bl', 'xmnt_xt']
## make a dataframe to hold the results in 
stat_df = pd.DataFrame(columns=stat_cols)

## loop over all projects
for prj in prjs:
    print(prj)
    
    ## determine if there's a control station
    ctrl_staids = df[(df[proj_col]==prj) & (df[site_ind_col]==0)]\
                    ['project_mon_stat_id']
    if ctrl_staids.shape[0] > 0:
    ## get control staid for this project
        ctrl_staid = df[(df[proj_col]==prj) & (df[site_ind_col]==0)]\
                        ['project_mon_stat_id'].values[0]
    ## if no control staid set to ''
    else:
        ctrl_staid = ''
        
    ## get all treatment staids for this project
    xmnt_staids = df[(df[proj_col]==prj) & (df[site_ind_col]>0)]\
                    ['project_mon_stat_id'].unique()
    ## sort xmnt_staids
    xmnt_staids.sort()
    
    ## loop over all treatment sites for this project and make a figure with
    ## subplots for each parameter
    for xmnt_staid in xmnt_staids:
        
        ## loop over all parameters to plot
        for par in par_cols:
            ## extract the date and parameter cols to dataframe for control 
            ## site data
            ctrl_data = df[(df[proj_col]==prj) &\
                           (df['project_mon_stat_id']==ctrl_staid) &\
                           (df[par]>=0)][['date', phas_ind_col, par]]
            ctrl_data.sort_values('date')

            ## get xment site data
            xmnt_data = df[(df[proj_col]==prj) &\
                           (df['project_mon_stat_id']==xmnt_staid) &\
                           (df[par]>=0)][['date', phas_ind_col, par]]
            xmnt_data.sort_values('date')

            if paired_obs == True:
                ## get list of paired event dates for treatment and control staid
                ## determine where 
                for d in xmnt_data['date']:
                    if ctrl_data[ctrl_data.date==d].shape[0]<1:
                        xmnt_data = xmnt_data.drop(xmnt_data[xmnt_data.date==d].index)
    
                for d in ctrl_data['date']:
                    if xmnt_data[xmnt_data.date==d].shape[0]<1:
                        ctrl_data = ctrl_data.drop(ctrl_data[ctrl_data.date==d].index)        
            
            ctrl_pts_bl = np.log10(ctrl_data[(ctrl_data[phas_ind_col]==0) &\
                                           (ctrl_data[par]>=min_val)][par].values)

            ctrl_pts_xt = np.log10(ctrl_data[(ctrl_data[phas_ind_col]==1) &\
                                           (ctrl_data[par]>=min_val)][par].values)
            
            xmnt_pts_bl = np.log10(xmnt_data[(xmnt_data[phas_ind_col]==0) &\
                                           (xmnt_data[par]>=min_val)][par].values)
                
            xmnt_pts_xt = np.log10(xmnt_data[(xmnt_data[phas_ind_col]==1) &\
                                           (xmnt_data[par]>=min_val)][par].values)
                
            ## run shapiro test
            if ctrl_pts_bl.shape[0] > 2:
                cbl_p = sp.stats.shapiro(ctrl_pts_bl)[1]
            else:
                cbl_p = 0.0
                
            if ctrl_pts_xt.shape[0] > 2:
                cxt_p = sp.stats.shapiro(ctrl_pts_xt)[1]
            else:
                cxt_p = 0.0
                
            if xmnt_pts_bl.shape[0] > 2:
                xbl_p = sp.stats.shapiro(xmnt_pts_bl)[1]
            else:
                xbl_p = 0.0
            
            if xmnt_pts_xt.shape[0] > 2:
                xxt_p = sp.stats.shapiro(xmnt_pts_xt)[1]
            else:
                xxt_p = 0.0
            
            stat_df = stat_df.append(pd.DataFrame([[prj, ctrl_staid, xmnt_staid,
                                                   par, cbl_p, cxt_p, xbl_p, 
                                                   xxt_p]], columns=stat_cols))