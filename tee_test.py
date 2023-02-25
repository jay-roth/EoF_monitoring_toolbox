# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 21:16:15 2023

@author: jay
"""
import os
import pandas as pd
import numpy as np
import scipy as sp

    
#### USE CODE BELOW FOR MULTIPLE scripts ######################################
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

## shapiro and t-test alphas
shap_p, test_p = [0.05, 0.05]
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

## columns for output dataframe
stat_cols = ['prj','cstaid', 'xstaid', 'param', 'shap_cbl_p', 'shap_xbl_p', 
             'shap_cxt_p',  'shap_xxt_p', 'cbl-cxt_t', 'cbl-xbl_t',
             'cxt-xxt_t', 'xbl-xxt_t']

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
            flo_dqi = dqi_cols[0]
            par_dqi = dqi_cols[par_cols.index(par)]
            if par in ['peak_discharge_cfs', 'runoff_in']:
                cols = ['date', phas_ind_col, par, flo_dqi]
            else: 
                cols = ['date', phas_ind_col, par, flo_dqi, par_dqi]
            
            ctrl_data = df[(df[proj_col]==prj) &\
                           (df['project_mon_stat_id']==ctrl_staid) &\
                           (df[par]>=0)][cols]
            ctrl_data.sort_values('date')

            ## get xment site data
            xmnt_data = df[(df[proj_col]==prj) &\
                           (df['project_mon_stat_id']==xmnt_staid) &\
                           (df[par]>=0)][cols]
            xmnt_data.sort_values('date')
            
            if metric in ['LOAD', 'YIELD'] and\
                            par not in ['peak_discharge_cfs', 'runoff_in']:
                ctrl_data[par_dqi] = (ctrl_data[flo_dqi]*ctrl_data[par_dqi])
                xmnt_data[par_dqi] = (xmnt_data[flo_dqi]*xmnt_data[par_dqi])

            ctrl_data = ctrl_data[(ctrl_data[par_dqi])>=min_par_dqi]
            xmnt_data = xmnt_data[(xmnt_data[par_dqi])>=min_par_dqi]
            ## get list of paired event dates for treatment and control staid
            ## determine where 
            if paired_obs == True:
                for d in xmnt_data['date']:
                    if ctrl_data[ctrl_data.date==d].shape[0]<1:
                        xmnt_data = xmnt_data.drop(xmnt_data[xmnt_data.date==d].index)
                for d in ctrl_data['date']:
                    if xmnt_data[xmnt_data.date==d].shape[0]<1:
                        ctrl_data = ctrl_data.drop(ctrl_data[ctrl_data.date==d].index)        
        
            ## query paired points for control and treatement for baseline and
            ## treatment phases
            ctrl_pts_bl = np.log10(ctrl_data[(ctrl_data[phas_ind_col]==0) &\
                                           (ctrl_data[par]>=min_val)][par].values)

            ctrl_pts_xt = np.log10(ctrl_data[(ctrl_data[phas_ind_col]==1) &\
                                           (ctrl_data[par]>=min_val)][par].values)
            
            xmnt_pts_bl = np.log10(xmnt_data[(xmnt_data[phas_ind_col]==0) &\
                                           (xmnt_data[par]>=min_val)][par].values)
                
            xmnt_pts_xt = np.log10(xmnt_data[(xmnt_data[phas_ind_col]==1) &\
                                           (xmnt_data[par]>=min_val)][par].values)
            
#### USE ABOVE CODE FOR MULTIPLE SCRIPTS#######################################  
            ## run shapiro test for normality
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
                
            ## now lets run t-test where shapiro was good 
            ## to see if we have differences
            'cbl-cxt_t', 'cbl-xbl_t', 'xbl-xxt_t', 'cxt-xxt_t'
            
            if cbl_p > shap_p and cxt_p > shap_p:
                ## two tailed, these should be the same
                cbl_cxt_t = sp.stats.ttest_ind(ctrl_pts_bl, ctrl_pts_xt,
                                               equal_var=False, 
                                               alternative='two-sided')[1]
            else:
                cbl_cxt_t = ""
            if cbl_p > shap_p and xbl_p > shap_p:
                ## two tailed, these should be the same
                cbl_xbl_t = sp.stats.ttest_ind(ctrl_pts_bl, xmnt_pts_bl,
                                               equal_var=False, 
                                               alternative='two-sided')[1]
            else:
                cbl_xbl_t = ""   
            if cxt_p > shap_p and xxt_p > shap_p:
                ## one tailed, these should be different
                cxt_xxt_t = sp.stats.ttest_ind(ctrl_pts_xt, xmnt_pts_xt,
                                               equal_var=False, 
                                               alternative='greater')[1]
            else:
                cxt_xxt_t = ""    
            if xbl_p > shap_p and xxt_p > shap_p:
                ## one tailed, these should be different
                xbl_xxt_t = sp.stats.ttest_ind(xmnt_pts_bl, xmnt_pts_xt,
                                               equal_var=False, 
                                               alternative='greater')[1]
            else:
                xbl_xxt_t = ""                
            stat_df = stat_df.append(pd.DataFrame([[prj, ctrl_staid, xmnt_staid,
                                                   par, cbl_p, cxt_p, xbl_p, 
                                                   xxt_p, cbl_cxt_t, cbl_xbl_t,
                                                   cxt_xxt_t, xbl_xxt_t]], columns=stat_cols))