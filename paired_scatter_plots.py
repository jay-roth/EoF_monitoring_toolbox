# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 21:16:15 2023

@author: jay
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression as lr
import scipy as sp

def regPval(X, Y, m, b):
    
    pvals = []
    Xssr = np.sum([(i-X.mean())**2 for i in X])
    Yssr = np.sum([(m*(X[Y.index(j)]+b)-Y.mean)**2 for j in Y])
    dfr = X.shape[0]- 2
    se = np.sqrt((1./(X.shape[0]-2.))*Yssr/Xssr)
    for a in [m,b]:
        pvals.append(sp.stats.t.sf(abs(a/se), dfr)*2)
    return pvals


######## INPUT VARIABLES ######################################################
## metric: EMC, LOAD, YIELD
met = 'YIELD'

## name of file containing paired events of control and treatment site data
data_file ="AR_focus_data_{0}_final.csv".format(met)

## minimum event dqi to plot
dqi_min = 1

## label for plot legend
plt_label = "${0}, n:{1:1.0f}, m:{2:1.2f}, r^2:{3:1.2f}$" 

## name of log file
log_file = "log_{0}.txt".format(dt.now().strftime("%m%d%Y-%H%M"))

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
params = {'peak_discharge_cfs':0,
          'runoff_in':1,
          'NH4_mg.l': 1,
          'nitrate:nitrite_mg.l': 1, 
          'TN_mg.l': 1,
          'dissolvedp_mg.l': 1,
          'TP_mg.l':1,
          'TSS_mg.l':1}


######## BEGIN CODE ###########################################################
base_nodata_msg = "project:{0},  control:{1}, treatment:{2}, parameter:{3}, "+\
                    "no baseline data\n"

xmnt_nodata_msg = "project:{0},  control:{1}, treatment:{2}, parameter:{3}, "+\
                    "no treatment data\n"

plot_msg = "project:{0}, control:{1}, treatment:{2}, parameter:{3}, plot created\n"

## get the current working dir
cwd = os.getcwd()

log_file = os.path.join(cwd, log_file)

## import the data into a data 
#encoding='windows-1252'
df = pd.read_csv(os.path.join(cwd, data_file))

## convert date strings in "date" column to date time values
df['date'] = pd.to_datetime(df['date'])

df = df[df.event_dqi>=dqi_min]

## get a list of unique projects
prjs = df[proj_col].unique()
prjs.sort()

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
        
        ## rows to have in subplot
        plt_rows = int((len(params)+1)/2)
        ## row column indexes for subplots
        r = 0
        c = 0
        
        ## ctrl var to determine if a plot is worthy of saving
        plt_chk = 0
        
        ## instantiate figure to put the plots on. Plots will be n x 2 panel.
        fig, axs = plt.subplots(nrows=plt_rows, ncols=2, figsize=[8.5,11], 
                                constrained_layout=True)
        
        ## loop over all parameters to plot
        for par in params.keys():
            ## set the axis to the sublot we want
            ax = axs[r,c]
            ax.set_aspect='equal'
            ax.tick_params(labelsize='xx-small')
            ## extract the date and parameter cols to dataframe for control 
            ## site data
            ctrl_data = df[(df[proj_col]==prj) &\
                           (df['project_mon_stat_id']==ctrl_staid) &\
                           (df[par]>0)][['date', phas_ind_col, par]]
            ctrl_data.sort_values('date')

            ## get xment site data
            xmnt_data = df[(df[proj_col]==prj) &\
                           (df['project_mon_stat_id']==xmnt_staid) &\
                           (df[par]>0)][['date', phas_ind_col, par]]
            xmnt_data.sort_values('date')

            ## get list of paired event dates for treatment and control staid
            ## determine where 
            for d in xmnt_data['date']:
                if ctrl_data[ctrl_data.date==d].shape[0]<1:
                    xmnt_data = xmnt_data.drop(xmnt_data[xmnt_data.date==d].index)

            for d in ctrl_data['date']:
                if xmnt_data[xmnt_data.date==d].shape[0]<1:
                    ctrl_data = ctrl_data.drop(ctrl_data[ctrl_data.date==d].index)        
            
            ## check this parameter for transorm
            if params[par]>0:
                if params[par] == 1:
                    ctrl_data[par] = np.log(ctrl_data[par])
                    xmnt_data[par] = np.log(xmnt_data[par])
            ## query paired points for control and treatement for baseline and
            ## treatment phases
            ctrl_pts_bl = ctrl_data[(ctrl_data[phas_ind_col]==0)][par].values
                
            xmnt_pts_bl = xmnt_data[(xmnt_data[phas_ind_col]==0)][par].values
                
            ctrl_pts_xt = ctrl_data[(ctrl_data[phas_ind_col]==1)][par].values
                
            xmnt_pts_xt = xmnt_data[(xmnt_data[phas_ind_col]==1)][par].values
            
            ## if there are sufficient points to plot for paired plot and
            ## regression, do it               
            if ctrl_pts_bl.shape[0] > 0 and xmnt_pts_bl.shape[0] > 0 and \
                ctrl_pts_xt.shape[0] > 0 and xmnt_pts_xt.shape[0] > 0:
                
                # there will be a plot to save for this project
                plt_chk = 1
                    
                ## get max for both plot axes
                xy_max = 1.2* max(ctrl_pts_bl.max(), xmnt_pts_bl.max(),
                             ctrl_pts_xt.max(), xmnt_pts_xt.max())
                
                xy_min = 1.2* min(ctrl_pts_bl.min(), xmnt_pts_bl.min(),
                             ctrl_pts_xt.min(), xmnt_pts_xt.min(), 0)

                ## make regression model for baseline phase
                
                bl_reg = sp.stats.linregress(ctrl_pts_bl, xmnt_pts_bl)
                m_bl = bl_reg.slope
                b_bl = bl_reg.intercept,
                r_bl = bl_reg.rvalue**2
                p_bl = bl_reg.pvalue
                sem_bl = bl_reg.stderr
                seb_bl = bl_reg.intercept_stderr
                ## use regression model to predict over the domain of the axes
                bl_x = [xy_min, xy_max]
                bl_y = [m_bl*bl_x[0]+b_bl, m_bl*bl_x[1]+b_bl]
                
                ## make regression model for treatemnt phase
                xt_reg = sp.stats.linregress(ctrl_pts_xt, xmnt_pts_xt)
                m_xt = xt_reg.slope
                b_xt = xt_reg.intercept,
                r_xt = xt_reg.rvalue**2
                p_xt = xt_reg.pvalue
                sem_xt = xt_reg.stderr
                seb_xt = xt_reg.intercept_stderr
                                
                xt_x = [xy_min, xy_max]
                xt_y = [m_xt*xt_x[0]+b_xt, m_xt*xt_x[1]+b_xt]
            
                ## set axis limits
                ax.set_xlim(left=xy_min, right=xy_max)
                ax.set_ylim(bottom=xy_min, top=xy_max)
                
                ## plot the scatter points
                ax.scatter(ctrl_pts_bl, xmnt_pts_bl, facecolor='none', 
                           edgecolor='b', marker='s',
                           label=plt_label.format('baseline', 
                                                  ctrl_pts_bl.shape[0], 
                                                  m_bl, 
                                                  r_bl))
                ax.scatter(ctrl_pts_xt, xmnt_pts_xt, facecolor='none', 
                           edgecolor='r', marker='o', 
                           label=plt_label.format('treatment', 
                                                  ctrl_pts_xt.shape[0], 
                                                  m_xt, 
                                                  r_xt))
                ## set axis limits
                ax.set_xlim(left=xy_min, right=xy_max)
                ax.set_ylim(bottom=xy_min, top=xy_max)

                ## format the regression line lable for the baseline regression
                ## hide lines from legend
                ax.plot(bl_x, bl_y, color='b', label='_Hiddden')
                ax.plot(xt_x, xt_y, color='r', label='_Hiddden')
                
            plab=par
            if (par not in ['peak_discharge_cfs', 'runoff_in']) and (met !='EMC'):
                if met == 'YIELD':
                    plab = par.split('_')[0]+"_lbs/ac"
                elif met == 'LOAD':
                    plab = par.split('_')[0]+"_lbs"
                else:
                    plab=par
            axs[r,c].set_aspect='equal'
            axs[r,c].tick_params(labelsize='xx-small')
            
            if params[par]>0:
                axs[r,c].set_ylabel("log({0})".format(plab), fontsize='x-small')
                axs[r,c].set_xlabel("log({0})".format(plab), fontsize='x-small')
            else:
                axs[r,c].set_ylabel("{0}".format(plab), fontsize='x-small')
                axs[r,c].set_xlabel("{0}".format(plab), fontsize='x-small')
                
            axs[r,c].set_title("{0}".format(par), fontsize='small', 
                                             fontweight='bold')
            ax.legend(loc='upper left', fontsize='small')    

            
            ## control for subplot location on panel
            if c==1:
                r+=1
                c=0
            else:
                c+=1
        
        if plt_chk != 0:
            fig.tight_layout()
            fig.subplots_adjust(top=0.925)
            fig.suptitle("{0} {1} DQI: {2}".format(prj, met, dqi_min),fontweight='bold')
            fig.savefig(os.path.join(cwd,"scipy_reg_{0}_{1}_DQI-{2}.jpg".format(prj, met, dqi_min)),
                        dpi=300)