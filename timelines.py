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
import matplotlib.dates as mdates

######## INPUT VARIABLES ######################################################

## name of file containing paired events of control and treatment site data
data_file ="AR_focus_data_{0}_final.csv".format('EMC')

## minimum event dqi to plot
dqi_min = 0

## label for plot legend
plt_label = "{0}" 

#### names of columns for use in logic and labeling
## site indicator column, control or treatment: n>0 is treatment
site_ind_col= 'site_indicator'

## phase indicator column, baseline or treatment: n>0 treatment
phas_ind_col = 'phase_indicator'

## date of event
date_col = 'date'

## project name column
prj_col = 'project_title'

## parameter to scale the bars to
par = 'runoff_in'

## min parameter value.
min_par_val = 1e-2

## name of the figure
fig_name = "Monitoring Events for {0}"

## file name to save fig as
fil_name = "{0}_events.jpg"

## series lables
ct_lab = "Control- {0} (BL:{1}, TX:{2})"
tx_lab = "Treatment- {0} (BL:{1}, TX:{2})"

x_ax_lab ='Date'

y_ax_lab = 'Total Runoff (in)'

evt_txt = "Number of unique events: {0}\nNumber of paired events: {1}"

######## BEGIN CODE ###########################################################
## get the current working dir
cwd = os.getcwd()

## import the data into a data 
#encoding='windows-1252'
df = pd.read_csv(os.path.join(cwd, data_file))

## convert date strings in "date" column to date time values
df['date'] = pd.to_datetime(df['date'])

df = df[df.event_dqi>=dqi_min]

## get a list of unique projects
prjs = df[prj_col].unique()
prjs.sort()

all_xmnt_staids = df[(df[site_ind_col]>0)]['project_mon_stat_id'].unique()

plt_idx = 0
## loop over all projects
for prj in prjs:
    ## instantiate figure to put the plot on
    fig = plt.figure(figsize=[6, 1.75], constrained_layout=True)
    ax = fig.add_subplot(111)
    
    ctrl_dates = np.array([])
    xmnt_dates = np.array([])
    
    ## determine if there's a control station
    ctrl_staids = df[(df[prj_col]==prj) & (df[site_ind_col]==0)]\
                    ['project_mon_stat_id']
    if ctrl_staids.shape[0] > 0:
    ## get control staid for this project
        ctrl_staid = df[(df[prj_col]==prj) & (df[site_ind_col]==0)]\
                        ['project_mon_stat_id'].values[0]
    ## if no control staid set to ''
    else:
        ctrl_staid = ''
        
    ## get all treatment staids for this project
    xmnt_staids = df[(df[prj_col]==prj) & (df[site_ind_col]>0)]\
                    ['project_mon_stat_id'].unique()
    ## sort xmnt_staids
    xmnt_staids.sort()
    
    ## get control data
    ctrl_data = df[(df[prj_col]==prj) &\
                   (df['project_mon_stat_id']==ctrl_staid) &\
                    df[par]>=min_par_val]\
                    [['date', phas_ind_col, par]]
    ctrl_data.sort_values('date')
    ctrl_dates = ctrl_data['date'].unique()
     
    ## plot control data
    if ctrl_staid != '':
        
        ctrl_bl_evts = ctrl_data[ctrl_data[phas_ind_col]==0].shape[0]
        ctrl_tx_evts = ctrl_data[ctrl_data[phas_ind_col]==1].shape[0]
        
        ax.bar(ctrl_data['date'], ctrl_data[par], 
               label=ct_lab.format(ctrl_staid, ctrl_bl_evts, ctrl_tx_evts))
        
    d_idx=1
    for xmnt_staid in xmnt_staids:
        ## get treatment data
        xmnt_data = df[(df[prj_col]==prj) &\
                       (df['project_mon_stat_id']==xmnt_staid) &\
                        df[par]>=min_par_val]\
                        [['date', phas_ind_col, par]]
        xmnt_data.sort_values('date')   
        
        xmnt_bl_evts = xmnt_data[xmnt_data[phas_ind_col]==0].shape[0]
        xmnt_tx_evts = xmnt_data[xmnt_data[phas_ind_col]==1].shape[0]
        
       
        
        ax.bar(xmnt_data['date']+pd.tseries.offsets.DateOffset(days=d_idx), 
               xmnt_data[par], label=tx_lab.format(xmnt_staid, 
                                                   xmnt_bl_evts, 
                                                   xmnt_tx_evts))

        if d_idx == 1:
            xmnt_dates = xmnt_data['date'].unique()
        else:
            xmnt_dates = np.append(xmnt_dates, xmnt_data['date'].unique())
        d_idx+=1
    
    ## get total number of unique and paired events
    tot_evt = np.unique(np.append(ctrl_dates, xmnt_dates)).shape[0]
    
    for cd in ctrl_dates:
        if cd not in xmnt_dates:
            ctrl_dates = np.delete(ctrl_dates, np.where(ctrl_dates==cd))
    for xd in xmnt_dates:
        if xd not in ctrl_dates:
            xmnt_dates = np.delete(xmnt_dates, np.where(xmnt_dates==xd))
    
    pair_evt = np.unique(np.append(ctrl_dates, xmnt_dates)).shape[0]
    
    
    ## add event counts to legend by making empty plot
    #ax.bar(xmnt_dates, np.zeros(xmnt_dates.shape[0]), facecolor='None', 
    #       label=evt_txt.format(tot_evt, pair_evt))
    
    ## add event counts to legend at specific location
    ytxt = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.8
    xtxt = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.5
    ax.text(mdates.num2date(xtxt), ytxt, evt_txt.format(tot_evt, pair_evt), 
            ha='center', fontsize='xx-small')
    
    ## shade baseline and treatment phases
    if ctrl_data[ctrl_data['phase_indicator']==0].shape[0] > 1:
        bl_beg = min(ctrl_data[ctrl_data['phase_indicator']==0]['date'].min(),
                       xmnt_data[xmnt_data['phase_indicator']==0]['date'].min())
        bl_beg = mdates.num2date(ax.get_xlim()[0])
        bl_end = max(ctrl_data[ctrl_data['phase_indicator']==0]['date'].max(),
                       xmnt_data[xmnt_data['phase_indicator']==0]['date'].max())
        bl_end = dt(bl_end.year, 12, 31)
        tx_end = max(ctrl_data[ctrl_data['phase_indicator']==1]['date'].max(),
                       xmnt_data[xmnt_data['phase_indicator']==1]['date'].max())
        tx_end = mdates.num2date(ax.get_xlim()[1])
        ax.fill_between([bl_beg, bl_end],[ax.get_ylim()[1],ax.get_ylim()[1]],
                        facecolor='blue', alpha=0.04)
        ax.fill_between([bl_end, tx_end],[ax.get_ylim()[1],ax.get_ylim()[1]],
                        facecolor='red', alpha=0.04)
        ## add event counts to legend at specific location
        ytxt = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.5
        xtxt = (mdates.date2num(bl_end)+mdates.date2num(bl_beg))/2
        ax.text(mdates.num2date(xtxt), ytxt, "baseline", 
                ha='center', fontsize='xx-small', color='blue', alpha=0.25)
        xtxt = (mdates.date2num(tx_end)+mdates.date2num(bl_end))/2
        ax.text(mdates.num2date(xtxt), ytxt, "treatment", 
                ha='center', fontsize='xx-small', color='red', alpha=0.25)
        
    ax.tick_params(labelsize='xx-small')
    ax.xaxis.set_major_locator(mdates.MonthLocator([1,7], 1, 1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.set_xlabel(x_ax_lab, fontsize='xx-small')
    ax.set_ylabel(y_ax_lab, fontsize='xx-small')
    ax.legend(loc='upper left', fontsize='xx-small')    
    #ax.set_title(fig_name.format(prj), fontsize='small', fontweight='bold')
    plt_idx+=1
            
    fig.suptitle(fig_name.format(prj),fontweight='normal', fontsize='small')
    fig.savefig(os.path.join(cwd,fil_name.format(prj)), dpi=300)