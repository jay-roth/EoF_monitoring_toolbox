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
from matplotlib.collections import PolyCollection
import seaborn as sns

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')
    
######## INPUT VARIABLES ######################################################
## metric: EMC, LOAD, YIELD
met = 'EMC'

## name of file containing paired events of control and treatment site data
data_file ="AR_focus_data_{0}_final.csv".format(met)

## minimum event dqi to plot
min_evt_dqi = 1

min_par_dqi = 1

## label for plot legend
plt_label = "{0}, m:{1:1.1f}, r^2:{2:1.2f}" 

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
par_cols = ['peak_discharge_cfs', 'runoff_in', 'NH4_mg.l', 'nitrate:nitrite_mg.l', 
             'TN_mg.l', 'dissolvedp_mg.l', 'TP_mg.l', 'TSS_mg.l']

dqi_cols = ['runoff_dqi', 'runoff_dqi', 'NH4_dqi', 'nitrate:nitrite_dqi', 
             'TN_dqi', 'dissolvedP_dqi', 'TP_dqi', 'TSS_dqi']
## min value of data to plot, just to make sure nan and zeros are out.
min_val = 1e-6

## use paired observation or all observations
paired_obs = False

## colors and labels for plots
ctrl_clr = (0.9, 0.3, 0.2, 0.5)
xmnt_clr = (0.5, 0.8, 0.9, 0.5)

vio_clr = [ctrl_clr, xmnt_clr, ctrl_clr, xmnt_clr]

labs = ['control\nbaseline', 'treatment\nbaseline', 
        'control\ntreatment', 'treatment\ntreatment']

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

df = df[df.event_dqi>=min_evt_dqi]

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
        plt_rows = int((len(par_cols)+1)/2)
        ## row column indexes for subplots
        r = 0
        c = 0
        
        ## ctrl var to determine if a plot is worthy of saving
        plt_chk = 0
        
        ## instantiate figure to put the plots on. Plots will be n x 2 panel.
        fig, axs = plt.subplots(nrows=plt_rows, ncols=2, figsize=[8.5,11], 
                                constrained_layout=True)
        
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
                           (df['project_mon_stat_id']==ctrl_staid)][cols]
            ctrl_data.sort_values('date')
        
            ## get xment site data
            xmnt_data = df[(df[proj_col]==prj) &\
                           (df['project_mon_stat_id']==xmnt_staid)][cols]
            xmnt_data.sort_values('date')
            
            ## need to operated differently on dqis for Loads and Yields
            ## to incorporate the flow as a component of the load and yield
            ## dqi for each WQ constituent
            if met in ['LOAD', 'YIELD'] and\
                            par not in ['peak_discharge_cfs', 'runoff_in']:
                ctrl_data[par_dqi] = (ctrl_data[flo_dqi]*ctrl_data[par_dqi])
                xmnt_data[par_dqi] = (xmnt_data[flo_dqi]*xmnt_data[par_dqi])
                  
                ctrl_data = ctrl_data[(ctrl_data[par_dqi])>=min_par_dqi]
                xmnt_data = xmnt_data[(xmnt_data[par_dqi])>=min_par_dqi]
                
            
            ## if selected query paired points for control and treatement 
            ## for baseline and treatment phases
            if paired_obs == True:
                for d in xmnt_data['date']:
                    if ctrl_data[ctrl_data.date==d].shape[0]<1:
                        xmnt_data = xmnt_data.drop(xmnt_data[xmnt_data.date==d].index)
                for d in ctrl_data['date']:
                    if xmnt_data[xmnt_data.date==d].shape[0]<1:
                        ctrl_data = ctrl_data.drop(ctrl_data[ctrl_data.date==d].index)        
            
            ## make some containers for the plot data
            plot_data, plot_dqis, plot_cnts, plot_meds, plot_avgs, plot_vars= [[],[],[],[],[], []]
            
            ## append the data for control baseline phase > min_dqi
            plot_data.append(np.log(ctrl_data[(ctrl_data[phas_ind_col]==0) &\
                                       (ctrl_data[par]>=min_val) &\
                                       (ctrl_data[par_dqi]>=min_par_dqi)][par].values))
            ## get the number of total events for this site and phase    
            plot_cnts.append(ctrl_data[(ctrl_data[phas_ind_col]==0)].shape[0])
            ## get the average dqi for total events for this site and phase
            plot_dqis.append(ctrl_data[(ctrl_data[phas_ind_col]==0)][par_dqi].mean())
            plot_meds.append(ctrl_data[(ctrl_data[phas_ind_col]==0)][par].median())
            plot_avgs.append(ctrl_data[(ctrl_data[phas_ind_col]==0)][par].mean())
            plot_vars.append(ctrl_data[(ctrl_data[phas_ind_col]==0)][par].var())
            
            ## append the data for control treatment phase > min_dqi
            plot_data.append(np.log(ctrl_data[(ctrl_data[phas_ind_col]==1)&\
                                       (ctrl_data[par]>=min_val) &\
                                       (ctrl_data[par_dqi]>=min_par_dqi)][par].values))
            ## get the number of total events for this site and phase 
            plot_cnts.append(ctrl_data[(ctrl_data[phas_ind_col]==1)].shape[0])
            ## get the average dqi for total events for this site and phase
            plot_dqis.append(ctrl_data[(ctrl_data[phas_ind_col]==1)][par_dqi].mean())   
            plot_meds.append(ctrl_data[(ctrl_data[phas_ind_col]==1)][par].median())
            plot_avgs.append(ctrl_data[(ctrl_data[phas_ind_col]==1)][par].mean())
            plot_vars.append(ctrl_data[(ctrl_data[phas_ind_col]==1)][par].var())
            
            ## append the data for treatement baseline phase > min_dqi 
            plot_data.append(np.log(xmnt_data[(xmnt_data[phas_ind_col]==0)&\
                                       (xmnt_data[par]>=min_val) &\
                                       (xmnt_data[par_dqi]>=min_par_dqi)][par].values))
            ## get the number of total events for this site and phase     
            plot_cnts.append(xmnt_data[(xmnt_data[phas_ind_col]==0)].shape[0])
            ## get the average dqi for total events for this site and phase
            plot_dqis.append(xmnt_data[(xmnt_data[phas_ind_col]==0)][par_dqi].mean())
            plot_meds.append(xmnt_data[(xmnt_data[phas_ind_col]==0)][par].median())
            plot_avgs.append(xmnt_data[(xmnt_data[phas_ind_col]==0)][par].mean())
            plot_vars.append(xmnt_data[(xmnt_data[phas_ind_col]==0)][par].var())
            ## append the data for treatement treatment phase > min_dqi 
            plot_data.append(np.log(xmnt_data[(xmnt_data[phas_ind_col]==1)&\
                                       (xmnt_data[par]>=min_val) &\
                                       (xmnt_data[par_dqi]>=min_par_dqi)][par].values))
            ## get the number of total events for this site and phase 
            plot_cnts.append(xmnt_data[(xmnt_data[phas_ind_col]==1)].shape[0])
            ## get the average dqi for total events for this site and phase
            plot_dqis.append(xmnt_data[(xmnt_data[phas_ind_col]==1)][par_dqi].mean())
            plot_dqis.append(xmnt_data[(xmnt_data[phas_ind_col]==1)][par_dqi].mean())
            plot_meds.append(xmnt_data[(xmnt_data[phas_ind_col]==1)][par].median())
            plot_avgs.append(xmnt_data[(xmnt_data[phas_ind_col]==1)][par].mean())
            plot_vars.append(xmnt_data[(xmnt_data[phas_ind_col]==1)][par].var())
            sns.violinplot(data=plot_data, widths=0.25, cut=3, palette=vio_clr, 
                           ax=axs[r,c])
            
            ## adjust the lower axis limit to make room for annotation
            ylim = axs[r,c].get_ylim()
            axs[r,c].set_ylim(bottom=ylim[0]-3)
            ylim = axs[r,c].get_ylim()
                        
            for i in range(len(plot_cnts)):
                if plot_data[i].shape[0] > 0:
                    if plot_vars[i] > 1e4:
                        txt ="n:{0:1.0f}\nmean:{1:1.2f}\nmedian:{2:1.2f}\nvariance:{3:1.1e}"
                    else:
                        txt ="n:{0:1.0f}\nmean:{1:1.2f}\nmedian:{2:1.2f}\nvariance:{3:1.2f}"
                    txt = txt.format(plot_cnts[i], plot_avgs[i],plot_meds[i], plot_vars[i])
                    #txt ="n:{0:1.0f}\n".format(plot_cnts[i])
                    axs[r,c].text(i, ylim[0]+1, txt, ha='center', fontsize='xx-small')
                else: 
                    txt = 'n:na'
            
            if (par not in ['peak_discharge_cfs', 'runoff_in']) and (met !='EMC'):
                if met == 'YIELD':
                    par = par.split('_')[0]+"_lbs/ac"
                if met == 'LOAD':
                    par = par.split('_')[0]+"_lbs"
                    
            axs[r,c].set_aspect='equal'
            axs[r,c].tick_params(labelsize='xx-small')
            axs[r,c].set_xticklabels(labs)
            axs[r,c].set_ylabel("log({0})".format(par), fontsize='xx-small')
            axs[r,c].set_title("{0}".format(par), fontsize='small', 
                                     fontweight='normal')

            ## control for subplot location on panel
            if c==1:
                r+=1
                c=0
            else:
                c+=1

        #fig.tight_layout()
        fig.subplots_adjust(top=0.925)
        fig.suptitle("{0} {1} DQI: {2}".format(prj, met, min_evt_dqi),fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(cwd,"{0}_{1}_DQI-{2}.jpg".format(prj, met, min_evt_dqi)),
                    dpi=300)