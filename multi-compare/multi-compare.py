# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:38:56 2023

@author: Jason.Roth
@email: jason.roth@usda.gov
@position:Env.Engr, WQQT
@agency: USDA-NRCS

Description:
standalone script for performing anova analysis on paired watershed data
and generating plots and tables from the analysis.

Requires standard library, numpy, pandas, and matplotlib

Ingests a regularly formatted instruction file and csv dataset.
Conducts analysis on dataset as specified in instruction file.


1.) import data.
2.) iterate over groups
3.) iterate over treatments wihtin a group
4.) iterate over observations
5.) remove outliers for each phase, site combo
    a.) enough data left?
6.) test for normality
    a.) can they be made normal?
    
    a.) data are normal?
        i.) perform ANOVA -> Tukey hsd
    b.) data can be transformed normal?
        i.) transform data
       ii.) perform ANOVA -> Tukey hsd
    b.) data are not normal
        ii.) perform KW -> Dunns test
    d.) plot figures
    c.) print stats
"""

import os
import pandas as pd
import numpy as np
from scipy import stats as stats
from matplotlib import pyplot as plt

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')

def format_axis(ax, ylo=3.5, yhi=2):
    ax.set_aspect='equal'
    ax.tick_params(labelsize='xx-small')
    ymin = int(1.1*ax.get_ylim()[0])-ylo
    ymax = int(1.2*ax.get_ylim()[1])+yhi
    ax.set_ylim(ymin,ymax)
    return ax

def annotate_axis(ax, xlab, ylab, grp_lab, sta_lab, offset=1):
    ax.set_xticklabels(xlab)
    
    if ylab != '':    
        ax.set_ylabel(ylab)
    
    for i in range(len(xlab)):
        ax.text(i+offset, 0.90*ax.get_ylim()[1]-0.5, grp_lab[i], 
            ha='center', fontsize='small')
        ax.text(i+offset, ax.get_ylim()[0]+0.5, sta_lab[i], 
            ha='center', fontsize='xx-small')
        
    ax.set_title("{0}".format(ylab), fontsize='small', 
                     fontweight='normal')

def box_plot(ax, dat):
    ## colors and labels for plots
    ctrl_clr = (0.9, 0.3, 0.2, 0.25)
    xmnt_clr = (0.5, 0.8, 0.9, 0.25)

    box_clr = [ctrl_clr, xmnt_clr, ctrl_clr, xmnt_clr]
    
    bplot = ax.boxplot(dat, showmeans=True, patch_artist=True)
    
    #ax.set_title('Box plot')
    for patch, color in zip(bplot['boxes'], box_clr):
        patch.set_facecolor(color)
    return format_axis(ax)

def read_params(file_path):
    """
    reads in the input data file for the ancova analysis
    
    Parameters
    ----------
    file_path : TYPE - str
                DESCRIPTION - path to data file  
        
    Returns
    -------
    d : TYPE - list, 
        DESCRIPTION - list containing contents of file
    msg : TYPE - string
        DESCRIPTION - message containing results of file read.

    10/25/2023: created function - j.roth    
    """    
    
    ## dictionary to contain input parameters
    in_dat = {'file':{'val':'','type':[str]},
              'group':{'val':'','type':[str]},
              'site':{'val':'','type':[str]},
              'type':{'val':'','type':[str]},
              'phase':{'val':'','type':[str]},
              'date':{'val':'','type':[str]},
              'data':{'val':'','type':[list]},
              'units':{'val':'','type':[list]},
              'alpha':{'val':'','type':[float]},
              'scale':{'val':'','type':[str, float]},
              'norm':{'val':'','type':[str, float]},
              'conv':{'val':'','type':[float]}}
    
    ## parameters that are given as comma sep lists.
    lst_params = ['data', 'units']
    
    ## required parameters
    req_params = ['file','group','site','type','phase','date','data','units',
                  'alpha']
                  
    ## parameters with possible numeric values
    num_params = ['alpha', 'scale', 'norm', 'conv']
    
    ## read in the input file
    with open(file_path, 'r') as f:
        i=0
        for l in f.readlines():
            ## if line is not a comment, keep it
            if l[0] != '#':
                v = [i.strip() for i in l.split('=')]
                ## is this data singular or a list type
                if v[0] not in lst_params:
                    ## if not possible to be numeric store string
                    if v[0] not in num_params:
                        in_dat[v[0]]['val'] = v[1]
                    else:
                        ## if possibly numeric, check and store numeric,
                        ## else store string
                        if v[1][0].isnumeric():
                            in_dat[v[0]]['val'] = float(v[1])
                        else:
                            in_dat[v[0]]['val'] = v[1]
                # data are a list of values
                else:
                    in_dat[v[0]]['val'] = [i.strip() for i in v[1].split(',')]                
            i+=1
    
    ## check that required values are present and all data types are correct
    msg = 'PARAMETER FILE READ ERRORS\n'
    req_msg = "\tParameter '{0}' missing or wrong type, must be type '{1}'\n"
    opt_msg = "\tOptional parameter '{0}' wrong type, must be type '{1}'\n"
    err = 0
    ## run check on data existence and types
    for k in in_dat.keys():
        if k in req_params:
            if not (in_dat[k]['val'] != '' or\
                type(in_dat[k]['val']) not in in_dat[k]['type']):
                msg+=req_msg.format(k,in_dat[k]['type'])
                err = 1
        else:
            if in_dat[k]['val'] != '':
                if type(in_dat[k]['val']) not in in_dat[k]['type']:
                    msg+=opt_msg.format(k,in_dat[k]['type'])
                    err = 1
                    
    ## check that length of units is same as data
    if len(in_dat['data']['val']) != len(in_dat['units']['val']):
        msg += "\tNumber of data columns and number of units must be equal\n"
        err = 1
        
    # return data and error msg
    if err == 0:
        fname = os.path.split(file_path)[-1]
        msg = "PARAMETER FILE '{0}' READ SUCCESSFULLY\n".format(fname)
    return in_dat, msg, err

def readWqData(params):
    
    """
    Operates on data columns of dataset multiplying them by values of a 
    scalar or multiplier data column if provided, dividing them by a scalar 
    or values in a normalization column if provided, and multiplying them by 
    a scalar conversion factor, if provided.
    
    Parameters
    ----------
    df : TYPE - pandas dataframe
        DESCRIPTION - paired watershed data
    par : TYPE - dictionary
        DESCRIPTION - script control parameters and data columns 

    Returns
    -------
    msg : TYPE - string
        DESCRIPTION - message conveying results of data check
    
    10/25/2023: created function - j.roth    
    """ 
    req_cols = ['group','site','type','phase','date']
    opt_cols = ['norm', 'scale', 'conv']
    dat_cols = 'data'
    opt_vals = []
    msg = 'DATA FILE READ ERRORS\n'
    err = 0
    err_msg = '\tspecified {0} column: {1} not found in input dataset\n'
    dat_msg = '\tspecified data column: {0} not found in input dataset\n'
    opt_msg = '\toptional column: {0} not found in input dataset\n'
    
    ## read in the data
    df = pd.read_csv(params['file']['val'])
    
    ## check for required columns
    for c in req_cols:
        if params[c]['val'] not in df.columns:
            msg+=err_msg.format(c, params[c]['val'])
            err = 1
            
    ## check for data columns 
    for d in params[dat_cols]['val']:
        if d not in df.columns:
            msg+=dat_msg.format(params[c]['val'])
            err = 1
            
    ## check for optional columns
    for oc in opt_cols:
        print(oc)
        opt_vals.append(params[oc]['val'])
        
    for ov in opt_vals:
        if ov != '':
            if type(ov) == str:
                if ov not in df.columns:
                    msg+=opt_msg.format(ov)
                    err = 1
                else:
                    ## if an optional column is specified. Remove all entries
                    ## where optional column is na
                    df = df[df[ov].notna()]
                    for c in params[dat_cols]['val']:
                        ## switch operation based on scaling or 
                        if c not in opt_vals:
                            if ov != opt_vals[0]:
                                df[c] = df[c]*df[ov]
                            else:
                                df[c] = df[c]/df[ov]
            else:
                for c in params[dat_cols]['val']:
                    if c not in opt_vals:
                        if oc != opt_vals[0]:
                            df[c] = df[c]*ov
                        else:
                            df[c] = df[c]/ov

    if df.shape[0] == 0:
        msg += 'DATA READ ERRORS\n'
        msg += "\tno data after removing null values for scale and/or norm\n"
        err = 1
    
    ## check that there is only 1 control per site group
    if df[params['type']['val']].unique().shape[0]<2:
        msg += "\tInsufficient number of site types, min=2"
        err = 1
    
       
    if df[df[params['type']['val']]!=0].shape[0] == 0:
        msg += "\tInsufficient number of phases, min=2"
        err = 1    
       
    ## convert dates to date time values
    df[params['date']['val']]=pd.to_datetime(df[params['date']['val']])
    
    ## negate message if all operations were successful
    if err == 0:
        msg = "DATA FILE '{0}' READ SUCCESSFULLY\n".format(params['file']['val'])

    return df, msg, err
        
def appendLogFile(f, msg, m='a+'):
    """
    Parameters
    ----------
    f : TYPE- String
        DESCRIPTION- Path to log file
    msg :TYPE- String
         Description- String to append to log file
    m : TYPE- String, optional
        DESCRIPTION- Mode to open file in, The default is 'a+'.

    Returns
    -------
    None.
    
    10/25/2023: created function - j.roth 
    """
    
    with open(f, m) as f:
        f.write(msg)


def getPairedData(pdf, date_col, type_col, min_evt=5):
    """
    

    Parameters
    ----------
    df : TYPE - dataframe
        DESCRIPTION - sample data for >1 sites and corresponding dates 
    prj : TYPE - 
        DESCRIPTION.
    ctrl_staid : TYPE
        DESCRIPTION.
    xmnt_staid : TYPE
        DESCRIPTION.
    par : TYPE
        DESCRIPTION.

    Returns
    -------
    pdf : TYPE
        DESCRIPTION.
    chk : TYPE
        DESCRIPTION.

    2/25/2023: created function - j.roth 
    """
    types = pdf.types.unique()
     

    ct_data = pdf[(pdf[type_col]==types[0]) &\
                   (df[obs_col]>0)][[date_col, 
                                     phase_col, 
                                     obs_col]]

    ct_data.sort_values('date')

    ## get xment site data
    tr_data = pdf[(pdf[type_col]!=types[0]) &\
                   (df[obs_col]>0)][[date_col, 
                                     phase_col, 
                                     obs_col]]
  
    tr_data.sort_values('date')
    ## get list of paired event dates for treatment and control staid
    ## determine where 
    for d in tr_data[date_col]:
        if ct_data[ct_data[date_col]==d].shape[0]<1:
            tr_data = tr_data.drop(tr_data[tr_data[date_col]==d].index)

    for d in ct_data[date_col]:
        if tr_data[tr_data[date_col]==d].shape[0]<1:
            ct_data = ct_data.drop(ct_data[ct_data[date_col]==d].index)      

    ## change names of data columns before concatenating
    ## treatment data is dependant ergo Y, control is independant ergo X
    ct_data = ct_data.rename(columns={obs_col:types[0]})
    tr_data = tr_data.rename(columns={obs_col:types[1]})

    tr_data.set_index(ct_data.index, inplace=True)

    par_df = pd.concat([ct_data, tr_data[types[1]]], axis=1)

    msg, err = checkPairedDataCount(par_df, phase_col, min_evt)   
    
    return par_df, msg, err

def checkPairedDataCount(df, phase_col, min_evt=5):
    """
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    phas_col : TYPE
        DESCRIPTION.
    min_evt : TYPE
        DESCRIPTION.

    Returns
    -------
    msg : TYPE
        DESCRIPTION.
    err : TYPE
        DESCRIPTION.

    """

    msg = '\t\tPAIRED EVENT PROCESSING RESULTS\n'
    err = 0
    
    ## move to its own function
    ## check to see if there enough events in the control and treatment phase
    phases = df[phase_col].unique()
    if phases.shape[0]>1:
        for phs in phases[0:2]:
            if df[df[phase_col]==phs].shape[0] < min_evt:
                err = 1
                tmp = '\t\t\tInsufficient # events (min = {0}) during {1} phase\n'
                msg+= tmp.format(min_evt, phs)
            else:
                tmp = '\t\t\t{0} # events (min = {1}) during {2} phase\n'
                msg += tmp.format(df[df[phase_col]==phs].shape[0], min_evt, phs)
    else:
        err = 1
        if df[phase_col][0].values[0] == 0:
            msg += '\t\t\tNo paired events during treatment phase\n'
        else:
            msg += '\t\t\tNo paired events during baseline phase\n'
        
    return msg, err
    
def removeOutliers(df, typ, phs, par, m=1.5, min_evt=5):
    """
    Take paired data and removes the outliers from each data set as m% of iqr.
    to preserve paired data.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    m : TYPE, optional
        DESCRIPTION. The default is 1.5.

    Returns
    -------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    2/25/2023: created function - j.roth
    10/30/2023: modified to work with new standalone script - j.roth    
    """
    #sites = df[sit].unique()
    types = df[typ].unique()
    phases = df[phs].unique()
    init_count = df.shape[0]

    for t in types:
        for p in phases:
            ## loop through phases
            dat = df[(df[phs]==p) & (df[typ]==t)][par].values
            if dat.shape[0] > 3:
                
                ## get the interquartile range
                x25, x75 = np.percentile(dat, [25,75])
                x_iqr = x75-x25
                
                ## get the min and max values for outliers
                x_lo = x25-x_iqr*m
                x_hi = x75+x_iqr*m
            
                ## drop indices that fall outside of ranges
                d_idx = df[(df[typ]==t) & (df[phs]==p) & (df[par]<x_lo)].index
                df = df.drop(d_idx)
                d_idx = df[(df[typ]==t) & (df[phs]==p) & (df[par]>x_hi)].index
                df = df.drop(d_idx)            
            
            #msg, err = checkPairedDataCount(df, phs, min_evt=5)
            #if err == 1:
            #    msg = 'OUTLIER REMOVAL ERROR\n' + msg
            #    msg.format(min_evt, par)  
            #else:
            
            final_count = df.shape[0]
            msg = '\t\tOUTLIER REMOVAL RESULTS\n'
            delta = init_count-final_count
            temp = '\t\t\t{0} outliers found in {1} samples for par {2}\n'
            msg += temp.format(delta, init_count, par)
            
            err = 0 
                
        return df, msg, err
            
def transformData(df, typ, phs, par, alpha=0.025, min_pass=2):
    """

    Parameters
    ----------
    pdf : TYPE
        DESCRIPTION.
    sit_col : TYPE
        DESCRIPTION.
    par_col : TYPE
        DESCRIPTION.
    phs_col : TYPE
        DESCRIPTION.
    min_pval : TYPE, optional
        DESCRIPTION. The default is 0.025.
    min_pass : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    cv : TYPE
        DESCRIPTION.
    xform : TYPE
        DESCRIPTION.
    nonpar : TYPE
        DESCRIPTION.

    """
    err = 0
    t = '\t'
    
    #sites = df[sit].unique()
    types = df[typ].unique()
    phases = df[phs].unique()
    init_count = df.shape[0]
    
    
    ## containers to keep tally of best transform
    ## first index is levene's equal variances.
    ## second is shapiro test for each site/phase combo
    raw_tst = [0]*(len(types)*len(phases)+1)
    xfm_tst = [0]*(len(types)*len(phases)+1)
    
    raw_dat = []
    xfm_dat = []
    
    msg = 2*t+"DATA TRANSFORMATION RESULTS\n"
    
    lev_msg = 3*t+"{0} variances {1} equal, alpha {2:0.2f}\n"
    
    shap_msg = 3*t+"{0} of {1} {2} data sets are normal, alpha {3:0.2f}\n"
    
    raw_msg = ''
    xfm_msg = ''
    
    ## Loop over sites in the sit_col
    for t in df[typ].unique():
        for p in df[phs].unique():
            raw_dat.append(df[(df[typ]==t) & (df[phs]==p)][par].values)
    for rd in raw_dat:
        xfm_dat.append(np.log(rd))
            
    ## First do a levenes test on untransformed data
    raw_res = stats.levene(raw_dat[0], raw_dat[1], raw_dat[2], 
                                 raw_dat[3])
    ## Check Levenes test result
    if raw_res[1] >= alpha:
        raw_tst[0] = 1
        raw_msg += lev_msg.format("Raw", "are", alpha)
    else:
        raw_msg += lev_msg.format("Raw", "are not", alpha)
        
    xfm_res = stats.levene(xfm_dat[0], xfm_dat[1], xfm_dat[2], 
                                 xfm_dat[3])
    
    if xfm_res[1] >= alpha:
        raw_tst[0] = 1
        xfm_msg += lev_msg.format("Log", "are", alpha)
    else:
        xfm_msg += lev_msg.format("Log", "are not", alpha)
        
    ## move onto shapiro if both are still equal
    for l in range(len(raw_dat)):
        raw_p = stats.shapiro(raw_dat[l])                
        if raw_p[1] >= alpha:
            raw_tst[l+1] = 1
    raw_msg += shap_msg.format(sum(raw_tst[1:]), len(raw_tst)-1, "Raw", alpha )        

    ## Set index         
    for l in range(len(raw_dat)):  
        xfm_p = stats.shapiro(xfm_dat[l])
        if xfm_p[1] >= alpha:
            xfm_tst[l+1] = 1

    xfm_msg += shap_msg.format(sum(xfm_tst[1:]), len(xfm_tst)-1, "Log", alpha ) 

    if sum(raw_tst)<5:
        raw_msg += 3*t+"WARNING raw data sets not all normal\n"
        
    if sum(xfm_tst)<5:
        xfm_msg += 3*t+"WARNING transformed data sets not all normal\n"

    if sum(xfm_tst) > min_pass or sum(raw_tst) > min_pass:
        nrm = 1
        if sum(xfm_tst) > sum(raw_tst):
            df[par] = np.log(df[par].values)
            msg += xfm_msg
            msg += 3*t+'Using Log data values for par: {0}\n'.format(par)
            xfm = 1
        else:
            msg += raw_msg
            msg += 3*t+'Using raw data values for par : {0}\n'.format(par)
            xfm = 0
    else:
        msg+= 3*t+"Raw and Log data failed to pass test for normality\n"
        nrm = 0
        xfm = 0
        
    return df, nrm, xfm, msg


#def sort_groups(df)


def get_groups(df, pval):
    """
    distinguish groups based on pvals from multiple comparison assessment
    
    """
    
    cv = ['a','b','c','d','e','f','g']
    i=1
    t=0
    odf = df.shape[0]*[cv[0]]
    for r in range(0, df.shape[0]-1):
        for c in range(r+1, df.shape[1]):
            sec=0
            print("comparing {0} and {1}: pval:{2}".format(r+1, c+1, df[c][r] ))
            print("current group values: {0}, {1}".format(odf[r], odf[c]))
            if df[c][r] < pval:
                if len(odf[c]) > len(odf[r]):
                    if odf[r] in odf[c]:
                        odf[c] = cv[i]
                        t=1
                else:
                    if odf[c] in odf[r]:
                        odf[c] = cv[i]
                        t=1
                sec=1
            else:
                if len(odf[c]) > len(odf[r]):
                    if odf[r] not in odf[c]:
                        odf[c]+=odf[c-1]
                        sec=2
                else:
                    if odf[c] not in odf[r]:
                        odf[r]+=odf[c]
                        sec=3
            print("new group values: {0}, {1}".format(odf[r], odf[c]))
            print("section {0}".format(sec))
        if t > 0:
            i+=1
            t=0
    return odf

## BEGIN MAIN #################################################################
if __name__ == "__main__":
    
    ## establish some vanilla parameters: log_file and out_dir
    log_file = "log.txt"
    out_dir = "output"
    
    ## name for indvidual anova and regression tables
    sum_tbl_name = "{0}_sites#{1}+{2}_param#{3}_reg-summary.txt"
    
    ## name for individual plots
    plt_name = "{0}_sites#{1}+{2}_param#{3}_plot_multi-comp.png"
    
    ## name for multi plot file format with group name
    pnl_plt_name = "{0}_sites#{1}+{2}_multi-multi-comp.png"
    
    ## file to read input parameters from
    param_file = 'NRCS_EoF_MULTICOMP_input.txt'
    
    ## columns to include in the output dataframe
    out_cols = ['group', 'ctrl_id', 'trmt_id', 'param', 'phase', 'n', 'x_avg', 
                'y_avg', 'slope', 'slope_pval', 'intercept', 'intercept_pval',
                'r2', 'xform']
    
    os.path.join(out_dir, log_file)
    outdir = os.path.join(os.getcwd(), out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    #TODO: these should be in the control file
    remove_outliers = True
    paired_events = True
    ## minimum event count
    min_evt = 5
    ## switch to remove outliers
    rem_out = 1
    
    ## read in the parameter file data
    params, msg, err = read_params(param_file)
    
    appendLogFile(log_file, msg, 'w')
    
    if err == 0:
    
        ## read in the water quality data
        df, msg, err = readWqData(params)
        appendLogFile(log_file, msg)
        
        if err == 0:
            ## start analysis, loop over groups 
            group_col = params['group']['val']
            site_col = params['site']['val']
            type_col = params['type']['val']
            phase_col = params['phase']['val']
            date_col = params['date']['val']
            alpha = params['alpha']['val']
    
            for g in df[group_col][:].unique():
                msg = "PROCESSING DATA & ANALYSIS FOR GROUP: {0}\n".format(g)
                appendLogFile(log_file, msg)
                
                ## 
                sites = df[df[group_col]==g][site_col][:].unique()

                ## the first site in the group is assumed to be the control
                cs = sites[0]
                
                ##
                tr_sites = sites[1:]
                    
                ## check if there are control and treatment sites
                if tr_sites.shape[0] > 0:
                    
                    ## loop over treatment sites
                    for ts in tr_sites:
                        msg = "Performing analysis for control {0} and treatment {1}\n"
                        appendLogFile(log_file, msg.format(cs, ts))
                        
                        ## get the types of sites: typically "control" and "treatment"
                        ## but may be something more specific like "no practice" and "cover crop"
                        ## the "type" specified by the first station is assumed to be control
                        ## and any subsequents are treatmemnt
                        types = df[(df[group_col]==g) & ((df[site_col]==cs) |
                                       (df[site_col]==ts))][type_col].unique()[0:2]
                        
                        ## phases are assumed to be chronological i.e. baseline and treatmemnt
                        ## names can be arbitrary but plots will be made using the sequence in
                        ## which they occur in the data.
                        phases = df[(df[group_col]==g) & ((df[site_col]==cs) |
                                       (df[site_col]==ts))][phase_col].unique()[0:2]
                        
                        ## loop over parameters and create a
                        ## for each instance
                        obs_cols = params['data']['val']
                        
                        ## Instantiate the panel plot
                        plt_rows = int((len(obs_cols)+1)/2)
                        ## row column indexes for subplots
                        plt_row = 0
                        plot_col = 0
                        
                        ## ctrl var to determine if a plot is worth saving
                        plt_chk = 0
                        
                        #TODO: bump this to the end if multi-figure is wanted
                        # save figs to a list and dump into a subplot later
                        ## instantiate figure to put the plots on. 
                        ## Plots will be n x 2 panel. Scale fig size to number of
                        ## rows needed
                        x_fig, x_axs = plt.subplots(nrows=plt_rows, ncols=2, 
                                                figsize=[8.5,max(11/4*plt_rows,11)], 
                                                constrained_layout=True)
                        
                        ## iterate over the observations
                        for obs_col in obs_cols:
                            msg = "\tProcessing Observation {0}\n".format(obs_col)
                            appendLogFile(log_file, msg)
                            
                            ## get req'd data for this group of sites and obs
                            pdf = df[(df[group_col]==g) 
                                     & (df[obs_col].notna()) 
                                     & df[site_col].isin([cs,ts])][[site_col, date_col, type_col, phase_col, obs_col]]
                            
                            ## remove outlier data if selected
                            if remove_outliers == True:
                                pdf, msg, err = removeOutliers(pdf,
                                                               type_col, 
                                                               phase_col,
                                                               obs_col,
                                                               m=1.5)
                                
                                appendLogFile(log_file, msg)
                            else:
                                msg='Outlier removal option not selected\n'
                                appendLogFile(log_file, msg)
                        
                            if paired_events == True:
                               
                                ## pair remaining data for final analysis
                                pdf, msg, err = getPairedData(pdf, 
                                                              date_col, 
                                                              type_col,
                                                              min_evt=5)
                                
                                msg='Utilizing data for only paired events\n'
                                appendLogFile(log_file, msg)
                                
                            else:
                                msg='Utilizing data for all events\n'
                                appendLogFile(log_file, msg)
                                
                            ## check to see if data are normal or capable of being 
                            ## transformed to normal
                            pdf, norm, xfrm, msg, err = transformData(pdf, 
                                                                      obs_col,
                                                                      site_col,
                                                                      type_col,
                                                                      phase_col, 
                                                                      alpha)

                            appendLogFile(log_file, msg)
                            #TODO You are here
                            
                            ## run the anova analysis
                            if norm == 1:

                                r0, r1, adf, mdf, cdf = runAncova(pdf, 
                                                                  iv_col=types[0],
                                                                  dv_col=types[1], 
                                                                  cov_col=phase_col,
                                                                  alpha=alpha)
                                r0.name = cs
                                r1.name = ts
                                if xf == 1:
                                    r0.xfrm = 1
                                    r1.xfrm = 1

                                unit = params['units']['val'][obs.index(obs_col)]
                                plt_lab = "{0} ({1})"
                                plt_lab = plt_lab.format(obs_col, unit)
     
                                fig = plot_ancova([r0, r1], types, phases,
                                                  plt_lab, alfa=0.1, 
                                                  plt_mean=False,
                                                  plt_perr=False, 
                                                  plt_conf=False)
                                
    
                                fig_nam = plt_name.format(g, cs, ts, obs_col)
                                fig_pth = os.path.join(outdir, fig_nam)
                                fig.savefig(fig_pth)
                                tbl_nam = reg_tbl_name.format(g, cs, ts, 
                                                              obs_col)
                                tbl_pth = os.path.join(outdir, tbl_nam)
                                printRegStats(adf, mdf, cdf, g, cs, ts, 
                                              obs_col, tbl_pth)
                            if err == 1:
                                msg = "\t\tData volume or quality insufficient to "+\
                                        "perform ANCOVA for " +\
                                        "sites {0}, {1} and observation {2}\n".format(cs, 
                                                                                  ts, 
                                                                                  obs_col)
                            else:
                                msg ="\t\tData volume or quality sufficient to "+\
                                        "perform ANCOVA for " +\
                                        "sites {0}, {1} and observation {2}\n".format(cs, 
                                                                                  ts, 
                                                                                  obs_col)
                                msg+="\t\tCheck output directory for plot and regression table\n"
                            appendLogFile(log_file, msg)