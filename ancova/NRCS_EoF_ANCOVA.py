# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:12:39 2023

@author: Jason.Roth
@email: jason.roth@usda.gov
@position:Env.Engr, WQQT
@agency: USDA-NRCS

Description:

standalone script for performing ancova analysis on paired watershed data
and generating plots and tables from the analysis.

Requires standard library, numpy, pandas, and matplotlib

"""
import os
import pandas as pd
import numpy as np
from scipy import stats as stats
from matplotlib import pyplot as plt

class olsReg:
    """
    Object to contain regression information for ANCOVA analysis
    
    FUTURE DEV:
        port over to ols regression model and addon additional feats.
    
    self.xvals: 1-d numpy array, containing sample of independant variable
    self.yvals: 1-d numpy array, containing sample of dependant variable
    self.cnt: 1-d numpy array, containing sample of dependant variable
    self.slp: float, slope of linear regression model determined from ANCOVA
    self.yint: float, y intercept of linear regression model determined from ANCOVA
    self.rsq = float, coef of determination for the linear regression mode and yvals
    self.fval = float, f value for the slope of linear regression model
    self.pval = float, p value for the slope of linear regression model
    self.pint = float, predictive interval for specified alpha value
    self.alfa = float, alpha value for the predictive interval
    self.xfrm = bool, True if values were transformed for analysis
    self.nonp = bool, True if data are NOT normal
    self.lmod = lmod, 'redx'
    
   """
    def __init__(self, xvals=None, yvals=None, ypred=None, cnt=0, name='', 
                 slp=0, yint=0, rsq=0, fval=0, pval=0, pint=0, alfa=0, xfrm=0, 
                 nonp=0, lmod='redx'):
        if type(xvals) == None:
            self.xvals = np.zeros([])
            self.yvals = np.zeros([])
            self.ypred = np.zeros([])
            self.cnt = 0
        else:
            self.xvals = xvals
            self.yvals = yvals
            self.cnt = 0
        self.name=''
        self.slp = slp
        self.yint = yint
        self.rsq = rsq
        self.fval = fval
        self.pval = pval
        self.pint = pint
        self.alfa = alfa
        self.xfrm = xfrm
        self.nonp = nonp
        self.lmod = lmod
        
        
def calcTeeSF(t, deg, tail=2, delta=0.01):
    """
    Calculates the survival function for a t value and 
    specified degrees of freedom. 
    
    https://en.wikipedia.org/wiki/Student%27s_t-distribution

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    deg : TYPE
        DESCRIPTION.
    tail : TYPE
        DESCRIPTION.
    delta : TYPE
        DESCRIPTION.
    Returns
    -------
    p : TYPE
        DESCRIPTION.
    
    Change Log:
    11/3/2023: created function - j.roth 
    """
    
    cdf = 0
    i = -10
    
    ## make t negative to reduce numerical integration
    if t > 0:
        t = -t
    
    ## calculate the coefficient for the distribution
    coef = calcTdistCoef(deg)
    
    ## numerically integrate density distrubution to t val
    if i >= t:
        i = t-20.
    while i <= t:
        cdf+=(1+i**2/deg)**(-1*(deg+1)/2)*delta
        i+=delta
    
    ## adjust with deg freedom coefficient
    p = cdf*coef
    
    ## if looking for 2 tailed 
    if tail == 2:
        p *= 2
        
    return p
    
def calcTdistCoef(nu):
    """
    Authoer
    Calculates the coefficient of the t distribution pdf for a given 
    degree of freedom
    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    
    Parameters
    ----------
    nu : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    10/23/2023: created function - j.roth 
    """
    num = 1
    den = 1
    if nu%2 == 0:
        i=1
        while nu-i >= 3:
            num *= (nu-i)
            i+=2
        i=2
        while nu-i>=2: 
            den *= (nu-i)
            i+=2
    else:
        i=1
        while nu-i >= 2:
            num *= (nu-i)
            i+=2
        i=2
        while nu-i>=3: 
            den *= (nu-i)
            i+=2

    if nu%2 == 0:
        coef = 2*np.sqrt(nu)
    else:
        coef = np.pi*np.sqrt(nu)
    
    return num/(coef*den)

def calcRegCoefLinAlg(y, X):
    
    """
    X is of the form dat
    
    """
    
    b = np.matmul(X.T, y)
    c = np.linalg.inv(np.matmul(X.T, X))
    B = np.matmul(c,b)

    return B

def calcSumSqr(x, u=None):
    """
    Calculates the sum of square deviates for 1D array x using u, or mean of x
    
    Parameters
    ----------
    x :1D numpy vector 
        DESCRIPTION. population values
    u : float, optional
        DESCRIPTION. specified value to calculate deviates from. If None
        uses mean of 'vals'

    Returns
    -------
    float
        DESCRIPTION. sum of squared deviates

    ## add ability to specify u as scalar and also check u dims 
    10/23/2023: created function - j.roth 
    """
    ## check u was supplied else use mean of x
    if type(u)==np.ndarray:
        s =  sum([(z[0]-z[1])**2 for z in zip(x, u)])
    elif type(u) == int or type(u) == float:
        sum([(i-u)**2 for i in x])
    else:
        u = x.mean()
        s = sum([(i-u)**2 for i in x])
    return s


def calcRsqr(y, yp):
    """
    calculates the coefficient of determination for 2 samples

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    yp : TYPE
        DESCRIPTION.

    Returns
    -------
    r : TYPE
        DESCRIPTION.
        
    10/23/2023: created function - j.roth     
    """
    t1 = len(y)* sum(y*yp)
    t2 = sum(y)*sum(yp)
    t3 = len(y)*sum([i**2 for i in y])-sum(y)**2
    t4 = len(yp)*sum([i**2 for i in yp])-sum(yp)**2
    r = ((t1-t2)/(np.sqrt(t3*t4)))**2

    return r

def calcStdErr(y, yp, k=0):
    """
    
    Parameters
    ----------
    obs : TYPE
        DESCRIPTION.
    mod : TYPE
        DESCRIPTION.
    k : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.
        
    10/23/2023: created function - j.roth 
    """
    return np.sqrt(calcSumSqr(y,yp)/(y.shape[0]-k))

def calcVar(vals):
    """

    Parameters
    ----------
    vals : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
        
    10/23/2023: created function - j.roth 
    """
    return calcSumSqr(vals)/(vals.shape[0]-1)

def calcStdDev(vals):
    """

    Parameters
    ----------
    vals : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
        
    10/23/2023: created function - j.roth 
    """
    return np.sqrt(calcVar(vals))
    
def calcPredY(X, B):
    """

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
        
    10/23/2023: created function - j.roth 
    """
    if B.shape[0] == 1:
        yp = B[0]*X[:,0]
    elif B.shape[0] == 2:
        yp = B[0]*X[:,0] + B[1]*X[:,1]
    elif B.shape[0] == 3:
        yp = B[0]*X[:,0] + B[1]*X[:,1] + B[2]*X[:,2]
    else:
        yp = B[0]*X[:,0] + B[1]*X[:,1] + B[2]*X[:,2] + B[3]*X[:,3]
            
    return yp

def fTestLinReg(Y, Yp, n):
    """
    https://online.stat.psu.edu/stat501/lesson/6/6.2
    dyi implementation of F test.
    validated with ols and sp lin regress pkgs
    
    FUTURE: only pass y and yp. No need for x, calculate yp before the call. 
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.

    Returns
    -------
    eff : TYPE
        DESCRIPTION.
    pee : TYPE
        DESCRIPTION.
        
    2/25/2023: created function - j.roth 
    """
    
    ## calculate model predictions of dependant variable
    ## 
      
    ## calculate sum of square errors
    mse = calcSumSqr(Y, Yp)/(Y.shape[0]-n)
    
    ## calculate sum of square errors
    msr = (calcSumSqr(Y)-calcSumSqr(Y, Yp))
    
    ## calculate the F-statistic
    f = msr/mse
    
    ## get the corresponding p-value for the F-statistic
    p = 1 - stats.f.cdf(f, 1, Y.shape[0]-n)

    return f, p  

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
    f : TYPE
        DESCRIPTION.
    msg : TYPE
        DESCRIPTION.
    m : TYPE, optional
        DESCRIPTION. The default is 'a+'.

    Returns
    -------
    None.
    
    10/25/2023: created function - j.roth 
    """
    
    with open(f, m) as f:
        f.write(msg)

def getPairedData(ct_data, tr_data, types, phases, date_col, obs_col, 
                  phase_col, min_evt=5):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    prj : TYPE
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
    
    
def removeOutliers(df, par, typ, phs, m=1.5, min_evt=5):
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
    types = df[typ].unique()
    phases = df[phs].unique()
    init_count = df.shape[0]
    
    for t in types:
        for p in phases:
            ## loop through phases
            dat = df[(df[phs]==p) & (df[typ]==t)][par].values
            if dat.shape[0] > 3:
                x25, x75 = np.percentile(dat, [25,75])
        
                x_iqr = x75-x25
                x_lo = x25-x_iqr*m
                x_hi = x75+x_iqr*m
            
            
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
            
    
def transformData(df, par_col, sit_col, phs_col, alpha=0.025, min_pass=2):
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
    
    ## containers to keep tally of best transform
    ## first index is levene's equal variances.
    ## second is shapiro test for each site/phase combo
    raw_tst = [0,0,0,0,0]
    xfm_tst = [0,0,0,0,0]
    
    raw_dat = []
    xfm_dat = []
    
    msg = 2*t+"DATA TRANSFORMATION RESULTS\n"
    
    lev_msg = 3*t+"{0} variances {1} equal, alpha {2:0.2f}\n"
    
    shap_msg = 3*t+"{0} of {1} {2} data sets are normal, alpha {3:0.2f}\n"
    
    raw_msg = ''
    xfm_msg = ''
    ## Loop over sites in the sit_col
    for s in df[sit_col].unique():
        for p in df[phs_col].unique():
            raw_dat.append(df[(df[sit_col]==s) & (df[phs_col]==p)][par_col].values)
            
    for rd in raw_dat:
        xfm_dat.append(np.log(rd))
            
    ## First do a levenes test on both data
    raw_res = stats.levene(raw_dat[0], raw_dat[1], raw_dat[2], 
                                 raw_dat[3])
    
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
        
    ## Set index 

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
           

    if sum(xfm_tst)<5:
        xfm_msg += 3*t+"WARNING transformed data sets not all normal\n"
    if sum(raw_tst)<5:
        raw_msg += 3*t+"WARNING raw data sets not all normal\n"
    
    if sum(xfm_tst) > min_pass or sum(raw_tst) > min_pass:
        if sum(xfm_tst) > sum(raw_tst):
            df[par_col] = np.log(df[par_col].values)
            msg += xfm_msg
            msg += 3*t+'Using Log data values for par: {0}\n'.format(par_col)
            xf = 1
        else:
            msg += raw_msg
            msg += 3*t+'Using raw data values for par : {0}\n'.format(par_col)
            xf = 0
    else:
        msg+= 3*t+"Raw and Log data failed to pass test for normality\n"
        err=1
        xf = 0
    return df, xf, msg, err

def getRegAnova(x, y, yp):
    
    df = pd.DataFrame(index=['Constant', 'Res. Err.', 'Total    '],
                          columns=['   df  ','   ss  ', '   ms  ', '   f  ', '   p  '])
    
    for c in df.columns:
        for r in df.index:
            if c == '   df  ':
                ## calc deg free for all
                if r == 'Constant':
                    df.loc[r, c] = 1
                elif r == 'Res. Err.':
                    df.loc[r, c] = y.shape[0]-2
                else:
                    df.loc[r, c] = y.shape[0]-1
                    
            if c == '   ss  ':
                ## calc deg free for all
                if r == 'Constant':
                    df.loc[r, c] = calcSumSqr(yp, y.mean())
                elif r == 'Res. Err.':
                    df.loc[r, c] = calcSumSqr(y, yp)
                else:
                    df.loc[r, c] = calcSumSqr(y)           
                
            if c == '   ms  ':
                ## calc deg free for all
                if r in ['Constant', 'Res. Err.']:
                    df.loc[r, c] = df['   ss  '][r]/df['   df  '][r]
                else:
                    df.loc[r, c] = ''

            if c == '   f  ':
                ## calc deg free for all
                if r == 'Constant':
                    df.loc[r, c] = df['   ms  '][r]/df['   ms  ']['Res. Err.']
                else:
                    df.loc[r, c] = ''
                    
            if c == '   p  ':
                ## calc deg free for all
                if r == 'Constant':
                    f = df['   f  '][r]
                    dff = df['   df  '][r]
                    dfr = df['   df  ']['Res. Err.']
                    df.loc[r, c] = 1 - stats.f.cdf(f, dff, dfr)
                else:
                    df.loc[r, c] = '' 
    return df


def getModSummary(x, y, yp, p=2):
    
    ## keeps the model fit data
    df = pd.DataFrame(index=['Model Summary'], columns=['  s  ', '  r2  ', 'r2_adj'])
    
    df['  s  ']['Model Summary'] = np.sqrt(calcSumSqr(y,yp)/(y.shape[0]-2))
    df['  r2  ']['Model Summary'] = calcRsqr(y, yp)
    n = y.shape[0]
    df['r2_adj']['Model Summary'] = 1 - (1-df['  r2  ']['Model Summary'])*\
        (n-1)/(n-p-1)
    
    return df

def runCoefTest(X, Y, Yp, B):
    
    
    ## create dataframe to holds stats
    cols = ['val', 'se_coef', 't_val', 'p_val' ]
    df = pd.DataFrame(columns=cols)
    
    ## calculate the mse and dfr.
    mse = calcSumSqr(Y, Yp)/(Y.shape[0]-B.shape[0])
    dfr = Y.shape[0]-B.shape[0]
    i = 0
    for b in B:
        if b == B[0]:
            idx = 'const'
            se = np.sqrt(mse)*\
                np.sqrt(1/Y.shape[0]+(X[:,1].mean()**2/calcSumSqr(X[:,1])))
        else:
            idx = 'coef{0}'.format(i)
            se = np.sqrt(mse)/np.sqrt(calcSumSqr(X[:,1]))

        t = b/se
        p = calcTeeSF(t, dfr)
        
        df = pd.concat([df, pd.DataFrame([[b,se,t,p]],
                                         index=[idx], columns=cols)])
        i+=1
    return df

def runAncova(df, iv_col, dv_col, cov_col, alpha=0.1):
    """
    Performs an ancova analysis on paired event data
    
    """

    ## augment data for intercept and covariant interaction
    df['cons'] = 1
    
    
    df['cv'] = 0
    
    cvs = df[phase_col].unique()
    df.loc[df[cov_col]!= cvs[0],'cv'] = 1
       
    
    df['intx'] = df[iv_col].values*df['cv'].values
    
    ## these are the columns needed for the multi variate regression
    reg_cols = ['cons', iv_col, 'cv', 'intx']
    ## construct matrix of independant variables
    X = np.array(df[reg_cols].values)
    Y = df[dv_col].values
    ## get the coefficients for each of the columns
    B = calcRegCoefLinAlg(Y, X)
    Yp = calcPredY(X, B)
    ## test the coefficients
    cdf = runCoefTest(X, Y, Yp, B)
    
    ## if interaction term is insignificant, but intercepts are different
    if cdf['p_val']['coef3'] > alpha and cdf['p_val']['coef2'] < alpha:
        ## these are the columns needed for the multi variate regression
        reg_cols = ['cons', iv_col, 'cv']
        ## construct matrix of independant variables
        X = np.array(df[reg_cols].values)
        Y = df[dv_col].values
        
        ## get the coefficients for each of the columns
        B = calcRegCoefLinAlg(Y, X)  
        Yp = calcPredY(X, B)
        ## test the coefficients
        cdf = runCoefTest(X, Y, Yp, B)
        if cdf['p_val']['coef2'] < alpha:
            adf = getRegAnova(X[:,1], Y, Yp)
            ## get model fit metrics
            mdf = getModSummary(X[:,1], Y, Yp, B.shape[0])
           
    ## interaction term is significant, need 2 separate regressions
    else:
        ## get the model predictions
        Yp = calcPredY(X, B)
        ## get the anova table
        adf = getRegAnova(X[:,1], Y, Yp)
        ## get model fit metrics
        mdf = getModSummary(X[:,1], Y, Yp, B.shape[0])
    
    ## make 2 regression objects for each paired data set
    
    reg0 = olsReg()
    reg1 = olsReg()
    ## separate data into phases
    reg0.xvals = X[np.where(X[:,2]==0),1][0]
    reg0.yvals = Y[np.where(X[:,2]==0)]
    reg1.xvals = X[np.where(X[:,2]>0),1][0]
    reg1.yvals = Y[np.where(X[:,2]>0)]
    
    reg0.yint = cdf.val[0]
    reg0.slp= cdf.val[1]
    reg1.yint = reg0.yint + cdf.val[2]
 
    ## get the slope for second phase, 
    ## dependant on results of ancova
    if cdf.shape[0] == 3:
        reg1.slp = reg0.slp
    else:
        reg1.slp = reg0.slp + cdf.val[3] 

    ## calculate the predictions and run r2
    reg0.ypred = reg0.xvals*reg0.slp+reg0.yint
    reg0.rsq = calcRsqr(reg0.yvals, reg0.ypred)
    reg0.fval, reg0.pval = fTestLinReg(reg0.yvals, reg0.ypred, n=2)
    reg1.ypred = reg1.xvals*reg1.slp+reg1.yint
    reg1.rsq = calcRsqr(reg1.yvals, reg1.ypred)
    reg1.fval, reg1.pval = fTestLinReg(reg1.yvals, reg1.ypred, n=2)
    
    return reg0, reg1, adf, mdf, cdf
 
def get_max_conf_int(r1, r2):
    """
    Blah blah.
    
    Parameters
    ----------
    r1 : TYPE
        DESCRIPTION.
    r2 : TYPE
        DESCRIPTION.

    Returns
    -------
    ci1 : TYPE
        DESCRIPTION.
    ci2 : TYPE
        DESCRIPTION.
    p_test : TYPE
        DESCRIPTION.

    """
    chk = False

    x_bar = (r1.xvals.sum()+r2.xvals.sum())/\
                            (r1.xvals.shape[0]+r2.xvals.shape[0])

    yp1 = np.array([r1.yint+r1.slp*i for i in r1.xvals])
    yp2 = np.array([r2.yint+r2.slp*i for i in r2.xvals])
    if r1.lmod == 'redx':
        yh1 = r1.yint+r1.slp*x_bar
        yh2 = r2.yint+r2.slp*x_bar                 

        p_test = 0.99
        while chk == False and p_test > 0.01:
            ci1 = calc_conf_int(r1.xvals, r1.yvals, yp1, x_bar, k=2, prob=p_test)
            ci2 = calc_conf_int(r2.xvals, r2.yvals, yp2, x_bar, k=2, prob=p_test)

            if yh1 > yh2:
                if yh1-ci1>yh2+ci2:
                    chk = True
            else:
                if yh1+ci1<yh2-ci2:
                    chk = True
 
            if chk == False:
                p_test-=0.01
    if chk == False or p_test ==0.02:
        ci1=0
        ci2=0
        p_test=0
    return ci1, ci2, p_test

def calc_conf_int(x, y, y_pred, x_test, k=0, prob=0.9):
    """
    Blah.
    ##https://towardsdatascience.com/confidence-interval-vs-prediction-interval-what-is-the-difference-64c45146d47
    
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.
    x_test : TYPE
        DESCRIPTION.
    k : TYPE, optional
        DESCRIPTION. The default is 0.
    prob : TYPE, optional
        DESCRIPTION. The default is 0.9.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    n = x.shape[0]
    sy = calcStdErr(y, y_pred, k)
    sx = calcVar(x)
    cx = np.sqrt(1 + 1/n + (x_test-x.mean())**2/(sx**2))
    t = stats.t.ppf((1-prob)/2, df=y.shape[0]-k-1)
    
    return abs(t*sy*cx)

def plot_ancova(fig, ax, R,  typ_lab, phs_lab, plt_lab, alfa=0.1, 
                plt_mean=True, plt_perr=False, plt_conf=True):
    """
    Blah blak.

    Parameters
    ----------
    R : TYPE
        DESCRIPTION.
    plab : TYPE
        DESCRIPTION.
    alfa : TYPE, optional
        DESCRIPTION. The default is 0.1.
    plt_mean : TYPE, optional
        DESCRIPTION. The default is True.
    plt_perr : TYPE, optional
        DESCRIPTION. The default is False.
    plt_conf : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    ## label for plot legend
    lin_lab="${0}, n:{1}, m:{2:1.1f}, b:{3:1.1f}, r^2:{4:1.2f}, p:{5:1.2f}$"
    
    if R[0].xfrm == 0:            
        ax_lab = "{0} : {1}"
    else:
        ax_lab = "{0} : log[{1}]"

    axlim =  [10000,-10000]
    
    for i in range(len(R)):
        if min(R[i].xvals.min(), R[i].yvals.min())<axlim[0]:
            axlim[0]=min(R[i].xvals.min(), R[i].yvals.min())
        if max(R[i].xvals.max(), R[i].yvals.max())>axlim[1]:
            axlim[1]=max(R[i].xvals.max(), R[i].yvals.max())
    
    for i in range(len(R)):
        clr = "C{0}".format(i)
        
        ax.scatter(R[i].xvals, R[i].yvals, marker='o',c='none', 
                   edgecolors=clr, alpha=0.5, label='_Hidden')
        
        y_pred = np.array([R[i].yint+R[i].slp*x for x in axlim])
    
        phs = phs_lab[i]
            
        lab = lin_lab.format(phs, R[i].xvals.shape[0], R[i].slp, R[i].yint, 
                             R[i].rsq, R[i].pval)
               
        ax.plot(axlim, y_pred, c=clr, label=lab)
        
        if plt_mean == True:
            x_bar = (R[0].xvals.sum()+R[1].xvals.sum())/\
                                    (R[0].xvals.shape[0]+R[1].xvals.shape[0])
                                    
            lsm = R[i].yint+R[i].slp*x_bar
            
            if plt_perr == True:
                
                if i == 0:
                    mu_lab = "Bl_mean: {0:1.2f} +/- {1:1.2f}, CI={2:1.2f}"
                    
                else:
                    mu_lab = "Tx_mean: {0:1.2f} +/- {1:1.2f}, CI={2:1.2f}"
                
                ax.errorbar(x_bar, lsm, R[i].pint, marker='s',
                            color=clr, ecolor=clr,
                            capsize=5, label=mu_lab.format(lsm, 
                                                           R[i].pint, 
                                                           1-R[i].alfa))
            else:
                if i == 0:
                    mu_lab = "Bl_mean: {0:1.2f}"
                else:
                    mu_lab = "Tx_mean: {0:1.2f}"
                    
                ax.scatter(x_bar, lsm, marker='s',
                            color=clr, facecolor=clr,
                            label=mu_lab.format(lsm))

        buff = 0.1*(axlim[1]-axlim[0])

        ax.set_xlim(axlim[0]-buff,axlim[1]+buff)

        ax.set_ylim(axlim[0]-buff, axlim[1]+buff)
                
    if plt_conf:
        c1, c2, p = get_max_conf_int(R[0], R[1])
        ci_lab = "Confidence Interval: {0:1.2f}%".format((p/2.+0.5)*100.)
        
        if R[0].yint > R[1].yint:
            y_conf = np.array([R[0].yint-c1+R[0].slp*x for x in axlim])
            
        else:
            y_conf = np.array([R[0].yint+c1+R[0].slp*x for x in axlim])
            
        ax.plot(axlim, y_conf, c='r', linestyle='dashed', alpha=0.25, 
                label=ci_lab)    
            
    ax.set_aspect='equal'
    ax.tick_params(labelsize='xx-small')
    ax.set_xlabel(ax_lab.format(typ_lab[0], plt_lab), fontsize='small')
    ax.set_ylabel(ax_lab.format(typ_lab[1], plt_lab), fontsize='small')
    ax.set_title("{0} & {1} : {2}".format(R[0].name, R[1].name, plt_lab), 
                 fontsize='small', fontweight='bold')
    ax.legend(loc='upper left', fontsize='small')

    return fig, ax

def printRegStats(anova, model, coef, g, s1, s2, p, outfile):
    """
    simple function to print regression table to a file
    
    """
    spc = 2*'\t'
    wid = 80
    fmt = "{0:3.3f}"
    with open(outfile.format(g, s1, s2), "w") as f:
        
        ## first write the anova summary
        l = "Summary of ANCOVA Results\n"
        l += "Group: {0}, Sites: {1} & {2}, Parameter: {3}\n" +\
            wid*"-"+"\n" +\
            "ANALYSIS OF VARIANCE\nSource"+spc
        f.write(l.format(g, s1, s2, p))
        
        for col in anova.columns:
            f.write(col+spc)
        f.write('\n'+wid*'-'+'\n')
        for idx in anova.index:
            f.write(idx+spc)
            for col in anova.columns:
                if anova[col][idx] != '':
                    f.write(fmt.format(anova[col][idx])+spc)
            f.write('\n')  
        f.write('\n')
            
        ## then write the model summary
        f.write("MODEL SUMMARY\n")
        for col in model.columns:
            f.write(col+spc)
        f.write('\n'+wid*'-'+'\n')
        for idx in model.index:
            for col in model.columns:
                f.write(fmt.format(model[col][idx])+spc)
            f.write('\n')   
        f.write('\n')
        
        ## then write the coefficient summaary
        f.write("MODEL COEFFICIENTS\n")
        f.write(wid*'-'+'\nPredictor'+spc)
        for col in coef.columns:
            f.write(col+spc)
        f.write('\n'+wid*'-'+'\n')
        for idx in coef.index:
            f.write(idx+spc)
            for col in coef.columns:
                f.write(fmt.format(coef[col][idx])+spc)
            f.write('\n')
        f.write('\n')

## BEGIN MAIN #################################################################
if __name__ == "__main__":
    log_file = "log.txt"
    
    outdir = "output"
    
    outdir = os.path.join(os.getcwd(),outdir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    ## name for indvidual anova and regression tables
    reg_tbl_name = "{0}_sites#{1}+{2}_param#{3}_reg-summary.txt"
    
    ## name for individual plots
    plt_name = "{0}_sites#{1}+{2}_param#{3}_plot_ancova.png"
    
    ## name for multi plot file format with group name
    multi_plt_name = "{0}_sites#{1}+{2}_multi-plot_ancova.png"
    
    ## file to read input parameters from
    param_file = 'NRCS_EoF_ANCOVA_par.txt'
    
    ## minimum event count
    min_evt = 5
    
    ## switch to remove outliers
    rem_out = 1
    
    ## columns to include in the output dataframe
    out_cols = ['group', 'ctrl_id', 'trmt_id', 'param', 'phase', 'n', 'x_avg', 
                'y_avg', 'slope', 'slope_pval', 'intercept', 'intercept_pval',
                'r2', 'xform']
    
    ## read in the parameter file data
    params, msg, err = read_params(param_file)
    appendLogFile(log_file, msg, 'w')
    
    if err == 0:
        
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
                
                sites = df[df[group_col]==g][site_col][:].unique()
    
                ## the first site in the group is assumed to be the control
                cs = sites[0]
                    
                tr_sites = sites[1:]
                    
                ## check if there are control and treatment sites
                if tr_sites.shape[0] > 0:
                    
                    ## loop over treatment sites
                    for ts in tr_sites:
                        msg = "Performing analysis for control {0} and treatment {1}\n"
                        appendLogFile(log_file, msg.format(cs, ts))
                        
                        types = df[(df[group_col]==g) & ((df[site_col]==cs) |
                                       (df[site_col]==ts))][type_col].unique()[0:2]
                        
                        phases = df[(df[group_col]==g) & ((df[site_col]==cs) |
                                       (df[site_col]==ts))][phase_col].unique()[0:2]
                        
                        ## loop over parameters and create a paired event dataframe
                        ## for each instance
                        obs = params['data']['val']
                        
                        ## Instantiate the panel plot
                        plt_row_cnt = int((len(obs)+1)/2)
                        ## row column indexes for subplots
                        plt_row = 0
                        plt_col = 0
                        
                        ## ctrl var to determine if a plot is worthy of saving
                        plt_chk = 0
                        
                        ## instantiate figure to put the plots on. 
                        ## Plots will be n x 2 panel. Scale fig size to number of
                        ## rows needed
                        
                        fig_height = max(11, 11/4*plt_row_cnt)
                        
                        multi_fig, multi_axs = plt.subplots(nrows=plt_row_cnt, 
                                                            ncols=2, 
                                                            figsize=[8.5, fig_height], 
                                                            constrained_layout=True)
                        
                        for obs_col in obs:
                            msg = "\tProcessing Observation {0}\n".format(obs_col)
                            appendLogFile(log_file, msg)
                            
                            ## get data for this group of sites
                            pdf = df[(df[group_col]==g) & (df[obs_col].notna())][[
                                site_col, date_col, type_col, phase_col, obs_col]]
                            
                            ## remove outlier data if selected
                            pdf, msg, err = removeOutliers(pdf, obs_col, type_col, 
                                                           phase_col, m=1.5)
                            
                            appendLogFile(log_file, msg)
                        
                            ## check to see if data need transformed
                            pdf, xf, msg, err = transformData(pdf, obs_col, 
                                                              site_col, phase_col, 
                                                              alpha)
                            appendLogFile(log_file, msg)
                            
                            if err == 0:
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
                                
                                ## pair remaining data for final analysis
        
                                pdf, msg, err = getPairedData(ct_data, tr_data, 
                                                              types, phases, 
                                                              date_col, obs_col, 
                                                              phase_col, min_evt=5)
                                appendLogFile(log_file, msg)
                                
                                ## run the ancova analysis
                                if err == 0:
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
                                        
                                ## results have no errors. Plot and save
                                unit = params['units']['val'][obs.index(obs_col)]
                                plt_lab = "{0} ({1})"
                                plt_lab = plt_lab.format(obs_col, unit)
                                fig, ax = plt.subplots()
                                fig, ax = plot_ancova(fig, ax, [r0, r1], types, 
                                                    phases,
                                                    plt_lab, alfa=0.1, 
                                                    plt_mean=False,
                                                    plt_perr=False, 
                                                    plt_conf=False)
                                
                                fig_nam = plt_name.format(g, cs, ts, obs_col)
                                fig_pth = os.path.join(outdir, fig_nam)
                                fig.savefig(fig_pth)
                                 
                                multi_fig, multi_axs[plt_row, plt_col] = plot_ancova(multi_fig, 
                                                                 multi_axs[plt_row, plt_col], 
                                                                 [r0, r1], types, 
                                                                 phases,
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
    
                                if plt_col == 0:
                                    plt_col += 1
                                else:
                                    plt_col = 0
                                    plt_row += 1
                                    
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
                        multi_fig_name = "all_params_ancova.png"
                        multi_fig_pth = os.path.join(outdir, multi_fig_name)
                        multi_fig.savefig(multi_fig_pth)