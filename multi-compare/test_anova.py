# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:57:01 2023

@author: Jason.Roth
@email: jason.roth@usda.gov
@position:Env.Engr, WQQT
@agency: USDA-NRCS

Description:

1.) read input file
2.) perform operations: multiplication and scaling
3.) loop through params
4.) remove outliers if spec'd
5.) check for normality (transform, untransformed)
6.) Perform parametric, non-parametric stats 
7.) Generate specified plots and tables
8.) dump output file
    



"""
import pandas as pd
import numpy as np
import os
from scipy import stats
from collections import namedtuple
import warnings


F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))


def _create_f_oneway_nan_result(shape, axis):
    """
    This is a helper function for f_oneway for creating the return values
    in certain degenerate conditions.  It creates return values that are
    all nan with the appropriate shape for the given `shape` and `axis`.
    """
    axis = np.core.multiarray.normalize_axis_index(axis, len(shape))
    shp = shape[:axis] + shape[axis+1:]
    if shp == ():
        f = np.nan
        prob = np.nan
    else:
        f = np.full(shp, fill_value=np.nan)
        prob = f.copy()
    return F_onewayResult(f, prob)

def _first(arr, axis):
    """Return arr[..., 0:1, ...] where 0:1 is in the `axis` position."""
    return np.take_along_axis(arr, np.array(0, ndmin=arr.ndim), axis)


def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)

    return a, outaxis

def _sum_of_squares(a, axis=0):
    """Square each element of the input array, and return the sum(s) of that.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis along which to calculate. Default is 0. If None, compute over
        the whole array `a`.

    Returns
    -------
    sum_of_squares : ndarray
        The sum along the given axis for (a**2).

    See Also
    --------
    _square_of_sums : The square(s) of the sum(s) (the opposite of
        `_sum_of_squares`).

    """
    a, axis = _chk_asarray(a, axis)
    return np.sum(a*a, axis)

def _square_of_sums(a, axis=0):
    """Sum elements of the input array, and return the square(s) of that sum.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis along which to calculate. Default is 0. If None, compute over
        the whole array `a`.

    Returns
    -------
    square_of_sums : float or ndarray
        The square of the sum over `axis`.

    See Also
    --------
    _sum_of_squares : The sum of squares (the opposite of `square_of_sums`).

    """
    a, axis = _chk_asarray(a, axis)
    s = np.sum(a, axis)
    if not np.isscalar(s):
        return s.astype(float) * s
    else:
        return float(s) * s

def f_oneway(*samples, axis=0):
    """
    Performs one-way ANOVA. 
    Adopted from 
    https://github.com/scipy/scipy/blob/v1.11.4/scipy/stats/_stats_py.py#L3985-L4217
    by J. Roth 12/27/2023 
    

    The one-way ANOVA tests the null hypothesis that two or more groups have
    the same population mean.  The test is applied to samples from two or
    more groups, possibly with differing sizes.

    Parameters
    ----------
    data frame
        The sample measurements for each group.  There must be at least
        two arguments.  If the arrays are multidimensional, then all the
        dimensions of the array must be the same except for `axis`.
    axis : int, optional
        Axis of the input arrays along which the test is applied.
        Default is 0.

    Returns
    -------
    statistic : float
        The computed F statistic of the test.
    pvalue : float
        The associated p-value from the F distribution.

    """
    if len(samples) < 2:
        raise TypeError('at least two inputs are required;'
                        f' got {len(samples)}.')

    samples = [np.asarray(sample, dtype=float) for sample in samples]

    ## ANOVA on N groups, each in its own array
    num_groups = len(samples)

    ## We haven't explicitly validated axis, but if it is bad, this call of
    ## np.concatenate will raise np.AxisError.  The call will raise ValueError
    ## if the dimensions of all the arrays, except the axis dimension, are not
    ## the same.
    alldata = np.concatenate(samples, axis=axis)
    bign = alldata.shape[axis]

    ## Check this after forming alldata, so shape errors are detected
    ## and reported before checking for 0 length inputs.
    if any(sample.shape[axis] == 0 for sample in samples):
        warnings.warn(stats.DegenerateDataWarning('at least one input '
                                                  'has length 0'))
        return _create_f_oneway_nan_result(alldata.shape, axis)
    ## Must have at least one group with length greater than 1.
    if all(sample.shape[axis] == 1 for sample in samples):
        msg = ('all input arrays have length 1.  f_oneway requires that at '
               'least one input has length greater than 1.')
        warnings.warn(stats.DegenerateDataWarning(msg))
        return _create_f_oneway_nan_result(alldata.shape, axis)
    
    ## Check if all values within each group are identical, and if the common
    ## value in at least one group is different from that in another group.
    ## Based on https://github.com/scipy/scipy/issues/11669

    ## If axis=0, say, and the groups have shape (n0, ...), (n1, ...), ...,
    ## then is_const is a boolean array with shape (num_groups, ...).
    ## It is True if the values within the groups along the axis slice are
    ## identical. In the typical case where each input array is 1-d, is_const is
    ## a 1-d array with length num_groups.
    is_const = np.concatenate(
        [(_first(sample, axis) == sample).all(axis=axis,
                                              keepdims=True)
         for sample in samples],
        axis=axis
    )

    ## all_const is a boolean array with shape (...) (see previous comment).
    ## It is True if the values within each group along the axis slice are
    ## the same (e.g. [[3, 3, 3], [5, 5, 5, 5], [4, 4, 4]]).
    all_const = is_const.all(axis=axis)
    
    if all_const.any():
        msg = ("Each of the input arrays is constant;"
               "the F statistic is not defined or infinite")
        warnings.warn(stats.ConstantInputWarning(msg))

    ## all_same_const is True if all the values in the groups along the axis=0
    ## slice are the same (e.g. [[3, 3, 3], [3, 3, 3, 3], [3, 3, 3]]).
    all_same_const = (_first(alldata, axis) == alldata).all(axis=axis)

    ## Determine the mean of the data, and subtract that from all inputs to a
    ## variance (via sum_of_sq / sq_of_sum) calculation.  Variance is invariant
    ## to a shift in location, and centering all data around zero vastly
    ## improves numerical stability.
    offset = alldata.mean(axis=axis, keepdims=True)
    alldata -= offset

    normalized_ss = _square_of_sums(alldata, axis=axis) / bign

    sstot = _sum_of_squares(alldata, axis=axis) - normalized_ss

    ssbn = 0
    for sample in samples:
        ssbn += _square_of_sums(sample - offset,
                                axis=axis) / sample.shape[axis]

    ## Naming: variables ending in bn/b are for "between treatments", wn/w are
    ## for "within treatments"
    ssbn -= normalized_ss
    sswn = sstot - ssbn
    dfbn = num_groups - 1
    dfwn = bign - num_groups
    msb = ssbn / dfbn
    msw = sswn / dfwn
    with np.errstate(divide='ignore', invalid='ignore'):
        f = msb / msw
    
    prob = stats.special.fdtrc(dfbn, dfwn, f)   # equivalent to stats.f.sf

    ## Fix any f values that should be inf or nan because the corresponding
    ## inputs were constant.
    if np.isscalar(f):
        if all_same_const:
            f = np.nan
            prob = np.nan
        elif all_const:
            f = np.inf
            prob = 0.0
    else:
        f[all_const] = np.inf
        prob[all_const] = 0.0
        f[all_same_const] = np.nan
        prob[all_same_const] = np.nan

    return F_onewayResult(f, prob)


def get_sig_difs(df, a):
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
        if sum(df.loc[:,c]==1) > prv_cnt:
            grp_num += 1
            prv_cnt = sum(df.loc[:,c]==1)
            df_grps.loc[df.loc[:,c]==1,c] = grp_lab[grp_num]
            
    for c in df_grps.columns[1:]:
        df_grps.loc[:,0]+=df_grps.loc[:,c]
            
    return df_grps[0]

cwd=os.getcwd()

orig_dat = pd.read_csv(os.path.join(cwd,'data.csv')) 

a = 0.05

## sort the data columns by means, greatest to least
sort_dat = orig_dat[orig_dat.mean().sort_values(ascending=False).index]

z = [sort_dat[c].tolist() for c in sort_dat.columns]

for i in range(len(z)):
    z[i] = [j for j in z[i] if not np.isnan(j)]

r = stats.f_oneway(*z)

if r[1] < a:
    ## parametric or non-parametric check
    g = stats.tukey_hsd(*z)
    mc_pvals = pd.DataFrame(g.pvalue)
    mc_difs = get_sig_difs(mc_pvals.copy(), a)   
    mc_grps = get_comp_grps(mc_difs.copy())
    mc_grps = pd.Series(mc_grps.values,sort_dat.columns)
    grps = pd.DataFrame([mc_grps[orig_dat.columns].tolist()], 
                           columns=orig_dat.columns)
    