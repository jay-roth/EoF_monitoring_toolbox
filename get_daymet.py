# -*- coding: utf-8 -*-
"""
@author: Jason.Roth
@title: State Water Management Engineer
@affiliation: USDA-NRCS Minnesota
@email:jason.roth@usda.gov
Created on Wed Mar 14 11:25:06 2018
"""

#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import csv
import time
import datetime as dt
import pandas as pd
import numpy as np
import math

if sys.version_info[0] == 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

class InputError(Exception):
    """Exception raised for errors in the input."""
    pass


def et_hargreaves(tmx, tmn, srad, day=1, sra_cal=False, lat=45.0, Kt=0.162):
    """
    @email:jason.roth@mn.usda.gov
    Calculates ETo via method of Hargreaves and Semani.
    """
    # Kt: use 0.17 for semi-arid, 0.190 for coastal
    
    et_o = np.zeros(tmx.shape[0])
    
    for i in range(tmx.shape[0]):

        if sra_cal:
            # solar constant = 0.0820 MJ m-2 min-1
            Gsc = 0.082
            # inverse relative distance Earth-Sun
            d = min(day[i],365)
            Dr = 1.000 + 0.033 * math.cos(2*math.pi*d/365)
            # latitude [rad]
            theta = lat*math.pi/180
            # solar declination (Equation 24) [rad]
            delta = 0.409 * math.sin(2*math.pi*d/365 - 1.39)
            # sunset hour angle, rad
            Ws = math.acos(-math.tan(theta)*math.tan(delta))
            # extraterrestrial radiation [MJ m-2 day-1]
            Ra = (24*60/math.pi) * Gsc*Dr * (Ws*math.sin(theta) * math.sin(delta) +
                                  math.cos(theta) * math.cos(delta) * math.sin(Ws))
            Ra *= 0.408
            sra = Kt*Ra*(tmx[i] - tmn[i])**0.5
        else:
            sra = srad[i]
        rt = max((tmx[i] + tmn[i])/2 + 17.8, 0)
        et_o[i] = 0.0135 * sra * rt

    return et_o

def calc_gdd(days, min_t, max_t, gdd_base_temp=18, gdd_kill_temp=-2, 
             gdd_kill_days=3, gdd_min_day=120, gdd_max_day=300, temp='C'):
    """
    @email:jason.roth@mn.usda.gov
    calculates growing degree days in celsius
    
    INPUTS:
        days: vector, daynumber,
        min_t, max_t: vector, min and max temps for day
        gdd_base_temp: float, base temp over which to calculate GDDs
        gdd_kill_temp: float, kill temp under which GDDs will not accumulate
        gdd_kill_days: int, number of days under which min temp needs to be 
        gdd_min_date: int, earliest day in year where gdd can accumulate
        gdd_max_data: int, latest day in year where gdd can accumulate
    OUTPUTS:
        tot_gdd: vector, accumulated growing degree days for record
        
    Future dev, change days to datime object
    
    """
    tot_gdd = np.zeros(days.shape[0])
    avg_min_t = 0.0
    gro_chk = 1
    tbar = 0.0
    ## iterate over days
    for i in range(days.shape[0]):
        d = days[i]
        # if beyond the min date and before the max gdd date proceed
        if d >= gdd_min_day and d <= gdd_max_day and gro_chk == 1:
            # calculate mean min temp over past gdd_kill days
            # simple check to make sure we have enough data to calculated
            j = min(i, gdd_kill_days)
            avg_min_t = np.mean(min_t[i-j:i])
            
            # check if above the kill temp
            if avg_min_t > gdd_kill_temp:
                # calculate average temp
                tbar = (min_t[i] + max_t[i]) / 2.0
                # are we above the base temp
                if tbar > gdd_base_temp:
                    # if so accumulate the gdds 
                    tot_gdd[i] = tot_gdd[i-1] + (tbar - gdd_base_temp)
                else:
                    tot_gdd[i] = tot_gdd[i-1]
            else:
                gro_chk = 0
        else:
            if d == 1:
                gro_chk = 1
            elif d > gdd_max_day:
                gro_chk = 0

    return tot_gdd

def daymet_timeseries(lat=36.0133, lon=-84.2625, start_year=2012, end_year=2014,
                      params=["prcp"], verbose=False, fil_nam_pfx=""):
    '''
    Download a Daymet timeseries for a single location as either a local csv or pandas dataframe
    Keyword arguments:
    lat -- geographic latitude of location for timeseries,  must be within Daymet extent
    long -- geographic longitude of location for timeseries,  must be within Daymet extent
    start_yr -- timeseris will begin on January 1st of this year ( >= 1980)
    end_yr -- timeseris will end on December 31st of this year ( < Current year)
    params --, list of parameters, [tmax,tmin,dayl,prcp,srad,swe]
    as_dataframe -- if True return a pandas data frame of the timesereis
                    if False return a local path to the CSV downloaded
    download_dname -- The local directory to save the downloaded csv into
                    if none specified saves the file into the temp workspace
                    returned by tempfile.gettempdir()
    https://github.com/khufkens/daymetpy
    '''
    max_year = dt.datetime.now().year - 1
    MIN_YEAR = 1980  # The begining of the Daymet time series

    if start_year < MIN_YEAR:
        start_year = MIN_YEAR
    if end_year > max_year:
        end_year = max_year

    year_range = ",".join([str(i) for i in range(start_year, end_year+1)])

    # create parameter string
    par_str = ""
    for i in params:
        par_str+="{0},".format(i)
    params=params[:len(params)-1]    
    # create download string / url
    TIMESERIES_URL = ("https://daymet.ornl.gov/data/send/saveData?lat={lat}&" +
                     "lon={lon}&measuredParams={params}&year={year_range}")
    
    timeseries_url = TIMESERIES_URL.format(lat=lat, lon=lon,params=params,
                                           year_range=year_range)

    if verbose:
        print("Daymet webservice URL:\n{}".format(timeseries_url))
        
    # create filename for the output file 
    
    if fil_nam_pfx == "":
        # if a file prefix_is not passed use this default 
        daymet_file = "Daymet_{}_{}_{}_{}.csv".format(lat, lon,
                                                      start_year, end_year)
    else:
        daymet_file = "{}_{}_{}.csv".format(fil_nam_pfx, start_year, 
                                                      end_year)

    if verbose:
        print("File downloaded to:\n{}".format(daymet_file))
        
    if not os.path.exists(daymet_file):
    # download the daymet data (if available)
        urlretrieve(timeseries_url, daymet_file)
    
        if os.path.getsize(daymet_file) == 0:
            os.remove(daymet_file)
            raise NameError("You requested data is outside DAYMET coverage," +
                            "the file is empty --> check coordinates!")

    df = pd.read_csv(daymet_file, header=6)
    df.year = df.year.astype(int)
    df.yday = df.yday.astype(int)
    df.index = pd.to_datetime(df.year.astype(str) + '-' +
                              df.yday.astype(str), format="%Y-%j")
    df.columns = [c[:c.index('(')].strip() if '(' in c else c for c in df.columns]
    return df


def make_water_year(df):
    df['water_year'] = "" 
    df.loc[(df.index.month>=10),'water_year'] = df[(df.index.month>=10)].index.year +1
    df.loc[(df.index.month<10),'water_year'] = df[(df.index.month<10)].index.year
    return df
    

if __name__ == "__main__":

    ###########################################################################
    ## USER INPUTS ############################################################
    ###########################################################################
    
    mm2in = 1.0/25.4
    
    site_file = "fetch_precip_by_station.csv"
    
    event_file = "event_dates_by_station.csv"
    
    id_fld = "nrcs_mon_stat_id"
    
    event_lookback = [3,5,7,14]
    ###########################################################################
    ## USER INPUTS ############################################################
    ###########################################################################
    
    site_df = pd.read_csv(site_file, index_col=id_fld)
    
    event_df = pd.read_csv(event_file, index_col=id_fld)
    
    event_df['date'] = pd.to_datetime(event_df['date'])
    mos = range(1,13)

    annual_cols = [id_fld, "Project ID", "Project Title",
                   "Monitoring Station ID", "year"]
    annual_cols += ["prcp_{0}".format(m) for m in mos]
    annual_cols += ["evap_{0}".format(m) for m in mos]
    annual_cols += ["prec_gs", "evap_gs", "gdd_gs", "prcp_yr", "evap_yr"]
    annual_data = pd.DataFrame(columns=annual_cols)

    event_cols = ["prcp_t", "evap_t", "gdd_t"]
    event_cols += ["prcp_sub_{0}".format(lb) for lb in event_lookback]
    event_cols += ["evap_sub_{0}".format(lb) for lb in event_lookback]
    no_event = []
    
    for ec in event_cols:
        event_df.loc[:, ec] = 0

    for i in site_df.index:
        by =int(site_df.loc[i,'Contract Beg Yr'])
        ey = int(site_df.loc[i,'Contract End Yr'])
        lat = site_df.loc[i,'Lat']
        lon = site_df.loc[i,'Lon']
        
        prj = site_df.loc[i,'Project ID']
        ttl = site_df.loc[i,'Project Title']
        dbid = site_df.loc[i,"Monitoring Station ID"]
        
        data = daymet_timeseries(lat=lat, lon=lon, start_year=by,
                                 end_year=ey, params=["prcp"], 
                                 verbose=False, fil_nam_pfx=i)
        
        #2.) convert srad to values of mm/d using latent heat of vaporization
        data.loc[:, 'srad'] = (data['srad'][:] * data['dayl'][:])/(10**6*2.454)
        
        data['gdd'] = calc_gdd(data['yday'], data['tmin'], data['tmax'],
                        gdd_base_temp=18, gdd_kill_temp=-2, gdd_kill_days=3,
                        gdd_min_day=100, gdd_max_day=330, temp='C')

        data['evap'] = et_hargreaves(data['tmax'], data['tmin'], data['srad'], 
                                     day=data['yday'],sra_cal=False,
                                     lat=site_df.loc[i,'Lat'], Kt=0.162)
        
        ## calculate annual and monthly statistcs
        for y in range(by, min(ey+1, 2022)):
            site_yr_data = [i, prj, ttl, dbid, y]
            for m in range(1, 13):
                site_yr_data.append(
                    data[(data.year==y) & (data.index.month==m)]['prcp'].sum()*mm2in)
            for m in range(1, 13):
                site_yr_data.append(
                    data[(data.year==y) & (data.index.month==m)]['evap'].sum()*mm2in)
            site_yr_data.append(data[(data.year==y) & (data.gdd>0)]['prcp'].sum()*mm2in)
            site_yr_data.append(data[(data.year==y) & (data.gdd>0)]['evap'].sum()*mm2in)
            site_yr_data.append(data[(data.year==y)]['gdd'].max())
            site_yr_data.append(data[(data.year==y)]['prcp'].sum()*mm2in)
            site_yr_data.append(data[(data.year==y)]['evap'].sum()*mm2in)
            new_data = pd.DataFrame([site_yr_data], columns=annual_cols)
            annual_data = annual_data.append(new_data)
        
        ## get the events associated with this site.
        site_events = event_df.loc[event_df.index==i]
        if site_events.shape[0] > 0 and i != '0':
            for d in site_events['date']:
                if d in data.index:
                    event_df.loc[(event_df.index==i) & (event_df.date==d),'prcp_t'] =\
                        data[(data.index==d)]['prcp'][0]*mm2in
                    event_df.loc[(event_df.index==i) & (event_df.date==d),'evap_t'] =\
                        data[(data.index==d)]['evap'][0]*mm2in
                    event_df.loc[(event_df.index==i) & (event_df.date==d),'gdd_t'] =\
                        data[(data.index==d)]['gdd'][0]
                for elb in event_lookback:
                    event_df.loc[(event_df.index==i) & (event_df.date==d),'prcp_sub_{0}'.format(elb)] =\
                                data[(data.index>=d - dt.timedelta(elb)) & (data.index<d)]['prcp'].sum()*mm2in
                    event_df.loc[(event_df.index==i) & (event_df.date==d),'evap_sub_{0}'.format(elb)] =\
                                data[(data.index>=d - dt.timedelta(elb)) & (data.index<d)]['evap'].sum()*mm2in
        else:
            no_event.append(i)
