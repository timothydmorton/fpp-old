import numpy as np
import pandas as pd
import gc
import os,re,sys,os.path

import numpy.random as rand

from scipy.stats import gaussian_kde
import scipy.stats as stats

import distributions as dists
#from utils import kde

try:
    import koiutils as ku
except ImportError:
    print 'warning: koiutils not imported.'

DF_FILE = os.environ['BASTIFILE']

DF = pd.read_hdf(DF_FILE,'table')
DF['teff'] = 10**DF['logteff']
DF['age'] = 10**DF['logage']
#DF['feh'] = DF['feh']*100
#DF.set_index('feh',inplace=True)

FEH_INDICES = np.arange(-1.7,0.4,0.02)

def subregion(fehrange,teffrange,loggrange):
    f1,f2 = fehrange
    t1,t2 = teffrange
    g1,g2 = loggrange
    query = '(%.2f <= feh <= %.2f) and (%.4f <= teff <= %.4f) and (%.4f <= logg <= %.4f)' %\
        (f1,f2,t1,t2,g1,g2)
    return DF.query(query)


def find_closest(df,fehs,teffs,loggs,maxsize=1e8):

    Ntot = len(fehs)*len(df)
    if Ntot > maxsize:
        #split into chunks
        max_chunksize = int(maxsize / len(fehs))
        close_df = pd.DataFrame()
        nchunks = len(df)/max_chunksize + 1
        chunksize = len(fehs)/nchunks
        print 'splitting into %i chunks...' % nchunks
        for i in np.arange(nchunks):
            print i+1
            lo = int(i*chunksize)
            hi = int((i+1)*chunksize)
            subdf = df[i::nchunks]
            teffnorm = subdf['teff'].mad()
            fehnorm = subdf['feh'].mad()
            loggnorm = subdf['logg'].mad()
            diffs = (((subdf['teff'][:,np.newaxis] - teffs[lo:hi])/teffnorm)**2 + 
                     ((subdf['feh'][:,np.newaxis] - fehs[lo:hi])/fehnorm)**2 +
                     ((subdf['logg'][:,np.newaxis] - loggs[lo:hi])/loggnorm)**2)
            inds = diffs.argmin(axis=0)
            close_df = close_df.append(df[i::nchunks].iloc[inds])

    else:
        teffnorm = df['teff'].mad()
        fehnorm = df['feh'].mad()
        loggnorm = df['logg'].mad()
        diffs = (((df['teff'][:,np.newaxis] - teffs)/teffnorm)**2 + 
                 ((df['feh'][:,np.newaxis] - fehs)/fehnorm)**2 +
                 ((df['logg'][:,np.newaxis] - loggs)/loggnorm)**2)
        inds = diffs.argmin(axis=0)
        close_df = df.iloc[inds]

    return close_df.copy()

def props_dist(fehdist,teffdist,loggdist,N=1000,return_samples=False,
               return_kde=False,return_df=False,return_df_all=False,
               m_bw=0.02,r_bw=0.02,age_bw=0.02,
               resample_factor=1):
    if type(fehdist)==type((1,2)):
        fehdist = stats.norm(*fehdist)
    if type(teffdist)==type((1,2)):
        teffdist = stats.norm(*teffdist)
    if type(loggdist)==type((1,2)):
        loggdist = stats.norm(*loggdist)
    

    fehs = fehdist.rvs(N)
    teffs = teffdist.rvs(N)
    loggs = loggdist.rvs(N)

    fehrange = (fehdist.ppf(0.01),fehdist.ppf(0.99))
    teffrange = (teffdist.ppf(0.01),teffdist.ppf(0.99))
    loggrange = (loggdist.ppf(0.01),loggdist.ppf(0.99))

    df = subregion(fehrange,teffrange,loggrange)

    closedf = find_closest(df,fehs,teffs,loggs)
    gc.collect()
    if return_samples:
        inds = rand.randint(N,size=N*resample_factor)
        return (np.array(closedf['mass'])[inds],
                np.array(closedf['radius'])[inds],
                np.array(closedf['logage'])[inds])
    elif return_df_all:
        return closedf.reset_index()
    elif return_df:
        return closedf[['mass','radius','logage']].reset_index()
    elif return_kde:
        return (dists.KDE_Distribution(np.array(closedf['mass']),
                                       adaptive=False,bandwidth=m_bw),
                dists.KDE_Distribution(np.array(closedf['radius']),
                                       adaptive=False,bandwidth=r_bw),
                dists.KDE_Distribution(np.array(closedf['logage']),minval=0,
                                       adaptive=False,bandwidth=age_bw))
    else:
        return (dists.fit_doublegauss_samples(np.array(closedf['mass'])),
                dists.fit_doublegauss_samples(np.array(closedf['radius'])),
                dists.fit_doublegauss_samples(np.array(closedf['logage'])))
    

def props_list(star_df,outfolder='starprops',kepler=True,overwrite=False):
    """input DataFrame must have the following columns:
    
         name,teff,e_teff,logg,e_logg,feh,e_feh
    """

    if type(star_df) == type(''):
        star_df = pd.read_table(star_df,delimiter='\s+')

    if kepler:
        star_df['name'] = star_df['name'].apply(ku.koiname,star=True)

    for i in np.arange(len(star_df)):
        name = star_df['name'].iloc[i]
        outfile = '%s/%s.h5' % (outfolder,name)
        if os.path.exists(outfile) and not overwrite:
            continue
        
        fehdist = dists.Gaussian_Distribution(star_df['feh'].iloc[i],
                                              star_df['e_feh'].iloc[i],
                                              name='[Fe/H]')
        teffdist = dists.Gaussian_Distribution(star_df['teff'].iloc[i],
                                               star_df['e_teff'].iloc[i],
                                               name='Teff')
        loggdist = dists.Gaussian_Distribution(star_df['logg'].iloc[i],
                                               star_df['e_logg'].iloc[i],
                                               name='log(g)')
        print name,fehdist,teffdist,loggdist
        try:
            df = props_dist(fehdist,teffdist,loggdist,return_df=True)
            df.to_hdf(outfile,'df')
            print '%i of %i done: %s' % (i,len(star_df),name)
        except KeyboardInterrupt:
            raise
        except:
            print 'error with',name

    
    

gc.collect()
