from __future__ import print_function,division
import os,sys,re,os.path,shutil,fnmatch
import numpy as np
from progressbar import Percentage,Bar,RotatingMarker,ETA,ProgressBar
import atpy
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('pylab not imported.')
import logging
import h5py
import pandas as pd
import traceback
import numpy.random as rand

from matplotlib import mlab
from numpy import ma
#import qalg,tval,keptoy
import plotutils as pu
import pickle
import glob
from consts import *

import transitFPP as fpp
#import parsetable

from scipy.integrate import quad

import koiutils as ku
import contrastcurves as ccs
import koi_imaging as kim

import utils


KEPLERDIR = os.environ['KEPLERDIR']
#KEPDATADIR = '%s/data/Batalha12' % KEPLERDIR
KEPDATADIR = '%s/data/pk' % KEPLERDIR
CHAINSDIR = '%s/data/chains' % KEPLERDIR
ASCIIDATADIR = '%s/data/ascii' % KEPLERDIR
FPMODELSDIR = '%s/FPP/models' % KEPLERDIR 
KOIDATAFILE = os.environ['KOIDATAFILE']

ROWEFOLDER = '%s/data/rowe' % KEPLERDIR
WEAKSECFILE = '%s/data/weakSecondary_socv9p2vv.csv' % KEPLERDIR

WEAKSECDATA = pd.read_csv(WEAKSECFILE,skiprows=8)
WEAKSECDATA.index = WEAKSECDATA['KOI'].apply(ku.koiname)

#KOIDATA = parsetable.parseKOItable(KOIDATAFILE)
#KOIDATA = parsetable.csv2dict(KOIDATAFILE,'kepoi_name',names=True)
#parsetable.correct_KICmags(KOIDATA)
KOIDATA = ku.DATA

#TTVDATA = pd.read_hdf('%s/ttvs_q1q12.h5' % os.environ['FPPDIR'],'table')
#TTVMAXQ = 12

MAXQ = 15

CDPP3 = np.loadtxt('%s/cdpp/cdpp3.txt' % KEPLERDIR,skiprows=1)
#TARGETKIC = np.recfromtxt('%s/targetkic.txt' % KEPLERDIR,names=True,
#                       usecols=(0,6,7,8,9,12,13,14,15,23,24,25,28),delimiter='|')
COMMENTFILE = '%s/koinotes.csv' % KEPLERDIR
COMMENTDATA = np.recfromcsv(COMMENTFILE,names=True)
KOICOMMENTS = {}

for k,c in zip(COMMENTDATA.koi,COMMENTDATA.comment):
    KOICOMMENTS['KOI%.2f' % k] = c


#obselete?
def make_trapfit_table(outfile='%s/trapfit.txt' % KEPLERDIR,refit=False):
    
    fout = open(outfile,'w')
    fout.write('KOI depth depthlo depthhi dur durlo durhi slope slopelo slopehi\n')

    ferr = open(outfile+'.errorlog','w')
    
    n = len(KOIDATA.keys())
    for i,k in enumerate(KOIDATA.keys()):
        try:
            if refit:
                sig = KeplerTransitsignal(k,mcmc=False)
                sig.MCMC(refit=True)
            else:
                sig = KeplerTransitsignal(k)

            #get star data from the right place!
            #a = (((sig.P*DAY/(2*np.pi))**2 * G* coolkoi_stardata[k]['M']*MSUN)**(1./3))
            #v = 2*np.pi*a / (sig.P*DAY)
            #Tc = 2*coolkoi_stardata[k]['R']*RSUN/v/DAY*24

            depth = sig.depthfit[0]*1e6
            depthlo = depth - sig.depthfit[1][0]*1e6
            depthhi = depth + sig.depthfit[1][1]*1e6
            dur = sig.durfit[0]*24
            durlo = dur - sig.durfit[1][0]*24
            durhi = dur + sig.durfit[1][1]*24
            slope = sig.slopefit[0]
            slopelo = slope - sig.slopefit[1][0]
            slopehi = slope + sig.slopefit[1][1]



            line = '%s %i %i %i %.2f %.2f %.2f %.2f %.2f %.2f\n' %\
                (k,depth,depthlo,depthhi,dur,durlo,durhi,slope,slopelo,slopehi)
            sys.stdout.write('%i of %i: %s' % (i+1,n,line))
            fout.write(line)
        except Exception,e:
            ferr.write('%s: %s\n' % (k,e))
            print('skipping %s: %s' % (k,e))
    fout.close()
        
def days_observed(kic,maxq=MAXQ):
    days = np.ones(maxq)*90.  
    days[0] = 33.  
    w = np.where(CDPP3[:,0]==kic)[0]
    row = CDPP3[w,1:maxq+1]
    return (days*((row > 0) & (row < np.inf))).sum()

def median_CDPP(kic,maxq=MAXQ):
    w = np.where(CDPP3[:,0]==kic)[0]
    row = CDPP3[w,1:maxq+1].copy()
    row = row[np.where((row < np.inf) & (row > 0))]
    return np.median(row)

#obselete?
def calc_Nstars(kois=KOIDATA.keys(),kics=None,outfile=None,**kwargs):
    Ns = {}
    if outfile is not None:
        fout = open(outfile,'w')
    if type(kois) not in [type([]),type(())]:
        kois = [kois]
    for k in kois:
        try:
            Ns[k] = len(which_observable(k,kics=kics,**kwargs))
            if outfile is not None:
                line = '%s %i\n' % (k,Ns[k])
                sys.stdout.write(line)
                fout.write(line)
        except KeyError,e:
            logging.warning('%s not in KOIDATA' % e)
    if outfile is not None:
        fout.close()
    return Ns

def transitSNR(Ntr,Npts,delta,sigma):
    return (Ntr*Npts)**0.5*(delta/sigma)

def SNRrampfn(lo=6,hi=16):
    return lambda s: SNRramp(s,lo,hi)

def SNRramp(snr,lo=6,hi=16):
    snr = np.atleast_1d(snr)
    probs = np.ones(snr.shape)
    probs[np.where(snr < lo)] = 0
    w = np.where((snr >= lo) & (snr <= hi))
    probs[w] = (snr[w]-lo)/(hi-lo)
    if np.size(probs)==1:
        probs = probs[0]
    return probs

def which_observable(koi,thresh=7,kics=None,maxq=MAXQ):
    """returns a list of target stars KIC numbers around which SNR > thresh"""
    data = KOIDATA[koi]
    Npts = data['duration']/3.
    #avgdays = 1./maxq * ((maxq-1)*90 + 33) #q1 is 33 days
    #Ntr = avgdays/data['period']  

    #add correction for planet radius, star radius...?

    days = np.ones(maxq)*90.  
    days[0] = 33.  
    Ntr = days/data['period']

    cdpps = CDPP3[:,1:maxq+1]
    cdpps[np.where(np.isnan(cdpps) | (cdpps==0))] = np.inf

    SNRs = transitSNR(Ntr,Npts,data['depth'],CDPP3[:,1:maxq+1])
    SNRs[np.where(np.isnan(SNRs))] = 0
    SNRtots = np.sqrt((SNRs**2).sum(axis=1))
    wok = np.where(SNRtots > thresh)[0]
    kic_observable = CDPP3[wok,0].astype(int)
    if kics is not None:
        s1 = set(kics)
        s2 = set(kic_observable)
        return np.array(list(s1.intersection(s2)))
    else:
        return kic_observable

def period_correct(koi,pdist,maxq=MAXQ,maxper=300,minper=0.5):
    snr = totalSNR(koi,maxq)
    P = KOIDATA[koi]['koi_period']
    pdist_norm = quad(pdist,np.log10(minper),np.log10(maxper))[0]

    def snr_of_p(per):
        return snr*(per/P)**(-1./3)  #scaling of SNR with period

    def prod(logper):
        return SNRramp(snr_of_p(10**logper))*pdist(logper)/pdist_norm

    return quad(prod,np.log10(minper),np.log10(maxper))[0]


def Pmax_detect(koi,maxq=MAXQ,thresh=7.1):
    snr,cdpps = totalSNR(koi,maxq,return_cdpps=True)

    days = np.ones(maxq)*90.
    days[0] = 33.  

    days[np.where(np.isinf(cdpps))] = 0
    totdays = days.sum()
    
    return snr**2*totdays/thresh**2


def totalSNR(koi,maxq=MAXQ,return_cdpps=False):
    data = KOIDATA[koi]
    kic = data['kepid']
    i = np.where(CDPP3[:,0] == kic)
    cdpps = CDPP3[i,1:maxq+1]
    cdpps[np.where(np.isnan(cdpps))] = np.inf
    cdpps[np.where(cdpps==0)] = np.inf
    #cdpps = cdpps[np.where(~np.isnan(cdpps) & (cdpps != 0))]
    Npts = data['koi_duration']/3.

    #most quarters have 90 days, Q1 has 33
    days = np.ones(maxq)*90.  
    days[0] = 33.  
    Ntr = days/data['koi_period']

    #avgdays = 1./maxq * ((maxq-1)*90 + 33) #q1 is 33 days
    #Ntr = avgdays/data['period']
    d = data['koi_depth']
    if d==0:
        d = data['koi_ror']**2 * 1e6
    SNRs = transitSNR(Ntr,Npts,d,cdpps)
    snrtot = np.sqrt((SNRs**2).sum())
    if return_cdpps:
        return snrtot,np.squeeze(cdpps)
    else:
        return snrtot


def calc_secthresh(koi,maxq=MAXQ,thresh=10.):
    d = KOIDATA[koi]['koi_depth']
    if d==0:
        d = KOIDATA[koi]['koi_ror']**2 *1e6
    return thresh/totalSNR(koi,maxq) * d * 1e-6

def rpdist_toy(alpha1=-1,alpha2=-3,rbreak=2,norm=0.75):
    return utils.broken_powerlaw(alpha1,alpha2,rbreak,norm=norm)
    

def kepler_fp_toy(rp,dr=None,alpha1=-1,alpha2=-3,rbreak=2,norm=0.75):
    plaw = rpdist_toy(alpha1,alpha2,rbreak,norm)
    if dr is None:
        dr = rp/3.
    return quad(plaw,rp-dr,rp+dr)[0]

def kepler_fp_fressin(rp,dr=None):
    if dr is None:
        dr = rp/3.
    return quad(fressin_occurrence,rp-dr,rp+dr)[0]

def fressin_occurrence(rp):
    rp = np.atleast_1d(rp)
    #bins = np.array([0.5,0.8,1.25,2,4,6,22]) #added extra bin from 0.5, same as next
    #rates = np.array([0,0.184,0.184,0.23,
    #                  0.235,0.0197,0.02,0.]) 

    sq2 = np.sqrt(2)
    bins = np.array([1/sq2,1,sq2,2,2*sq2,
                     4,4*sq2,8,8*sq2,
                     16,16*sq2])
    rates = np.array([0,0.155,0.155,0.165,0.17,0.065,0.02,0.01,0.012,0.01,0.002,0])

    return rates[np.digitize(rp,bins)]


def kepler_fp(rp,dr=None,N=1.6e5,kois=None):
    if dr is None:
        dr = rp/3.
    rlo = rp-dr
    rhi = rp+dr
    if kois is None:
        kois = ku.KOIDATA.keys()
    tot = 0
    for k in kois:
        try:
            if KOIDATA[k]['koi_disposition'] not in ('CANDIDATE','CONFIRMED') and\
                KOIDATA[k]['koi_pdisposition'] not in ('CANDIDATE','CONFIRMED'):
                continue
            if KOIDATA[k]['koi_prad'] > rlo and KOIDATA[k]['koi_prad'] < rhi\
                    and np.nan not in (KOIDATA[k]['koi_srad'],):
                smass = 10**KOIDATA[k]['koi_slogg']*(KOIDATA[k]['koi_srad']*RSUN)**2 / G / MSUN
                if np.isnan(smass):
                    smass = KOIDATA[k]['koi_srad']
                a = ((KOIDATA[k]['koi_period']*DAY)**2/(2*np.pi)**2 * G * smass*MSUN)**(1./3)
                ptrans = (KOIDATA[k]['koi_srad']*RSUN + KOIDATA[k]['koi_prad']*REARTH)/a
                tot += 1./ptrans
        except KeyError:
            pass
    fp = tot/N
    if rp < 2 and fp < 0.1:
        fp = 0.1
    return fp


def KOIcc(koi):
    return ccs.KOIcc(koi)

def KOI_imagedata(koi):
    return kim.all_imagedata(koi)


def generic_cc(mag=10,dmag=8,band='K'):
    """Returns a generic contrast curve.

    Keyword arguments:
    mag -- magnitude of target star in passband
    dmag -- can currently be either 8 or 4.5 (two example generic cc's being used)
    band -- passband of observation.

    """
    if dmag==8:
        return fpp.ContrastCurveFromFile('%s/data/contrast_curves/ex8_K.txt' % KEPLERDIR,band,mag)
    elif dmag==4.5:
        return fpp.ContrastCurveFromFile('%s/data/contrast_curves/ex4.5_K.txt' % KEPLERDIR,band,mag)


def KOIcc_old(koi):  #This needs to be updated; some hacks in here still re: dealing w/ multiple contrast curves...
    """Returns all contrast curves available for a given KOI (returns list if more than one)

    Positional arguments:
    koi -- koi name

    """
    if type(koi) == type(1.):
        koi = 'KOI%d' % koi
    m = re.search('KOI(\d+)\.(\d+)',koi)
    if m:
        koi = 'KOI'+m.group(1)
    
    ccdir = '%s/data/contrast_curves/%s' % (KEPLERDIR,koi)
    try:
        ccfiles = os.listdir(ccdir)
    except OSError:
        logging.warning('directory %s does not exist?' % ccdir)
        return None
    cc = []
    for f in ccfiles:
        m = re.search('^\w+\S*_(\w+)\.txt',f)
        if m:
            band = m.group(1)
            mag = KICmag(koi,band)
            fname = '%s/%s' % (ccdir,f)
            cc.append(fpp.ContrastCurveFromFile(fname,band,mag))
            print('using contrast curve from %s' % (fname))
            logging.info('using contrast curve from %s' % (fname))
    if len(cc)==0:
        logging.info('no contrast curve files in %s' % ccdir)
        return None
    if len(cc)==1:
        return cc[0]
    else:
        return cc

def KICmags(koi,bands=['g','r','i','z','j','h','k','kep']):
    return ku.KICmags(koi)
    #mags = {}
    #for b in bands:
    #    mags[b] = KOIDATA[koi]['koi_%smag' % b]
    #mags['J'] = mags['j']
    #mags['Ks'] = mags['k']
    #mags['H'] = mags['h']
    #mags['Kepler'] = mags['kep']
    #return mags

def KICmag(koi,band):
    return ku.KICmags(koi)[band]
    #if type(koi) == type(1.):
    #    koi = 'KOI%.2f' % koi
    #m = re.search('KOI\d+$',koi)
    #if m:
    #    koi += '.01'
    #band = band.lower()
    #return KOIDATA[koi]['koi_%smag' % band]


def modelpop(koi,model,cc=None,use_cc=True,vcc=None,
             noconstraints=False,tag=None,**kwargs):
    #if type(koi) == type(1.):
    #    name = 'KOI%.2f' % koi
    #else:
    #    name = koi
    
    name = ku.koiname(koi)

    if tag is None:
        folder = '%s/%s' % (FPMODELSDIR,name)
    else:
        folder = '%s/%s_%s' % (FPMODELSDIR,name,tag)
    if re.search('specific',model):
        m = re.search('(\w+)_(specific[0-9])',model)
        basemodel,suffix = (m.group(1),m.group(2))
        files = ('%s/%ss_%s.fits' % (folder,basemodel,suffix),
                 '%s/%ss_%s_params.fits' % (folder,basemodel,suffix))
    else:
        files = ('%s/%ss.fits' % (folder,model),'%s/%ss_params.fits' % (folder,model))

    if noconstraints:
        use_cc = False

    if cc is None and use_cc:
        cc = KOIcc(koi)

    if model=='eb':
        return fpp.EBpopulation(files[0],files[1],vcc=vcc,band='Kepler',
                                noconstraints=noconstraints,**kwargs)
    elif model=='beb':
        return fpp.BGEBpopulation(files[0],files[1],blendmag=KICmag(koi,'kep'),
                                  band='Kepler',cc=cc,noconstraints=noconstraints,**kwargs)
    elif model=='heb':
        return fpp.HEBpopulation(files[0],files[1],Kmag=KICmag(koi,'K'),band='Kepler',
                                 cc=cc,vcc=vcc,noconstraints=noconstraints,**kwargs)
    elif model=='bgpl':
        return fpp.BGTransitpopulation(files[0],files[1],blendmag=KICmag(koi,'kep'),
                                       band='Kepler',cc=cc,noconstraints=noconstraints,**kwargs)
    elif model=='pl':
        return fpp.Transitpopulation(files[0],files[1],noconstraints=noconstraints,**kwargs)

    elif re.search('beb_specific[0-9]',model):
        return fpp.Specific_BGEBpopulation(files[0],files[1],noconstraints=noconstraints,
                                           band='Kepler',cc=cc,blendmag=KICmag(koi,'kep'),
                                           **kwargs)
    elif re.search('heb_specific[0-9]',model):
        return fpp.Specific_HEBpopulation(files[0],files[1],noconstraints=noconstraints,
                                           band='Kepler',cc=cc,**kwargs)

    else:
        raise ValueError('invalid model name: %s' % model)

def koi_rexclusion(koi,default=4,rmin=0.5):
    maxrad = ku.DATA[koi]['koi_dicco_msky_err']*3
    if np.isnan(maxrad):
        maxrad = default
    if maxrad < rmin:
        maxrad = rmin
    return maxrad

class KOIFPP(fpp.FPPCalculation):
    def __init__(self,koi,secthresh=None,Teff=None,logg=None,use_cc=True,vcc=None,
                 maxrad=None,photfile=None,photcols=(0,1),
                 redo_MCMC=False,tag=None,maxq=MAXQ,maxslope=None,P=None,epoch=None,
                 include_specific=True,Tdur=None,ror=None):

        #eventually: move a lot of this into the general FPPCalculation class
        koi = ku.koiname(koi)


        popset = KOIPopulationSet(koi,tag=tag) #PopulationSet of base models
        #trsig = KOItransitsignal(koi)
        if redo_MCMC:
            trsig = KeplerTransitsignal(koi,mcmc=False,maxslope=maxslope,P=P,epoch=epoch,photfile=photfile,
                                        photcols=photcols,Tdur=Tdur,ror=ror)
            trsig.MCMC(refit=True)
        else:
            trsig = KeplerTransitsignal(koi,maxslope=maxslope,P=P,epoch=epoch,photfile=photfile,
                                        photcols=photcols,Tdur=Tdur,ror=ror)

        if maxrad is None:
            maxrad = koi_rexclusion(koi)

        self.imdata = kim.all_imagedata(koi)
        #if there is a source within maxrad, then add "specific" models to popset.  
        #if these models are not created yet, then complain.
        if include_specific:
            comps = self.imdata.within_radius(maxrad)
            for i in range(len(comps)):
                popset.add_population(modelpop(koi,'beb_specific{}'.format(i+1),noconstraints=True,
                                               tag=tag,blendmags=KICmags(koi)))
                popset.add_population(modelpop(koi,'heb_specific{}'.format(i+1),noconstraints=True,
                                               tag=tag))


        if tag is None:
            self.folder = '%s/%s' % (FPMODELSDIR,koi)
        else:
            self.folder = '%s/%s_%s' % (FPMODELSDIR,koi,tag)


        fpp.FPPCalculation.__init__(self,trsig,popset,lhoodcachefile='%s/FPPcache.txt' % self.folder)
        
        if use_cc:
            #cc = KOIcc(koi)
            #if cc is not None:
            #    if type(cc) == type([1,]):
            #        for c in cc:
            #            self.apply_cc(c)
            #    else:
            #        self.apply_cc(cc)
            for cc in self.imdata.ccs:
                self.apply_cc(cc)

        if vcc is not None:
            self.apply_vcc(vcc)

        if secthresh is None:
            noweaksec = False
            try:
                self.weaksec = WEAKSECDATA.ix[koi]
            except KeyError:
                raise NoWeakSecondaryError('{} not in weak secondary table.'.format(koi))

            if self.weaksec['depth'] == 0:
                raise NoWeakSecondaryError('Weak secondary table has depth=0 for {}'.format(koi))

            if self.weaksec['depth'] < 0:
                raise NoWeakSecondaryError('Weak secondary table gives depth<0 for {}'.format(koi))

            secthresh = (self.weaksec['depth'] + 3*self.weaksec['e_depth'])*1e-6

        #if secthresh is None:
        #    secthresh = calc_secthresh(koi,maxq)

        self.apply_secthresh(secthresh) 
        self.set_maxrad(maxrad)

        if Teff is not None:
            self.constrain_property('Teff',measurement=Teff,selectfrac_skip=True)
        if logg is not None:
            self.constrain_property('logg',measurement=logg,selectfrac_skip=True)

        self.name = koi



    def FPPplots(self,tag=None,format='png',folder=None,**kwargs):
        if folder is None:
            folder = self.folder
        fpp.FPPCalculation.lhoodplots(self,folder=folder,tag=tag,figformat=format,**kwargs)
        self.FPPsummary(folder=folder,saveplot=True,tag=tag,figformat=format)
        if tag is None:
            self.FPPsummary(folder=folder,saveplot=True,priorinfo=False,
                            starinfo=False,tag='simple',figformat=format)
        self.plotsignal(folder=folder,saveplot=True,figformat=format)
        self.write_results(folder=folder)

    def write_results(self,folder=None):
        if folder is None:
            folder = self.folder
        fout = open(folder+'/'+'results.txt','w')
        for m in self.popset.shortmodelnames:
            fout.write('%s ' % m)
        fout.write('fpV fp FPP\n')
        Ls = {}
        Ltot = 0
        for model in self.popset.modelnames:
            Ls[model] = self.prior(model)*self.lhood(model)
            Ltot += Ls[model]

        line = ''
        for model in self.popset.modelnames:
            line += '%.2e ' % (Ls[model]/Ltot)
        line += '%.3g %.3f %.2e\n' % (self.fpV(),self.priorfactors['fp_specific'],self.FPP())

        fout.write(line)
        fout.close()

        fout = open(folder+'/'+'keywords.txt','w')
        for k in self.keywords:
            fout.write('%s %s\n' % (k,self.keywords[k]))
        fout.close()

        fout = open(folder+'/'+'priorfactors.txt','w')
        for k in self.priorfactors:
            fout.write('%s %s\n' % (k,self.priorfactors[k]))
        fout.close()

        fout = open(folder+'/'+'constraints.txt','w')
        for k in self['beb'].constraints:
            fout.write('%s\n' % self['beb'].constraints[k])
        fout.close()



class KOIPopulationSet(fpp.PopulationSet):
    def __init__(self,koi,tag=None,**kwargs):
        mags = KICmags(koi)
        plpop = modelpop(koi,'pl',noconstraints=True,tag=tag,**kwargs)
        ebpop = modelpop(koi,'eb',noconstraints=True,tag=tag,**kwargs)
        hebpop = modelpop(koi,'heb',noconstraints=True,tag=tag,**kwargs)
        bebpop = modelpop(koi,'beb',blendmags=mags,noconstraints=True,tag=tag,**kwargs)
        bgplpop = modelpop(koi,'bgpl',blendmags=mags,noconstraints=True,tag=tag,**kwargs)

        fpp.PopulationSet.__init__(self,[plpop,ebpop,hebpop,bebpop,bgplpop])
        
        for kw in kwargs:
            setattr(self,kw,kwargs[kw])


def kepcandidate_info(name,filename='/Users/tdm/Dropbox/FPP/KOI_forFPP.tbl'):
    info = KOIDATA[name]
    

    m = re.search('(\d+)([a-z])',name)
    if m:
        kic = m.group(1)
        ltr = m.group(2)
    found = False
    for line in open(filename):
        if re.search('^[\\\\|]',line):
            continue
        line = line.split()
        if line[0]==kic and line[1]==ltr:
            P,dP,ra,dec,Teff,dTeff,Mstar,dMstar,epoch,foo,depth,dur = array(line[2:]).astype(float)
            found = True
            break
    if not found:
        raise ValueError('%s not in table' % name)
    return {'kic':int(kic),'planet':ltr,'P':P,'dP':dP,'ra':ra,'dec':dec,
            'Teff':Teff,'dTeff':dTeff,'Mstar':Mstar,'dMstar':dMstar,
            'epoch':epoch,'depth':depth,'tdur':dur/24,'Depth':depth,'Dur':dur}


def get_rowefit(koi):
    folder = '%s/koi%i.n' % (ROWEFOLDER,ku.koiname(koi,star=True,koinum=True))    
    num = np.round(ku.koiname(koi,koinum=True) % 1 * 100)    
    rowefitfile = '%s/n%i.dat' % (folder,num)
    try:
        return pd.read_table(rowefitfile,index_col=0,usecols=(0,1,3),
                             names=['par','val','a','err','c'],
                             delimiter='\s+')
    except IOError:
        raise MissingKOIError('{} does not exist.'.format(rowefitfile))

class KeplerTransitsignal(fpp.Transitsignal):
    def __init__(self,koi,mcmc=True,maxslope=None,refit_mcmc=False,
                 photfile=None,photcols=(0,1),Tdur=None,ror=None,P=None,**kwargs):
        self.folder = '%s/koi%i.n' % (ROWEFOLDER,ku.koiname(koi,star=True,koinum=True))
        num = np.round(ku.koiname(koi,koinum=True) % 1 * 100)

        if photfile is None:
            self.lcfile = '%s/tremove.%i.dat' % (self.folder,num)
            if not os.path.exists(self.lcfile):
                raise MissingKOIError('{} does not exist.'.format(self.lcfile))
            logging.debug('Reading photometry from {}'.format(self.lcfile))

            #break if photometry file is empty
            if os.stat(self.lcfile)[6]==0:
                raise EmptyPhotometryError('{} photometry file ({}) is empty'.format(ku.koiname(koi),
                                                                                      self.lcfile))

            lc = pd.read_table(self.lcfile,names=['t','f','df'],
                                                      delimiter='\s+')
            self.ttfile = '%s/koi%07.2f.tt' % (self.folder,ku.koiname(koi,koinum=True))
            self.has_ttvs = os.path.exists(self.ttfile)
            if self.has_ttvs:            
                if os.stat(self.ttfile)[6]==0:
                    self.has_ttvs = False
                    logging.warning('TTV file exists for {}, but is empty.  No TTVs applied.'.format(ku.koiname(koi)))
                else:
                    logging.debug('Reading transit times from {}'.format(self.ttfile))
                    tts = pd.read_table(self.ttfile,names=['tc','foo1','foo2'],delimiter='\s+')

            self.rowefitfile = '%s/n%i.dat' % (self.folder,num)

            self.rowefit = pd.read_table(self.rowefitfile,index_col=0,usecols=(0,1,3),
                                        names=['par','val','a','err','c'],
                                        delimiter='\s+')

            logging.debug('JRowe fitfile: {}'.format(self.rowefitfile))

            P = self.rowefit.ix['PE1','val']
            RR = self.rowefit.ix['RD1','val']
            aR = (self.rowefit.ix['RHO','val']*G*(P*DAY)**2/(3*np.pi))**(1./3)
            cosi = self.rowefit.ix['BB1','val']/aR
            Tdur = P*DAY/np.pi*np.arcsin(1/aR * (((1+RR)**2 - (aR*cosi)**2)/(1 - cosi**2))**(0.5))/DAY

            if 1/aR * (((1+RR)**2 - (aR*cosi)**2)/(1 - cosi**2))**(0.5) > 1:
                logging.warning('arcsin argument in Tdur calculation > 1; setting to 1 for purposes of rough Tdur calculation...')
                Tdur = P*DAY/np.pi*np.arcsin(1)/DAY

            if (1+RR) < (self.rowefit.ix['BB1','val']):
                #Tdur = P*DAY/np.pi*np.arcsin(1/aR * (((1+RR)**2 - (aR*0)**2)/(1 - 0**2))**(0.5))/DAY/2.
                raise BadRoweFitError('best-fit impact parameter ({:.2f}) inconsistent with best-fit radius ratio ({}).'.format(self.rowefit.ix['BB1','val'],RR))

            if RR < 0:
                raise BadRoweFitError('{0} has negative RoR ({1}) from JRowe MCMC fit'.format(ku.koiname(koi),RR))
            if RR > 1:
                raise BadRoweFitError('{0} has RoR > 1 ({1}) from JRowe MCMC fit'.format(ku.koiname(koi),RR))            
            if aR < 1:
                raise BadRoweFitError('{} has a/Rstar < 1 ({}) from JRowe MCMC fit'.format(ku.koiname(koi),aR))


            self.P = P
            self.aR = aR
            self.Tdur = Tdur
            self.epoch = self.rowefit.ix['EP1','val'] + 2504900

            logging.debug('Tdur = {:.2f}'.format(self.Tdur))
            logging.debug('aR={0}, cosi={1}, RR={2}'.format(aR,cosi,RR))
            logging.debug('arcsin arg={}'.format(1/aR * (((1+RR)**2 - (aR*cosi)**2)/(1 - cosi**2))**(0.5)))
            logging.debug('inside sqrt in arcsin arg={}'.format((((1+RR)**2 - (aR*cosi)**2)/(1 - cosi**2))))
            logging.debug('best-fit impact parameter={:.2f}'.format(self.rowefit.ix['BB1','val']))

            lc['t'] += (2450000+0.5)
            lc['f'] += 1 - self.rowefit.ix['ZPT','val']

            if self.has_ttvs:
                tts['tc'] += 2504900

            ts = pd.Series()
            fs = pd.Series()
            dfs = pd.Series()

            if self.has_ttvs:
                for t0 in tts['tc']:
                    t = lc['t'] - t0
                    ok = np.absolute(t) < 2*self.Tdur
                    ts = ts.append(t[ok])
                    fs = fs.append(lc['f'][ok])
                    dfs = dfs.append(lc['df'][ok])
            else:
                center = self.epoch % self.P
                t = np.mod(lc['t'] - center + self.P/2,self.P) - self.P/2
                ok = np.absolute(t) < 2*self.Tdur
                ts = t[ok]
                fs = lc['f'][ok]
                dfs = lc['df'][ok]

            logging.debug('{0}: has_ttvs is {1}'.format(koi,self.has_ttvs))
            logging.debug('{} light curve points used'.format(ok.sum()))


            if maxslope is None:
                #set maxslope using duration
                maxslope = max(Tdur*24/0.5 * 2, 30) #hardcoded in transitFPP as default=30

            p0 = [Tdur,RR**2,3,0]
            self.p0 = p0
            logging.debug('initial trapezoid parameters guess: {}'.format(p0))
            fpp.Transitsignal.__init__(self,np.array(ts),np.array(fs),np.array(dfs),p0=p0,
                                       name=ku.koiname(koi),
                                       P=P,maxslope=maxslope)
        else:
            if P is None:
                P = ku.DATA[koi]['koi_period']
            ts,fs = np.loadtxt(photfile,usecols=photcols,unpack=True)
            if Tdur is not None and ror is not None:
                p0 = [Tdur,ror**2,3,0]
            else:
                p0 = None
            fpp.Transitsignal.__init__(self,ts,fs,name=ku.koiname(koi),
                                       P=P,
                                       maxslope=maxslope,p0=p0)
        
        if mcmc:
            self.MCMC(refit=refit_mcmc)

        if self.hasMCMC and not self.fit_converged:
            logging.warning('Trapezoidal MCMC fit did not converge for {}.'.format(self.name))


    def MCMC(self,**kwargs):
        folder = '%s/%s' % (CHAINSDIR,self.name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        fpp.Transitsignal.MCMC(self,savedir=folder,**kwargs)


class Petigura_Transitsignal(KeplerTransitsignal,fpp.Transitsignal):
    def __init__(self,name,width=3,mcmc=True,maxslope=None,P=None,epoch=None,
                 useascii=False,asciicols=(1,2),correct_ttvs=False):

        name = ku.koiname(name)

        #m = re.search('K((\d\d\d\d\d)\.(\d\d))',name)  #fix some of this naming mess at some point
        #if m:
        #    name = 'KOI%.2f' % float(m.group(1))

        self.name = name

        #m = re.search('KOI(\d+\.\d+)',name)
        #if m:
        #    koinum = float(m.group(1))
        #else:
        #    raise ValueError('Please use KOIn+.xx or K?????.?? format to name KOIs')


        if name in KOICOMMENTS:
            self.comment = KOICOMMENTS[name]
            logging.warning('comment for %s: %s' % (name,self.comment))
        else:
            self.comment = ''


        koi_info = KOIDATA[name]
        if koi_info['koi_disposition']=='FALSE':
            logging.warning('%s is a known false positive.' % name)

        depth = koi_info['koi_depth']
        if depth==0:
            depth = koi_info['koi_ror']**2 * 1e6

        #p = {'epoch':koi_info['epoch'],'P':koi_info['period'],'Dur':koi_info['duration'],
        #     'Depth':koi_info['depth'],'kic':int(koi_info['kepid']),'tdur':koi_info['duration']/24.}
        p = {'epoch':koi_info['koi_time0bk'],'P':koi_info['koi_period'],'Dur':koi_info['koi_duration'],
             'Depth':depth,'kic':int(koi_info['kepid']),'tdur':koi_info['koi_duration']/24.}
        if P is not None:
            p['P'] = P
        if epoch is not None:
            p['epoch'] = epoch

        if maxslope is None:
            #set maxslope given duration information
            maxslope = max(koi_info['koi_duration']/0.5 * 2,20)

        
        filename = os.path.join(ASCIIDATADIR,'%s.ascii' % ku.koiname(name))
        if os.path.exists(filename) and useascii:
            print('using ascii light curve: %s, columns %s' % (filename,asciicols))
            self.filename = filename
            ts,fs = np.loadtxt(filename,usecols=asciicols,unpack=True)
            tc0 = 0
            p0 = np.array([p['Dur']/24,p['Depth']*1e-6,3.,tc0])
            fpp.Transitsignal.__init__(self,ts,fs,p0=p0,name=name,P=p['P'],maxslope=maxslope)
            if mcmc:
                self.MCMC()
            return
            
        filename = os.path.join(KEPDATADIR,'%s.h5' % ku.koiname(name))

        if not os.path.exists(filename) or os.path.getsize(filename)==0:
            #filename = os.path.join(KEPDATADIR,'%s.h5' % koiname(name))        
            #if not os.path.exists(filename) or os.path.getsize(filename)==0:
            raise MissingKOIError('%s photometry data (%s) not available (or zero-sized).' % 
                                  (name,filename))
        try:
            h5 = h5py.File(filename,'r')
        except IOError,e:
            raise BadPhotometryError('h5 file problem with %s. (%s)' % (name,e))
        
        self.filename = filename
        #self.h5 = h5

        try:
            PF = h5['lcPF0'][:]
            ts = PF['tPF']
            fs = PF['f']+1 #default is zero-mean flux; we need 1-mean
            tabs = PF['t']
        except KeyError,e:
            raise BadPhotometryError('Incomplete h5 file for %s. (%s)' % (name,e))

        tbin = h5['blc30PF0'][:]['tb']
        fbin = h5['blc30PF0'][:]['med']
        tc0 = tbin[np.argmin(fbin)]
        #tc0 = 0
        p0 = np.array([p['Dur']/24,p['Depth']*1e-6,3.,tc0])

        #if correct_ttvs:
        #    print('warning: TTV correction does not really work yet!')
        #    koinum = ku.koiname(self.name,koinum=True)
        #    ttvdata = TTVDATA.query('KOI==%.2f' % koinum)
        #    if len(ttvdata) > 0:
        #        w = np.where(PF['qarr'] <= TTVMAXQ)
        #        ts = ts[w]
        #        fs = fs[w]
        #        tabs = tabs[w]
        #        ntr = ((tabs-p['epoch'])/p['P']).round()
        #        for i in np.arange(ntr.max()+1):
        #            w = np.where(ntr==i)
        #            tcen = np.array(ttvdata.query('N==%i' % i)['tn']) #why is this so hard?
        #            #print(tcen)
        #            ts[w] = tabs[w]-tcen
                    

        mask = np.ones(ts.shape).astype(bool)
        m = re.search('^KOI(\d+)\.(\d+)$',name)
        if m:
            koistar = m.group(1)
            koinum = m.group(2)
            for num in ['01','02','03','04','05','06','07','08','09','10']:
                newname = 'KOI%s.%s' % (koistar,num)
                if newname in KOIDATA and num != koinum:
                    per,ep = (KOIDATA[newname]['koi_period'],KOIDATA[newname]['koi_time0bk'])
                    dur = KOIDATA[newname]['koi_duration']/24
                    #this masks a bit more than necessary--if centered correctly
                    notok = np.absolute((tabs - ep + per/2) % per - per/2) < dur
                    if notok.sum() > 0:
                        logging.info('%s: masking %i points (of %i) from %s' %\
                                         (name,notok.sum(),len(tabs),newname))
                    mask &= ~notok

        w = np.where(mask)
        self.epoch = p['epoch']
        fpp.Transitsignal.__init__(self,ts[w],fs[w],p0=p0,name=name,P=p['P'],maxslope=maxslope)        

        if mcmc:
            try:
                self.MCMC()
            except ValueError:
                self.MCMC(p0=[p['Dur']/24,p['Depth']*1e-6,3.,0],refit=True)

        h5.close()


        
class KeplerTransitsignal_old(fpp.Transitsignal):
    def __init__(self,name,width=3):
        self.name = name

        koi_info = KOIDATA[name]
        p = {'epoch':koi_info['koi_time0bk'],'P':koi_info['koi_period'],'Dur':koi_info['koi_duration'],
             'Depth':koi_info['koi_depth'],'kic':int(koi_info['kepid']),'tdur':koi_info['koi_duration']/24.}
        
        #p = kepcandidate_info(name)
        file = os.path.join(KEPDATADIR,'tLC%09d.fits' % p['kic'])
        tLC = atpy.Table(file,verbose=False)

        t = tLC.t #+ 2454833.0

        fm = ma.masked_array(tLC.fdt,mask=tLC.fmask)

        center = p['epoch'] % p['P']

        resL = tval.LDTwrap(t,fm,p)
        res = np.hstack(resL)
        ts = np.mod(res['tdt']-center+p['P']/2,p['P']) - p['P']/2
        #ts = mod(res['tdt'],p['P']) - p['P']/2
        #ts = mod(res['tdt']-center,p['P'])
        fs = res['fdt']

        w = np.where(np.absolute(ts) < width*p['Dur']/24)

        p0 = np.array([p['Dur']/24,p['Depth']*1e-6,3.,0])
        fpp.Transitsignal.__init__(self,ts[w],fs[w]+1,p0=p0,name=name,P=p['P'])        

        try:
            self.MCMC()
        except:
            logging.error("error with MCMC: %s" % self.name)
            
    def MCMC(self,**kwargs):
        folder = '%s/%s' % (CHAINSDIR,self.name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        fpp.Transitsignal.MCMC(self,savedir=folder,**kwargs)
                            

###############Exceptions################

class BadPhotometryError(Exception):
    pass

class MissingKOIError(Exception):
    pass

class BadRoweFitError(Exception):
    pass

class EmptyPhotometryError(Exception):
    pass

class NoWeakSecondaryError(Exception):
    pass
