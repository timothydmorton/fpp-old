import os,sys,re,os.path,time
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    print 'pylab not imported'

import matplotlib.ticker
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA


from consts import *
from scipy.interpolate import UnivariateSpline as interpolate
from scipy.stats import gaussian_kde,poisson
from scipy.integrate import quad,trapz
import numpy.random as rand

import parsetable as pt
try:
    import keplerfpp as kfpp
except:
    print 'keplerfpp not imported...'
from progressbar import Percentage,Bar,RotatingMarker,ETA,ProgressBar
import logging
import plotutils as plu

import distributions as dists

import transit_basic as tr

MAFN = tr.MAInterpolationFunction(nzs=200,nps=200,pmin=0.003,pmax=0.3)

import atpy

from astropy.table import Table,Column

import pandas as pd

#KICPROPS = pt.parsetxt('%s/targetkic.txt' % os.environ['KEPLERDIR'],delimiter='|')
KICPROPS = {}
try:
    DRESSINGPROPS = pt.parsetxt('%s/dressing_starprops.txt' % os.environ['FPPDIR'])
    DRESSINGPROPS[8561063] = dict(FeH=-0.48,Mstar=0.13,Rstar=0.17,Teff=3068)  # KOI-961
except IOError:
    logging.warning('Dressing props not loaded.')
    DRESSINGPROPS = {}

def write_starobs_props(filename,kics=DRESSINGPROPS.keys(),maxq=6):
    """Rstar, CDPP, Tobs
    """
    fout = open(filename,'w')
    fout.write('kic Rstar CDPP Tobs\n')
    for kic in kics:
        fout.write('%i %.2f %.1f %i\n' % (kic,DRESSINGPROPS[kic]['Rstar'],
                                          kfpp.median_CDPP(kic,maxq=maxq),
                                          kfpp.days_observed(kic,maxq=maxq)))
                                          

def semimajor(P,mstar=1):
    """P in days, mstar in Solar masses, returns a in AU
    """
    return ((P*DAY/2/np.pi)**2*G*mstar*MSUN)**(1./3)/AU

def transit_T14(P,Rp,Rs=1,b=0,Ms=1,ecc=0,w=0):
    """P in days, Rp in Earth radii, Rs in Solar radii, b=impact parameter, Ms Solar masses. Returns T14 in hours. w in deg.
    """
    a = semimajor(P,Ms)*AU
    k = Rp*REARTH/(Rs*RSUN)
    inc = np.pi/2 - b*RSUN/a
    return  P*DAY/np.pi*np.arcsin(Rs*RSUN/a * np.sqrt((1+k)**2 - b**2)/np.sin(inc)) *\
        np.sqrt(1-ecc**2)/(1+ecc*np.sin(w*np.pi/180)) / 3600.

def transit_T23(P,Rp,Rs=1,b=0,Ms=1,ecc=0,w=0):
    a = semimajor(P,Ms)*AU
    k = Rp*REARTH/(Rs*RSUN)

    return P*DAY/np.pi*np.arcsin(Rs*RSUN/a * np.sqrt((1-k)**2 - b**2)/np.sin(inc*pi/180)) *\
        np.sqrt(1-ecc**2)/(1+ecc*np.sin(w*pi/180)) #*24*60    

def eta_snrthresh(snr,thresh=7.1):
    snr = np.atleast_1d(snr)
    effs = snr*0
    effs[np.where(snr >= thresh)] = 1.
    if np.size(effs)==1:
        return effs[0]
    else:
        return effs

#def transit_T(*args,**kwargs):
#    return (transit_T14(*args,**kwargs) + transit_T23(*args,**kwargs))/2

#make SNRfn a class
def modify_SNRfn(snrfn,new,SNRmax=50):
    """Makes a new SNRfn object by transforming old one

    new must be dictionary with the following properties:
     -SNR
     -period
     -Rp
     -Rstar
     -duration
     -CDPP
     -Tobs
   
    """
    snrs = np.arange(0,SNRmax+.1,0.5)

    orig = snrfn.orig
    factor = 1
    #factor *= (float(new['SNR'])/snrfn.orig['SNR'])
    #factor *= (float(new['period'])/orig['period'])**(-1./2)
    factor *= (float(new['Rp'])/orig['Rp'])**(2)
    #factor *= (float(new['Rstar'])/orig['Rstar'])**(-2)
    factor *= (float(new['duration'])/orig['duration'])**(1./2)
    factor *= (float(new['CDPP'])/orig['CDPP'])**(-1)
    factor *= (float(new['Tobs'])/orig['Tobs'])**(1./2)

    print factor
    yvals = snrfn(snrs/factor)/factor
    fn = interpolate(snrs,yvals,s=0)
    return SNRfn(fn,new,snrfn.SNRmax,snrfn.kics,snrfn.logPdist,snrfn.maxq)



class SNRfn(object):
    def __init__(self,fn,orig,SNRmax,kics,logPdist,maxq):
        if type(fn) == type(''):
            snrs,vals = np.loadtxt(fn,unpack=True)
            fn = interpolate(snrs,vals,s=0)
        self.fn = fn
        self.orig = orig
        self.SNRmax = SNRmax
        self.kics = kics
        self.logPdist = logPdist
        self.maxq = maxq
        self.area = quad(fn,0,SNRmax,full_output=1)[0]
        #self.norm = quad(fn,0,SNRmax)[0]

    def savefits(self,filename,dsnr=0.2):
        """Save to fits file...?
        """
        snrs = np.arange(0,self.SNRmax+dsnr,dsnr)
        vals = self(snrs)
        t = atpy.Table()
        t.add_column('snr',snrs)
        t.add_column('pdf',vals)
        for key in self.orig:
            t.add_keyword(key,self.orig[key])

    def __call__(self,s):
        s = np.atleast_1d(s)
        vals = self.fn(s)#/self.norm
        vals = np.atleast_1d(vals)
        vals[np.where((s > self.SNRmax) | (s < 0))] = 0
        if np.size(vals)==0:
            return vals[0]
        else:
            return vals

    def plot(self,fig=None,SNRmax=None,dSNR=0.2,**kwargs):
        plu.setfig(fig)
        if SNRmax is None:
            SNRmax = self.SNRmax
        snrs = np.arange(0,SNRmax+dSNR,dSNR)
        plt.plot(snrs,self(snrs),**kwargs)
        plt.ylim(ymin=0)
        plt.xlabel('SNR')
        plt.ylabel('Probability Density')
        plt.yticks([])

    def integrate_efficiency(self,etafn=None):
        if etafn is None:
            etafn = kfpp.SNRramp
        foo = lambda x: etafn(x)*self(x)
        integral = quad(foo,0,self.SNRmax,full_output=1)[0]
        leftover = 1 - self.area
        return integral + leftover

    def save_fn(self,filename,dsnr=0.2):
        snrs = np.arange(0,self.SNRmax+dsnr,dsnr)
        vals = self(snrs)
        pts = np.array([snrs,vals]).T
        np.savetxt(filename,pts)

def make_SNRfn(orig,logPdist,kics=None,SNRmax=50,N=1e4,
               use_pbar=True,return_fn=True,maxq=6,Pfixed=False,
               simple=False):
    """
    
    original props:
     -SNR
     -period
     -Rp
     -Rstar
     -duration (T14)
     -CDPP (appropriately weighted--or just median)
     -Tobs (days observed)

    """
    if kics is None:
        kics = DRESSINGPROPS.keys()

    if not simple:
        SNRs = np.zeros((len(kics),N)) # np.linspace(0,SNRmax,Npts)
    else:
        N = 0
    if use_pbar and not simple:
        widgets = ['calculating SNR distribution: ',Percentage(),' ',Bar(marker=RotatingMarker()),' ',ETA()]
        pbar = ProgressBar(widgets=widgets,maxval=len(kics))
        pbar.start()
    for i,kic in enumerate(kics):
        if kic in DRESSINGPROPS:
            Rs = DRESSINGPROPS[kic]['Rstar']
            Ms = DRESSINGPROPS[kic]['Mstar']
        else:
            Rs = KICPROPS[kic]['kic_radius']
            logg = KICPROPS[kic]['kic_logg']
            Ms = (Rs*RSUN)**2*10**logg/G/MSUN
        newprops = dict(Rstar=Rs,Mstar=Ms,CDPP=kfpp.median_CDPP(kic,maxq=maxq),Tobs=kfpp.days_observed(kic,maxq=maxq))
        if simple:
            newsnr = orig['SNR']*\
                (newprops['Rstar']/orig['Rstar'])**(-2)*\
                (newprops['CDPP']/orig['CDPP'])**(-1)*\
                (newprops['Tobs']/orig['Tobs'])**(1./2)
            if newsnr > 7.1:
                N+=1
            continue
        SNRs[i,:] = SNR_distribution(orig,newprops,logPdist,N,return_kde=False,Pfixed=Pfixed)
        if use_pbar and i % 10==0:
            pbar.update(i)

    if simple:
        return N
    
    if use_pbar:
        pbar.finish()
    if return_fn:
        w = np.where(~np.isnan(SNRs.ravel()))
        #SNRmax = SNRs.ravel()[w].max()
        bins = np.arange(0,SNRmax+1,1)
        h,bins = np.histogram(SNRs.ravel()[w],bins=bins,normed=True)
        bins = (bins[1:] + bins[:-1])/2.
        h = np.concatenate((np.array([0]),h))
        bins = np.concatenate((np.array([0]),bins))
        #print bins,h
        fn = interpolate(bins,h,s=0)
        return SNRfn(fn,orig,SNRmax,kics,logPdist,maxq)
    else:
        return SNRs
    #return interpolate(SNRs,tot/len(kics),s=0)

def SNR_of_P(orig,Ps):
    return orig['SNR']*(Ps/orig['period'])**(-1./2)

def eta_snrthresh(snr,thresh=7.1):
    snr = np.atleast_1d(snr)
    effs = snr*0
    effs[np.where(snr >= thresh)] = 1.
    if np.size(effs)==1:
        return effs[0]
    else:
        return effs


def SNR_distribution(orig,new,logPdist,N=1e5,return_kde=False,Pfixed=False,return_factordict=False):
    """orig, new are dictionaries with relevant parameters for the original and new systems

    original props:
     -SNR
     -period
     -Rp
     -Rstar
     -duration (T14)
     -CDPP (appropriately weighted--or just median)
     -Tobs (days observed)

    new props:
     -Rstar
     -Mstar
     -CDPP (appropriately weighted--or just median)
     -Tobs (days observed)

    """

    if not Pfixed:
        newPs = np.squeeze(10**(logPdist.resample(int(N))))
    else:
        newPs = orig['period']
    newbs = rand.random(size=N)
    
    newTs = tr.transit_T14(newPs,orig['Rp'],new['Rstar'],newbs,new['Mstar'])
    factordict = {'Rstar':(new['Rstar']/orig['Rstar'])**(-2),
                  'period':(newPs/orig['period'])**(-1./2),
                  'duration':(newTs/orig['duration'])**(1./2),
                  'CDPP':(new['CDPP']/orig['CDPP'])**(-1),
                  'Tobs':(new['Tobs']/orig['Tobs'])**(1./2)}
    factors = (new['Rstar']/orig['Rstar'])**(-2)*\
        (newPs/orig['period'])**(-1./2)*\
        (newTs/orig['duration'])**(1./2)*\
        (new['CDPP']/orig['CDPP'])**(-1)*\
        (new['Tobs']/orig['Tobs'])**(1./2)    
    SNRs = orig['SNR']*factors
    if return_factordict:
        return factordict
    if return_kde:
        kde = gaussian_kde(SNRs)
        return kde
    else:
        return SNRs


class PlanetDistribution(object):
    def __init__(self):
        pass

#used for MS2014: p0=10,a=0.5,minp=0.5,max=150
class ToyLogPdist(object):
    def __init__(self,p0=10,a=1.,minp=0.5,maxp=90):
        self.p0 = p0
        self.minp = minp
        self.maxp = maxp

        self.logp0 = np.log10(p0)
        self.a = float(a)
        self.minlogp = np.log10(minp)
        self.maxlogp = np.log10(maxp)
        self.norm = self.a*(np.exp(self.logp0/self.a) - np.exp(self.minlogp/self.a)) +\
                            np.exp(self.logp0/self.a)*(self.maxlogp - self.logp0)
        
    def __call__(self,ps):
        return self.pdf(ps)

    def pdf(self,ps):
        ps = np.atleast_1d(ps)
        vals = ps*0
        vals[np.where(ps < self.logp0)] = np.exp(ps/self.a)
        vals[np.where(ps >= self.logp0)] = np.exp(self.logp0/self.a)
        vals[np.where((ps < self.minlogp) | (ps > self.maxlogp))] = 0
        if np.size(vals)==1:
            return vals[0]/self.norm
        else:
            return vals/self.norm

    def cdf(self,ps):
        ps = np.atleast_1d(ps)
        vals = ps*0
        w = np.where(ps < self.p0)
        vals[w] = self.a*(np.exp(ps[w]/self.a) - np.exp(self.minlogp/self.a))
        w = np.where(ps >= self.logp0)
        vals[w] = self.a*(np.exp(self.logp0/self.a) - np.exp(self.minlogp/self.a)) + np.exp(self.logp0/self.a)*(ps[w]-self.logp0)
        w = np.where(ps < self.minlogp)
        vals[w] = 0
        w = np.where(ps > self.maxlogp)
        vals[w] = 1
        if np.size(vals)==1:
            return vals[0]/self.norm
        else:
            return vals/self.norm
        
    def resample(self,N):
        u = rand.random(size=N)
        ps = u*0
        u0 = self.cdf(self.logp0)
        w1 = np.where(u < u0)
        ps[w1] = self.a*np.log(self.norm*u[w1]/self.a + np.exp(self.minlogp/self.a))
        w2 = np.where(u >= u0)
        ps[w2] = self.logp0 + (u[w2] - u0)*((self.maxlogp - self.logp0)/(1-u0))
        return ps

def transitSNR(P,Rp,Rs,Ms,b,Tobs,noise,u1=0.394,u2=0.261,ecc=0,w=0,
               npts=100,max_only=False,simple=False,noise_timescale=3,
               force_1d=False):
    """noise in ppm, noise_timescale in hours; P,Tobs in Days
    """
    Tdur = tr.transit_T14(P,Rp,Rs,b,Ms,ecc,w) #in hours; will slightly overestimate duration
    Npts = Tdur/noise_timescale
    Nobs = Tobs/P
    if simple:
        depth = ((Rp*REARTH)/(Rs*RSUN))**2
    else:
        depth = tr.eclipse_depth(MAFN,Rp,Rs,b,u1,u2,npts=npts,max_only=max_only,
                                 force_1d=force_1d)
    #print Tdur,Nobs,Npts,depth
    return depth/(noise*1e-6)*np.sqrt(Npts * Nobs)

PTsnrs = [6,7,8,9,10,11,12,13,14,15,16,17]
PTetas = [0.03,0.1,0.22,0.41,0.62,0.77,0.85,0.9,0.95,0.97,0.98,0.99]
PTfn = interpolate(PTsnrs,PTetas,s=0)

def etaPT(s):
    """ Kepler pipeline detection efficiency as function of SNR, from Peter Tenenbaum
    """
    s = np.atleast_1d(s)
    e = PTfn(s)
    e[np.where(s<5.5)] = 0
    e[np.where(s>17.5)] = 1
    return e

class TransitSurvey(object):
    def __init__(self,targets,detections,noise_timescale=3.,logper_kdewidth=0.15,
                 logperdist=None,
                 snrdist_filename='snrdist.fits',snrdist=None,etafn=etaPT,
                 etadisc_filename=None,
                 recalc=False,recalc_etadisc=False,maxp=None,
                 survey_transitprob=None,etafn_es=None,etafn_rs=None):
        """ A general-purpose class to analyze a transit survey

        targets and detections are both astropy Tables

        targets must contain the following columns:
          ID,R,M,Teff,noise,Tobs
             logg and feh are optional.  if not provided, logg is calculated from R and M, and feh=0
             u1, and u2 are also optional.  if not provided, they are calculated from Teff, logg, feh

        detections must contain the following:
           name, ID, Rp, dRp, P 
             transprob optional


        noise_timescale in hours
        """
        self.targets = targets
        if 'logg' not in targets.colnames:
            logg = np.log10(G*targets['M']*MSUN/(targets['R']*RSUN)**2)
            self.targets.add_column(Column(data=logg,name='logg'))
        if 'feh' not in targets.colnames:
            feh = np.zeros(len(self.targets))
            self.targets.add_column(Column(data=feh,name='feh'))
        if 'u1' and 'u2' not in targets.colnames:
            u1,u2 = tr.ldcoeffs(self.targets['Teff'],self.targets['logg'],self.targets['feh'])
            self.targets.add_columns([Column(data=u1,name='u1'),Column(data=u2,name='u2')])

        if 'transprob' not in detections.colnames:
            transprobs = self.mean_transitprob(detections['P'])
            detections.add_column(Column(data=transprobs,name='transprob'))

        self.detections = detections


        logpers = np.log10(self.detections['P'])
        if maxp is None:
            maxp = self.detections['P'].max()
        self.maxp = maxp
        self.logperdist_empirical = WeightedKDE(logpers,weights=1./self.detections['transprob'],
                                                widths=logper_kdewidth,norm=len(self.targets),
                                                minval=np.log10(self.detections['P'].min()),
                                                maxval=np.log10(maxp),normed=True)
        
        self.noise_timescale = noise_timescale
        self.etafn = etafn

        if os.path.exists(snrdist_filename):
                self.snrdist_filename = snrdist_filename
                if snrdist is None:
                    t = Table.read(snrdist_filename)
                else:
                    t = snrdist
                self.SNRdist_samples = t['snr']
                self.simtable = t
                survey_transitprob = t.meta['transitfraction']
                if re.search('ToyLogPdist',t.meta['logperdist']):
                    self.logperdist = ToyLogPdist(t.meta['P0'],
                                                  t.meta['A'],
                                                  t.meta['MINP'],
                                                  t.meta['MAXP'])  
                else:
                    if logperdist is None:
                        self.logperdist = self.logperdist_empirical
                    else:
                        self.logperdist = logperdist

                if 'MAXP' not in self.simtable.meta:
                    recalc = True
                elif self.maxp != self.simtable.meta['MAXP']:
                    recalc = True

        if recalc:
            if logperdist is None:
                self.logperdist = self.logperdist_empirical
            else:
                self.logperdist = logperdist
            
            self.generate_SNRdist(filename=snrdist_filename,recalc=True)

            

        if survey_transitprob is None:
            self.survey_transitprob = self.calc_survey_transitprob()
        else:
            self.survey_transitprob = survey_transitprob


        if not hasattr(self,'SNRdist_samples'):
            self.generate_SNRdist(filename=snrdist_filename,recalc=True)

        #print 'calculating eta_disc as function of r'
        self.calc_etadisc_of_r(filename=etadisc_filename,recalc=recalc_etadisc,
                               rs=etafn_rs,es=etafn_es)
        if not hasattr(self,'survey_transitprob'):
            self.calc_survey_transitprob()
        
        self.etadisc_rough = self.etadisc_of_r(self.detections['Rp'])
        self.etadisc_simple,self.snrs_simple = self.calc_etadisc_simple()
        self.etadisc_simple_thresh,self.snrs_simple_thresh = self.calc_etadisc_simple(eta_snrthresh)

    def write_etadisc(self,filename):
        rs = np.arange(0,3,0.01)
        etas = self.etadisc_of_r(rs)
        np.savetxt(filename,np.array([rs,etas]).T)


    def writetargets_ascii(self,filename):
        self.targets.write(filename,format='ascii')

    def writedetections_ascii(self,filename):
        self.detections.write(filename,format='ascii')

    def calc_survey_transitprob(self,N=1e4,logperdist=None):
        if logperdist is None:
            logperdist = self.logperdist
        pers = 10**logperdist.resample(N)
        self.survey_transitprob = self.mean_transitprob(pers).mean()
        #return self.mean_transitprob(pers).mean()

    def completeness_inbin(self,pbin=(-np.inf,np.inf),rp=1.):
        t = self.simtable
        w = np.where((t['P'] > pbin[0]) & (t['P'] < pbin[1]))
        snrs = t['snr'][w] * rp**2
        return (self.etafn(snrs)).sum()/float(len(snrs))
        
    def calc_etadisc_simple(self,etafn=None,rps=None,Ps=None):
        if rps is None:
            rps = self.detections['Rp']
        if Ps is None:
            Ps = self.detections['P']
        if etafn is None:
            etafn = self.etafn

        etadisc_simple = np.zeros(len(rps))
        i=0
        snrs_simple = np.zeros((len(rps),len(self.targets)))
        for rp,P in zip(rps,Ps):
            Tdurs = transit_T14(P,0,self.targets['R'],self.targets['M'])
            snrs = transitSNR(P,rp,self.targets['R'],self.targets['M'],0,
                              self.targets['Tobs'],self.targets['noise'],
                              noise_timescale=self.noise_timescale,simple=True)
            #snrs = (rp*REARTH / (self.targets['R']*RSUN))**2 * \
            #       np.sqrt(self.targets['Tobs']/P * Tdurs/self.noise_timescale) /\
            #       (self.targets['noise']*1e-6)
            etadisc_simple[i] = (etafn(snrs)).sum()/float(len(snrs))
            snrs_simple[i,:] = snrs
            i += 1

        return etadisc_simple,snrs_simple

    def calc_etadisc_of_r(self,res=0.1,etafn=None,rmax=10,filename=None,recalc=False,
                          rs=None,es=None):
        if rs is not None and es is not None:
            pass
        elif filename is not None and not recalc and os.path.exists(filename):
            rs,es = np.loadtxt(filename,unpack=True)
        else:
            print 'calculating etadisc of r'
            if etafn is not None:
                self.etafn = etafn
                print 'etafn is changed.'
            else:
                etafn = self.etafn

            rs = np.arange(0,rmax+res,res)
            es = rs*0

            for i,r in enumerate(rs):
                es[i] = self.eta_disc(r,etafn=etafn)

        etar_fn = interpolate(rs,es,s=0)
            
        def fn(r):
            r = np.atleast_1d(r)
            e = etar_fn(r)
            e[np.where(r > rmax)] = 1
            return e
        
        if filename is not None:
            np.savetxt(filename,np.array([rs,es]).T)
        self.etadisc_of_r = fn

    def mean_transitprob(self,P):
        P = np.atleast_1d(P)
        Ps = P[:,np.newaxis]
        Ms = self.targets['M'][np.newaxis,:]

        a = semimajor(Ps,Ms)

        return (RSUN*self.targets['R']/(a*AU)).mean(axis=1)

    def eta_disc(self,rp,etafn=None):
        snrs = self.SNRdist_samples * (rp**2)
        if etafn is None:
            etafn = self.etafn

        return (etafn(snrs)).sum()/float(len(snrs))


    def generate_SNRdist(self,N=5e4,npts=50,debug=False,filename='snrdist.fits',
                         save=True,recalc=False,savefilename=None,simple=False,
                         max_only=False,circular=True):
        """N is approximate number of simulations per target star, normalized to mean Rstar.  

        1 earth radius is default for planet
        """
        if filename is not None and not recalc and os.path.exists(filename):
            self.snrdist_filename = filename
            t = Table.read(filename)
            self.SNRdist_samples = t['snr']
            self.simtable = t
            #self.SNRdist_samples = np.load(filename)
            #self.SNRdist = dists.Hist_Distribution(self.SNRdist_samples,bins=np.arange(100),minval=0)
            return

        Rp = 1

        snrlist = []
        perlist = []
        blist = []
        Rlist = []
        Mlist = []
        Tobslist = []
        noiselist = []
        u1list = []
        u2list = []

        transprobs = []

        mean_Rs = self.targets['R'].mean()

        widgets = ['calculating SNR distribution: ',Percentage(),' ',
                   Bar(marker=RotatingMarker()),' ',ETA()]
        pbar = ProgressBar(widgets=widgets,maxval=len(self.targets))
        pbar.start()
        i=0
        transiting = 0
        lots_of_pers = 10**self.logperdist.resample(N*10)
        for Rs,Ms,noise,Tobs,u1,u2 in zip(self.targets['R'],
                                          self.targets['M'],
                                          self.targets['noise'],
                                          self.targets['Tobs'],
                                          self.targets['u1'],
                                          self.targets['u2']):
            inds = rand.randint(0,N*10,size=N)
            pers = lots_of_pers[inds]
            semimajors = semimajor(pers,Ms)
            incs = np.arccos(rand.random(N))
            ws = rand.random(N)*2*np.pi
            if circular:
                eccs = 0
            else:
                raise ValueError('have not implemented eccentricity here...')
            bs = semimajors*AU*np.cos(incs) / (Rs*RSUN) * (1-eccs**2)/(1 + eccs*np.sin(ws))
            #print bs[:100]

            #Rtots = (Rs*RSUN + REARTH)/(Rs*RSUN)
            tra = (bs < 1)
            wtra = np.where(tra)
            Ntra = tra.sum()
            #print Ntra,'transiting...(Rs = %.2f)' % Rs
            #print bs[wtra]
            #start = time.time()
            SNRs = transitSNR(pers[wtra],1,Rs,Ms,bs[wtra],Tobs,noise,u1,u2,npts=npts,
                              max_only=max_only,simple=simple,noise_timescale=self.noise_timescale,
                              )
            #stop = time.time()
            #print '(%.2f s)' % (stop-start)
            snrlist.append(SNRs)


            perlist.append(pers[wtra])
            transprobs.append(Rs*RSUN/(semimajors[wtra]*AU))
            if debug:
                blist.append(bs[wtra])
                Rlist.append(np.ones(Ntra)*Rs)
                Mlist.append(np.ones(Ntra)*Ms)
                noiselist.append(np.ones(Ntra)*noise)
                Tobslist.append(np.ones(Ntra)*Tobs)
                u1list.append(np.ones(Ntra)*u1)
                u2list.append(np.ones(Ntra)*u2)

            transiting += tra.sum()
            i+=1
            pbar.update(i)
        
        self.transitingfraction = float(transiting)/(N*len(self.targets))

        pbar.finish()

        self.SNRdist_samples = np.concatenate(snrlist)
        #self.SRNdist = dists.Hist_Distribution(self.SNRdist_samples,bins=np.arange(100),minval=0)
        perlist = np.concatenate(perlist)
        transprobs = np.concatenate(transprobs)
        if debug:
            self.blist = np.concatenate(blist)
            self.Rlist = np.concatenate(Rlist)
            self.Mlist = np.concatenate(Mlist)
            self.noiselist = np.concatenate(noiselist)
            self.Tobslist = np.concatenate(Tobslist)
            self.u1list = np.concatenate(u1list)
            self.u2list = np.concatenate(u2list)

        keywords = {'logperdist':str(type(self.logperdist)),
                    'transitfraction':self.transitingfraction}

        if self.logperdist != self.logperdist_empirical:
            for k,v in self.logperdist.__dict__.iteritems():
                keywords[k] = v                

        if 'maxp' not in keywords and 'MAXP' not in keywords:
            keywords['maxp'] = self.maxp


        t = Table(data=[self.SNRdist_samples,perlist,transprobs],
                  names=['snr','P','transprob'],
                  meta=keywords)

        self.simtable = t

        if save:
            if savefilename is None:
                savefilename = filename
            
            t.write(savefilename,overwrite=True)
            #np.save(savefilename,self.SNRdist_samples)
            


class TransitSurveyFromASCII(TransitSurvey):
    def __init__(self,target_filename,detection_filename,noise_timescale=3.,**kwargs):
        targets = Table.read(target_filename,format='ascii')
        detections = Table.read(detection_filename,format='ascii')

        TransitSurvey.__init__(self,targets,detections,noise_timescale=noise_timescale,**kwargs)
        

class WeightedKDE(object):
    def __init__(self,data,weights=None,widths=None,norm=None,minval=None,maxval=None,
                 custom_kernels=None,posteriors=None,
                 normed=True,widthfactor=1.,force_symmetric=False,
                 truedist=None):
        """A modified 1d kernel density estimator, allowing for weights for each data point

        custom_kernels can be a list of kernel functions, one for each data point.

        if used in the context of a survey, norm should be the number of total stars surveyed
        """
        self.truedist = truedist
        self.data = data
        self.N = len(data)
        if norm is None:
            norm = 1
        self.norm = norm
        self.normed = normed
        
        if weights is None:
            weights = np.ones(data.shape)
        self.weights = weights

        if widths is None:
            widths = self.N**(-1./5)*np.ones(data.shape)
        elif type(widths) in [type(1),type(0.1)]:
            widths = widths*np.ones(data.shape)
        self.widths = widths

        self.widthfactor = widthfactor

        if minval is None:
            minval = data.min() - 3*widths[np.argmin(data)]
        if maxval is None:
            maxval = data.max() + 3*widths[np.argmax(data)]
        self.minval = minval
        self.maxval = maxval

        self.custom_kernels = custom_kernels
        self.force_symmetric = force_symmetric
        self.posteriors = posteriors

        self.uncfn_p = None
        self.uncfn_m = None
        self.bootstrap_med = None
        self.bias = None

        self._setfns()

        #self.bootstrap()
        

    def adjust_width(self,factor):
        self.set_width(self,widths*factor)

    def change_widthfactor(self,widthfactor):
        self.widthfactor = widthfactor
        self._setfns()

    def set_width(self,widths):
        if type(widths) in [type(1),type(.1)]:
            self.widths = np.ones(self.data.shape)*widths
        else:
            self.widths = widths
        self._setfns()

    def _setfns(self,reset_uncfns=False,remove_bias_correction=False,npts=1000):
        vals = np.linspace(self.minval,self.maxval,npts)
        tot = self.evaluate(vals)
        if self.bias is not None and not remove_bias_correction:
            tot -= self.bias(vals)
        tot[np.where(tot < 0)] = 0

        #print vals,tot
        #print 'setting functions...'
        if self.normed:
            #self.norm *= trapz(tot,vals) #might not be right
            self.norm = quad(self.evaluate,self.minval,self.maxval)[0]
            tot /= self.norm

        self.pdf = interpolate(vals,tot,s=0)
        self.cdf = interpolate(vals,tot.cumsum()/tot.cumsum().max(),s=0)
        if reset_uncfns:
            self.uncfn_p = None
            self.uncfn_m = None
            self.bootstrap_med = None
            self.bias = None

        self.integral = quad(self,self.minval,self.maxval,full_output=1)[0]

    def save(self,filename='wkde.h5',npts=1000):
        """Saves crucial skeleton info to file

        sampled pdf, bias function, uncertainty functions
        data points
        weights
        """
        vals = np.linspace(self.minval,self.maxval,npts)
        pdf = self(vals)
        fn_df = pd.DataFrame({'vals':vals,'pdf':pdf})
        if self.bias is not None:
            fn_df['bias'] = self.bias(vals)
        if self.uncfn_p is not None:
            fn_df['uncfn_p'] = self.uncfn_p(vals)
            fn_df['uncfn_m'] = self.uncfn_m(vals)
        if self.truedist is not None:
            fn_df['truedist'] = self.truedist(vals)

        fn_df.to_hdf(filename,'functions')
        
        data_df = pd.DataFrame({'data':self.data,
                                'weights':self.weights})
        data_df.to_hdf(filename,'data',append=True)




        


    def evaluate(self,x):
        x = np.atleast_1d(x)
        tot = x*0

        if self.posteriors is not None:
            for w,p in zip(self.weights,self.posteriors):
                tot += w * p(x)

        elif self.custom_kernels is not None and not self.force_symmetric:  # kernels must be "zeroed"
            for d,w,k in zip(self.data,self.weights,self.custom_kernels):
                tot += w * k((x-d)/self.widthfactor) / self.widthfactor

        else:
            for d,sig,w in zip(self.data,self.widths,self.weights):
                tot += 1./np.sqrt(2*np.pi*sig**2)*np.exp(-((x-d)/self.widthfactor)**2/(2*sig**2))*w / self.widthfactor

        tot /= self.norm
        if np.size(tot)==1:
            return tot[0]
        else:
            return tot

    def bootstrap_bias(self,N=1000,npts=500):
        xs = np.linspace(self.minval,self.maxval,npts)
        norm = quad(self,self.minval,self.maxval)
        tots = np.zeros(npts)
        for i in np.arange(N):
            pass
            

    def bootstrap(self,N=1000,npts=500,use_pbar=True,return_vals=False):
        xs = np.linspace(self.minval,self.maxval,npts)
        tots = np.zeros((N,npts))
        if use_pbar:
            widgets = ['calculating bootstrap variance: ',Percentage(),' ',
                       Bar(marker=RotatingMarker()),' ',ETA()]
            pbar = ProgressBar(widgets=widgets,maxval=N)
            pbar.start()
        for i in np.arange(N):
            Nthis = poisson.rvs(self.N)
            inds = rand.randint(self.N,size=Nthis)
            if self.posteriors is not None:
                new = WeightedKDE(self.data[inds],self.weights[inds],
                                  self.widths[inds],norm=self.norm,posteriors=self.posteriors[inds],
                                  minval=self.minval,maxval=self.maxval,normed=self.normed)
            elif self.custom_kernels is not None:
                new = WeightedKDE(self.data[inds],self.weights[inds],
                                  self.widths[inds],norm=self.norm,custom_kernels=self.custom_kernels[inds],
                                  minval=self.minval,maxval=self.maxval,normed=self.normed)
            else:
                new = WeightedKDE(self.data[inds],self.weights[inds],
                                  self.widths[inds],norm=self.norm,
                                  minval=self.minval,maxval=self.maxval,normed=self.normed)
            tots[i,:] = new.evaluate(xs)
                
            if use_pbar and i % 10==0:
                pbar.update(i)
        if use_pbar:
            pbar.finish()
        sorted = np.sort(tots,axis=0)
        pvals = sorted[-N*16/100,:] #84th pctile pts
        medvals = sorted[N/2,:]  #50th pctile pts
        mvals = sorted[N*16/100,:] #16th pctile pts

        self.bootstrap_med = interpolate(xs,medvals,s=0)
        self.bias = lambda x: self.bootstrap_med(x) - self.evaluate(x)

        self.uncfn_p = interpolate(xs,pvals - self.bias(xs),s=0) 
        self.uncfn_m = interpolate(xs,mvals - self.bias(xs),s=0)

        self._setfns()

        if return_vals:
            return xs,tots

    def __call__(self,x):
        x = np.atleast_1d(x)
        vals = self.pdf(x)
        vals = np.atleast_1d(vals)
        vals[np.where((x < self.minval) | (x > self.maxval))] = 0
        if np.size(vals)==1:
            return vals[0]
        else:
            return vals

    def plot(self,fig=None,log=False,scale=1.,label=None,uncs=False,lines=True,xtickfmt=None,
             unc_color='k',unc_alpha=0.1,minval=None,maxval=None,public=False,
             weightlinenorm=0.0015,weightlinemin=0,**kwargs):
        plu.setfig(fig)
        if minval is None:
            minval = self.minval
        if maxval is None:
            maxval = self.maxval
        xvals = np.linspace(minval,maxval,1000)
        
        if uncs and self.uncfn_m is None:
            self.bootstrap()

        if log:
            yvals = scale*self(xvals)
            plt.semilogx(10**xvals,yvals,label=label,**kwargs)
            if lines:
                if self.posteriors is not None:
                    for p,w in zip(self.posteriors,self.weights):
                        if p.pctile(0.5) < self.maxval:
                            plt.axvline(p.pctile(0.5),color='r',lw=1,ymax=max(0.01,w*0.0015))
                else:
                    for d,w in zip(self.data,self.weights):
                        if d < self.maxval:
                            plt.axvline(10**d,color='r',lw=1,ymax=max(0.01,w*0.0015))
            if uncs:
                hi = scale*self.uncfn_p(xvals)
                lo = scale*self.uncfn_m(xvals)
                plt.fill_between(10**xvals,hi,lo,color=unc_color,alpha=unc_alpha)
                    
            plt.xlim((10**minval,10**maxval))
            ax = plt.gca()
            if xtickfmt is not None:
                ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(xtickfmt))
        else:
            plt.plot(xvals,scale*self(xvals),label=label,**kwargs)
            if lines:
                if self.posteriors is not None:
                    for p,w in zip(self.posteriors,self.weights):
                        plt.axvline(p.pctile(0.5),color='r',lw=1,ymax=max(0.01,w*0.0015))
                else:
                    for d,w in zip(self.data,self.weights):
                        plt.axvline(d,color='r',lw=1,ymax=max(0.01,w*0.0015))
            if uncs:
                if self.uncfn_p is None:
                    self.bootstrap()
                hi = scale*self.uncfn_p(xvals)
                lo = scale*self.uncfn_m(xvals)
                plt.fill_between(xvals,hi,lo,color=unc_color,alpha=0.2)
                
            plt.xlim((minval,maxval))

        plt.ylim(ymin=0)
        plt.ylabel('Probability Density')
        plt.yticks([])

    def resample(self,N):
        u = rand.random(size=N)
        vals = np.linspace(self.minval,self.maxval,1e4)
        pdf = self.pdf(vals).clip(0)
        cdf = pdf.cumsum()
        ys = cdf/cdf.max()
        inds = np.digitize(u,ys)
        return vals[inds]
        pass
        

class RadiusKDE_FromSurvey(WeightedKDE):
    def __init__(self,survey,widthfactor=1,maxp=None,simple=False,simple_thresh=False,
                 Pvals=None,fps_initial=None,usefpp=True,truedist=None,**kwargs):
        self.survey = survey

        self.Pvals = Pvals
        self.fps_initial = fps_initial
        self.fp_specifics = fps_initial
        if Pvals is not None:
            self.fpps = 1./(1 + self.fps_initial*self.Pvals)
        else:
            self.fpps = np.zeros(len(self.survey.detections))

        rps = survey.detections['Rp']

        if 'posteriors' in kwargs:
            for i,p in enumerate(kwargs['posteriors']):
                rps[i] = p.pctile(0.5)
            self.survey.detections['Rp'] = rps
            print 'planet radii assigned to be medians of provided posteriors'

        self.simple = False
        self.simple_thresh = False
        if simple:
            self.simple = True
        if simple_thresh:
            self.simple_thresh = True
            self.simple = False

        self.weights_simple = 1./(survey.detections['transprob'] * survey.etadisc_simple)
        self.weights_simple_thresh = 1./(survey.detections['transprob'] * survey.etadisc_simple_thresh)

        if self.simple:
            weights = self.weights_simple
            self.etadiscs = survey.etadisc_simple.copy()
        elif self.simple_thresh:
            weights = self.weights_simple_thresh
            self.etadiscs = survey.etadisc_simple_thresh.copy()
        else:
            if 'posteriors' not in kwargs:
                etadiscs = survey.etadisc_of_r(rps)
                weights = 1./(survey.survey_transitprob * etadiscs)
            else:
                weights = np.zeros(len(rps))
                etadiscs = weights*0
                for i,p in enumerate(kwargs['posteriors']):
                    fn = lambda x: survey.etadisc_of_r(x)*p(x)
                    etadisc = quad(fn,0,20,full_output=1)[0]
                    #print etadisc
                    weights[i] = 1/(survey.survey_transitprob * etadisc)
                    etadiscs[i] = etadisc
            self.etadiscs_normal = etadiscs.copy()
            self.etadiscs = etadiscs.copy()

        self.weights_normal = weights.copy()
        self.weights_nofpp = weights.copy()
        if usefpp:
            weights *= (1 - self.fpps)
        


        if 'dRp' in survey.detections.colnames:
            widths = survey.detections['dRp']
        elif 'dRp_p' in survey.detections.colnames:
            widths = (survey.detections['dRp_p'] + survey.detections['dRp_n'])/2.

        self.maxp = maxp 

        WeightedKDE.__init__(self,rps,weights,widths=widths,widthfactor=widthfactor,
                             norm=len(survey.targets),normed=False,truedist=truedist,**kwargs)

        if self.Pvals is not None and usefpp:
            self.iterate_fpp()

    def simulate_detections(self,full=False,return_full=True):
        if not full:
            N = poisson.rvs(self.survey.survey_transitprob *
                            len(self.survey.targets) * self.integral)  #this many transiting planets

            inds = rand.randint(len(self.survey.simtable),size=N)
            pers = self.survey.simtable['P'][inds]
            rps = self.resample(N)
            SNRs = self.survey.simtable['snr'][inds] * rps**2
            transprobs = self.survey.simtable['transprob'][inds]
            
            pr_detect = self.survey.etafn(SNRs)
            u = rand.random(size=N)
            detected = u < pr_detect
            w_detected = np.where(detected)
            
            
            detections = Table(data=[rps[w_detected],pers[w_detected],SNRs[w_detected],
                                     transprobs[w_detected]],
                               names=['Rp','P','SNR','transprob'])
            return detections

        else:
            N = poisson.rvs(len(self.survey.targets) * self.integral) # total number of planets

            target_inds = rand.randint(len(self.survey.targets),size=N)

            targets = self.survey.targets[target_inds]

            pers = 10**self.survey.logperdist.resample(N)
            rps = self.resample(N)
            
            semimajors = semimajor(pers,targets['M'])
            incs = np.arccos(rand.random(N))
            bs = semimajors*AU*np.cos(incs) / (targets['R']*RSUN)
        
            SNRs = bs*0

            tra = (bs < 1)
            SNRs[np.where(~tra)] = 0
            wtra = np.where(tra)
            SNRs[wtra] = transitSNR(pers[wtra],rps[wtra],targets['R'][wtra],
                                        targets['M'][wtra],bs[wtra],targets['Tobs'][wtra],
                                        targets['noise'][wtra],targets['u1'][wtra],
                                        targets['u2'][wtra],npts=50,noise_timescale=3,
                                    force_1d=True)
        
            pr_detect = self.survey.etafn(SNRs)
            u = rand.random(size=N)
            detected = u < pr_detect
            w_detected = np.where(detected)
        
            names = np.arange(detected.sum())+1

            alltransiting = Table(data=[rps[wtra],pers[wtra],SNRs[wtra],bs[wtra],u[wtra],
                                        targets['R'][wtra],targets['Tobs'][wtra],
                                        targets['noise'][wtra]],
                                  names=['Rp','P','SNR','b','u','R','Tobs','noise'])

            semimajors = semimajor(pers[w_detected],targets['M'][w_detected])*AU
            ptrans = targets['R'][w_detected]*RSUN / semimajors

            detections = Table(data=[names,targets['ID'][w_detected],
                                     rps[w_detected],0.2*rps[w_detected],
                                     pers[w_detected],SNRs[w_detected],
                                     bs[w_detected],u[w_detected],
                                     targets['R'][w_detected],targets['Tobs'][w_detected],
                                     targets['noise'][w_detected],ptrans],
                               names=['name','ID','Rp','dRp','P','SNR',
                                      'b','u','R','Tobs','noise','transprob'])
            if return_full:
                return detections,alltransiting
            else:
                return detections

            

    def bootstrap(self,N=1000,npts=500,use_pbar=True,return_vals=False,
                  width=0.15,plot=False,fig=None):

        if plot:
            plu.setfig(fig)

        xs = np.linspace(self.minval,self.maxval,npts)
        tots = np.zeros((N,npts))
        if use_pbar:
            widgets = ['calculating bootstrap variance: ',Percentage(),' ',
                       Bar(marker=RotatingMarker()),' ',ETA()]
            pbar = ProgressBar(widgets=widgets,maxval=N)
            pbar.start()
        for i in np.arange(N):
            detections = self.simulate_detections()
            rps = detections['Rp']
            if self.simple or self.simple_thresh:
                ptrans = detections['transprob']
                etadiscs = self.survey.calc_etadisc_simple(rps=rps,Ps=detections['P'])[0]
                weights = 1./(etadiscs * ptrans)
            else:
                ptrans = self.survey.survey_transitprob
                weights = 1./(self.survey.etadisc_of_r(rps) * ptrans)

            new = WeightedKDE(rps,weights,width*rps,norm=self.norm,
                              minval=self.minval,maxval=self.maxval,normed=self.normed)

            if plot:
                plt.plot(xs,new.evaluate(xs),'k',lw=1,alpha=0.1)

            #inds = rand.randint(self.N,size=Nthis)
            #if self.posteriors is not None:
            #    new = WeightedKDE(self.data[inds],self.weights[inds],
            #                      self.widths[inds],norm=self.norm,posteriors=self.posteriors[inds],
            #                      minval=self.minval,maxval=self.maxval,normed=self.normed)
            #elif self.custom_kernels is not None:
            #    new = WeightedKDE(self.data[inds],self.weights[inds],
            #                      self.widths[inds],norm=self.norm,custom_kernels=self.custom_kernels[inds],
            #                      minval=self.minval,maxval=self.maxval,normed=self.normed)
            #else:
            #    new = WeightedKDE(self.data[inds],self.weights[inds],
            #                      self.widths[inds],norm=self.norm,
            #                      minval=self.minval,maxval=self.maxval,normed=self.normed)

            tots[i,:] = new.evaluate(xs)
                
            if use_pbar and i % 10==0:
                pbar.update(i)
        if use_pbar:
            pbar.finish()
        sorted = np.sort(tots,axis=0)
        pvals = sorted[-N*16/100,:] #84th pctile pts
        medvals = sorted[N/2,:]  #50th pctile pts
        mvals = sorted[N*16/100,:] #16th pctile pts
        
        self.bootstrap_med = interpolate(xs,medvals,s=0)
        self.bias = lambda x: self.bootstrap_med(x) - self.evaluate(x)

        upper = pvals - medvals + self.evaluate(xs) - self.bias(xs)
        lower = medvals - pvals + self.evaluate(xs) - self.bias(xs)
        lower[np.where(lower<0)] = 0
        self.uncfn_p = interpolate(xs,upper,s=0) 
        self.uncfn_m = interpolate(xs,lower,s=0)

        if plot:
            #plt.plot(xs,medvals,'k')
            #plt.plot(xs,medvals,'w--')
            plt.plot(xs,self.evaluate(xs),'w',lw=3)
            plt.plot(xs,self.evaluate(xs),'r',lw=2,label='Original wKDE')

        self._setfns()

        if plot:
            plt.plot(xs,self(xs),'r--',label='De-biased wKDE')

        if return_vals:
            return xs,tots

    def set_transprob_normal(self):
        self.weights = 1./(self.survey.survey_transitprob * self.etadiscs) * (1 - self.fpps)
        self._setfns()

    def set_transprob_simple(self):
        self.weights = 1./(self.survey.detections['transprob'] * self.etadiscs) * (1 - self.fpps)
        self._setfns()

    def set_simple(self,transprob_simple=True):
        if transprob_simple:
            ptrans = self.survey.detections['transprob']
        else:
            ptrans = self.survey.survey_transitprob
        self.etadiscs = self.survey.etadisc_simple.copy()
        self.weights = 1./(self.etadiscs * ptrans) * (1 - self.fpps)
        self.simple = True
        self.simple_thresh = False
        self.simple_transprob = transprob_simple
        self._setfns(remove_bias_correction=True)

    def set_simple_thresh(self,transprob_simple=True):
        if transprob_simple:
            ptrans = self.survey.detections['transprob']
        else:
            ptrans = self.survey.survey_transitprob
        self.etadiscs = self.survey.etadisc_simple_thresh.copy()
        self.weights = 1./(self.etadiscs * ptrans) * (1 - self.fpps)
        self.simple = False
        self.simple_thresh = True
        self.simple_transprob = transprob_simple
        self._setfns(remove_bias_correction=True)

    def set_normal(self):
        self.weights = self.weights_normal * (1-self.fpps)
        self.etadiscs = self.etadiscs_normal.copy()
        self.simple = False
        self.simple_thresh = False
        self.simple_transprob = False
        self._setfns()

    def iterate_fpp(self,tol=1e-3):   
        fps = self.fps_initial
        newfps = np.zeros(fps.shape)
        dfps = newfps*0

        ndfps=10
        while ndfps > 0:
            for i in np.arange(len(fps)):
                newfps[i] = self.fp_specific(self.survey.detections['Rp'][i])
            dfps = np.absolute(fps-newfps)
            #print dfps
            newfpps = 1/(1 + newfps*self.Pvals)
            self.fpps = newfpps
            #set minimum of fpp=0.1 for non-calculated planets
            self.fpps[np.where(~self.fpp_calculated)] = np.clip(self.fpps[np.where(~self.fpp_calculated)],
                                                                0.1,1)
            self.weights = self.weights_nofpp * (1 - self.fpps)
            self._setfns()
            fps = newfps.copy()
            ndfps = (dfps > tol).sum()
            print ndfps,'dfps > %.2e, max=%.1e' % (tol,dfps.max())

        self.fp_specifics = fps

    def fp_specific(self,r,dr=None):
        if dr is None:
            dr = r/3.
        return quad(self,r-dr,r+dr)[0]

    def plot(self,fig=None,rmax=4,lines=True,hist=False,histymax=None,histlabel=True,
             histcolor='b',return_histvals=False,rmin=0.25,histbins=None,public=False,
             label=None,host=None,uncs=False,etalabel=False,npps_label=True,
             return_histax=False,**kwargs):

        plu.setfig(fig)
        if host is None and fig != 0 :
            host = host_subplot(111)#, axes_class=AA.Axes)

        if public:
            pass
            #lines = False
            #hist = True
            #uncs=True

        WeightedKDE.plot(self,fig=0,color='k',label=label,uncs=uncs,lines=lines,**kwargs)
        #WeightedKDE.plot(self,fig=0,label=label,uncs=uncs,lines=lines,**kwargs)
        if public:
            plt.ylabel('Planet Radius Distribution Function')
        else:
            plt.ylabel('Planet Radius Distribution Function $\phi^{%i}_r$' % self.maxp)
        plt.xlabel('Planet Radius [Earth Radii]',labelpad=10)
        if etalabel:
            plt.annotate('Inverse detection efficiencies',
                         xy=(0.55,0.1),bbox=dict(boxstyle='square',fc='w'),
                         fontsize=12,color='r',va='center')

        if npps_label:
            npps = quad(self,0,self.maxval)[0]
            if self.uncfn_p is not None:
                npps_ep = quad(self.uncfn_p,self.minval,self.maxval)[0]-npps
                npps_en = npps-quad(self.uncfn_m,self.minval,self.maxval)[0]
                e_npps = (npps_ep + npps_en)/2
                plt.annotate('%.2f$\pm$%.2f planets per star\n(P $<$ %i days)' % (npps,e_npps,self.maxp)
                             ,xy=(0.65,0.74),xycoords='axes fraction',
                             fontsize=16,ha='center')
            else:
                plt.annotate('%.2f planets per star\n(P $<$ %i days)' % (npps,self.maxp)
                             ,xy=(0.65,0.74),xycoords='axes fraction',
                             fontsize=16,ha='center')
                

        #if lines:
        #    if self.posteriors is not None:
        #        for p,w in zip(self.posteriors,self.weights):
        #            plt.axvline(p.pctile(0.5),color='r',lw=1,ymax=max(0.01,w*0.002))
        #    else:
        #        for r,w in zip(self.data,self.weights):
        #            plt.axvline(r,color='r',lw=1,ymax=max(0.01,w*0.0015))
        if hist:
            if histbins is None:
                histbins = np.logspace(np.log10(0.5),np.log10(4),7)
            rh = host.twinx()
            rh.axis['right'].toggle(all=True)
            #rh.yaxis.set_label_coords(1.08,0.25)
            #rh.set_ylabel('Occurrence Rate')
            vals = self.plot_hist(fig=0,ax=rh,color=histcolor,bins=histbins,
                                  integrate=True,**kwargs)
            if histymax is None:
                histymax = 2*max(vals)
            rh.set_ylim(ymax=histymax)
            rh.yaxis.label.set_color(histcolor)
            rh.set_yticks(np.arange(0.1,max(vals)+0.1,0.1))
            #rh.set_yticks([0.1,0.2,0.3,0.4,0.5])
            rh.tick_params(axis='y', colors=histcolor)
            rh.spines['right'].set_color(histcolor)
            #plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6])
            if histlabel:
                plt.annotate('Avg. # of planets per star, $P$ < %i days' % self.maxp,
                             xy=(1.07,0.80),xycoords='axes fraction',
                             rotation=90,fontsize=14,color='b')
            if return_histvals:
                return rh,vals

        plt.xlim((rmin,rmax))
        if return_histax:
            return rh


    def plot_hist(self,fig=None,bins=np.logspace(np.log10(0.5),np.log10(4),7),color='k',ls='-',
                  ax=None,integrate=False,noplot=False,errorbars=False,simple=False,simple_thresh=False,
                  maxp=None,**kwargs):
        if not noplot:
            plu.setfig(fig)
            if ax is None:
                ax = plt.gca()
        if maxp is None:
            if hasattr(self,'maxp'):
                maxp = self.maxp
            else:
                maxp = np.inf

        inds = np.digitize(self.data,bins)
        vals = []
        errs_lo = []
        errs_hi = []
        if errorbars and self.uncfn_p is None and not simple:
            self.bootstrap()
        for i in np.arange(len(bins)-1)+1:
            if integrate and not simple and not simple_thresh:
                lo = bins[i-1]
                hi = bins[i]
                val = quad(self,lo,hi)[0]
                xs = [lo,hi]
                ys = [val,val]
                if not noplot:
                    ax.plot(xs,ys,color=color,ls=ls,**kwargs)
                vals.append(val)
                if errorbars:
                    errs_lo.append(val - quad(self.uncfn_m,lo,hi)[0])
                    errs_hi.append(quad(self.uncfn_p,lo,hi)[0] - val)
            elif simple:
                if hasattr(self,'weights_simple'):
                    weights_simple = self.weights_simple
                else:
                    weights_simple = self.weights
                w = np.where((inds==i) & (self.survey.detections['P'] < maxp))
                val = weights_simple[w].sum()/self.norm
                dbin = bins[i]-bins[i-1]
                xs = np.array([bins[i-1],bins[i]])
                ys = np.array([val,val])
                if not noplot:
                    ax.plot(xs,ys,color=color,ls=ls,**kwargs)
                vals.append(val)
            elif simple_thresh:
                w = np.where((inds==i) & (self.survey.detections['P'] < maxp))
                val = self.weights_simple_thresh[w].sum()/self.norm
                dbin = bins[i]-bins[i-1]
                xs = np.array([bins[i-1],bins[i]])
                ys = np.array([val,val])
                if not noplot:
                    ax.plot(xs,ys,color=color,ls=ls,**kwargs)
                vals.append(val)
            else:
                w = np.where(inds==i)
                val = self.weights[w].sum()/self.norm
                dbin = bins[i]-bins[i-1]
                xs = np.array([bins[i-1],bins[i]])
                ys = np.array([val,val])
                if not noplot:
                    ax.plot(xs,ys,color=color,ls=ls,**kwargs)
                vals.append(val)
        plt.draw()
        if errorbars:
            return vals,errs_lo,errs_hi
        else:
            return vals

    def plot_bar(self,fig=None,bins=np.arange(0,4.1,0.5),width=0.4,color='k',
                 ax=None,ymax=0.8,**kwargs):
        if ax is None:
            plu.setfig(fig)
            ax = plt.gca()
                       

        vals,errs_lo,errs_hi = self.plot_hist(bins=bins,noplot=True,integrate=True,errorbars=True,)
        barbins = (bins[2:]+bins[1:-1])/2.
        #print barbins
        #print vals
        #print errs_lo
        #print errs_hi

        ax.bar(barbins,vals[1:],width=width,align='center',color='gray',
                edgecolor='k',lw=2,**kwargs)
        ax.errorbar(barbins,vals[1:],yerr=[errs_lo[1:],errs_hi[1:]],color='k',
                     ls='none')#,marker='o',ms=10,mec='k',mfc='w')
        ax.set_xlabel('Planet Radius [Earth Radii]')
        ax.set_ylabel('Mean number of planets per star, P < %i days' % self.maxp)
        npps = quad(self,0,self.maxval)[0]
        npps_ep = quad(self.uncfn_p,0,self.maxval)[0]-npps
        npps_en = npps-quad(self.uncfn_m,0,self.maxval)[0]
        e_npps = (npps_ep + npps_en)/2
        ax.annotate('%.2f$\pm$%.2f planets per star\n(P $<$ %i days)' % (npps,e_npps,self.maxp)
                     ,xy=(0.65,0.74),xycoords='axes fraction',
                     fontsize=16,ha='center')
        ax.set_ylim(ymax=ymax)
        plt.draw()

    #def write_table(self,filename='radfn_table.txt'):
    #    fout = open(filename,'w')
    #    fout.write('#KOI Rp dRp pr_trans eta_disc weight\n')
    #    for k,r,dr,ptr,eta in zip(self.kois,self.data,self.widths,self.transprobs,self.etadiscs):
    #        fout.write('%-12s %.2f %.2f %.3f %.2f %.1f\n' % (k,r,dr,ptr,eta,1./(ptr*eta)))
    #    fout.close()

class WeightedKDE_FromH5(WeightedKDE):
    def __init__(self,filename,minval=0.5,maxval=4,**kwargs):
        self.filename = filename
        fns_df = pd.read_hdf(filename,'functions')
        data_df = pd.read_hdf(filename,'data')
        self.data = np.array(data_df['data'])
        self.weights = np.array(data_df['weights'])
        self.pdf = interpolate(np.array(fns_df['vals']),np.array(fns_df['pdf']),s=0)
        self.bias = interpolate(np.array(fns_df['vals']),np.array(fns_df['bias']),s=0)
        self.uncfn_p = interpolate(np.array(fns_df['vals']),np.array(fns_df['uncfn_p']),s=0)
        self.uncfn_m = interpolate(np.array(fns_df['vals']),np.array(fns_df['uncfn_m']),s=0)
        self.truedist = interpolate(np.array(fns_df['vals']),np.array(fns_df['truedist']),s=0)

        
        
        self.posteriors = None
        self.custom_kernels = None

        self.minval = minval
        self.maxval = maxval

        for kw,val in kwargs.iteritems():
            setattr(self,kw,val)
    
    def __call__(self,x):
        return self.pdf(x)


class RadiusKDE_FromH5(WeightedKDE_FromH5,RadiusKDE_FromSurvey):
    pass





class LogperKDE(WeightedKDE):
    def __init__(self,kois=None,width=None):
        pass

    def plot(self,fig=None,**kwargs):
        WeightedKDE.plot(self,fig,**kwargs)
        plt.xlabel('log $P$')
        plt.ylabel('Probability Density')
        plt.yticks([])

    
    
