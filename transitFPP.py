from __future__ import print_function,division
import logging
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy.random as rand
from consts import *
#import utils
import scipy.stats as stats
import pickle


import h5py

import plotutils as pu
import mpl_toolkits.mplot3d.axes3d as p3
import atpy #,tval,keptoy
#try:
#    import pymc as pm
#except:
#    print 'pymc not loaded!  MCMC will not work.'
from statutils import conf2d
import re,os,sys,os.path
from scipy.interpolate import UnivariateSpline as interpolate
from scipy.ndimage import convolve1d
from scipy.integrate import quad
import atpy
import transit_basic as tr
import acor
try:
    import transit_utils as tru
except ImportError:
    logging.warning('transitFPP: did not import transit_utils.')
import statutils as statu
import pickle
import orbitutils as ou
from scipy.stats import gaussian_kde

from matplotlib import cm

import utils
import planetutils as plu

#DATAROOTDIR = os.environ['KOIDATADIR']

SHORT_MODELNAMES = {'Planets':'pl',
                    'EBs':'eb',
                    'HEBs':'heb',
                    'BEBs':'beb',
                    'Blended Planets':'bpl',
                    'Specific BEB':'sbeb',
                    'Specific HEB':'sheb'}
                        
INV_SHORT_MODELNAMES = {v:k for k,v in SHORT_MODELNAMES.iteritems()}

KEPLERDIR = os.environ['KEPLERDIR']
KOIDATAFOLDER = '%s/data/Batalha12' % KEPLERDIR #os.environ['KOIDATADIR']
FPMODELSDIR = '%s/FPP/models' % KEPLERDIR 
#FPMODELSDIR = os.environ['FPMODELSDIR']

FIELDSDIR = '%s/FPP/fields' % KEPLERDIR
#FIELDSDIR = '%s/TRILEGAL' % os.environ['DROPBOX']

BANDS = ['g','r','i','z','J','H','Ks','Kepler']

#KEPLERDIR = os.environ['KEPLERDIR'] #This should be replaced---currently only used for generic_cc

LHOODCACHEFILE = '%s/FPPcache.dat' % os.environ['FPPDIR']

def loadcache(cachefile=LHOODCACHEFILE):
    """  
    """
    cache = {}
    if os.path.exists(cachefile):
        for line in open(cachefile):
            line = line.split()
            if len(line)==2:
                try:
                    cache[int(line[0])] = float(line[1])
                except:
                    pass
    return cache


    #if os.path.exists(LHOODCACHEFILE):
    #    return pickle.load(open(LHOODCACHEFILE,'rb'))
    #else:
    #    return {}

#LHOODCACHE = loadcache()


def writecache(lhoodcache,cachefile=LHOODCACHEFILE):
    pickle.dump(lhoodcache,open(cachfile,'wb'))

def clearcache():
    os.remove(LHOODCACHEFILE)

def addmags(*mags):
    tot=0
    for mag in mags:
        tot += 10**(-0.4*mag)
    return -2.5*log10(tot)

def dilutedradius(rp,dm,dr=0):
    df = 10**(-0.4*dm)
    newr = rp*(1+df)**(0.5)
    return newr

def fpv_from_powerlaw(rp=1,fp=0.4,alpha=-2,rmin=0.5,rmax=30):
    """ fpv defined using rbin = rp +/- rp/3
    """
    plaw = utils.powerlaw(alpha,rmin,rmax)
    return fp*quad(plaw,2./3*rp,4/3.*rp)[0]

def rand_dRV(n,filename='%s/kepler_absrv.npy' % os.environ['ASTROUTIL_DATADIR']):
    rvs = load(filename)
    ind1 = rand.randint(len(rvs),size=n)
    ind2 = rand.randint(len(rvs),size=n)
    return rvs[ind1]-rvs[ind2]

def randpos_in_circle(n,rad,return_rad=False):
    x = rand.random(n)*2*rad - rad
    y = rand.random(n)*2*rad - rad
    w = where(x**2 + y**2 > rad**2)
    nw = size(w)
    while nw > 0:
        x[w] = rand.random(nw)*2*rad-rad
        y[w] = rand.random(nw)*2*rad-rad
        w = where(x**2 + y**2 > rad**2)
        nw = size(w)
    if return_rad:
        return sqrt(x**2 + y**2)
    else:
        return x,y

class FPPCalculation(object):
    def __init__(self,trsig,popset,lhoodcachefile=None):
        self.trsig = trsig
        self.name = trsig.name
        self.popset = popset
        self._get_priorfactors()
        self._get_keywords()
        self.n = self.popset.n
        self.lhoodcachefile = lhoodcachefile
        for pop in self.popset.poplist:
            pop.lhoodcachefile = lhoodcachefile
        
        #self.calc_lhoods()

    def plotsignal(self,fig=None,saveplot=True,folder='.',figformat='png',**kwargs):
        self.trsig.plot(plot_tt=True,fig=fig,**kwargs)
        if saveplot:
            plt.savefig('%s/signal.%s' % (folder,figformat))
            plt.close()

    def FPPsummary(self,fig=None,figsize=(10,8),folder='.',saveplot=True,starinfo=True,siginfo=True,
                   priorinfo=True,constraintinfo=True,tag=None,simple=False,figformat='png'):
        if simple:
            starinfo = False
            siginfo = False
            priorinfo = False
            constraintinfo = False

        #print('backend being used for summary plot: %s' % matplotlib.rcParams['backend'])

        pu.setfig(fig,figsize=figsize)
        # three pie charts
        priors = []; lhoods = []; Ls = []
        #for model in ['eb','heb','bgeb','bgpl','pl']:
        for model in self.popset.modelnames:
            priors.append(self.prior(model))
            lhoods.append(self.lhood(model))
            Ls.append(priors[-1]*lhoods[-1])
        priors = array(priors)
        lhoods = array(lhoods)
        Ls = array(Ls)
        logging.debug('modelnames={}'.format(self.popset.modelnames))
        logging.debug('priors={}'.format(priors))
        logging.debug('lhoods={}'.format(lhoods))

        #colors = ['b','g','r','m','c']
        nmodels = len(self.popset.modelnames)
        colors = [cm.jet(1.*i/nmodels) for i in range(nmodels)]
        legendprop = {'size':11}

        ax1 = plt.axes([0.15,0.45,0.35,0.43])
        plt.pie(priors/priors.sum(),colors=colors)
        labels = []
        #for i,model in enumerate(['eb','heb','bgeb','bgpl','pl']):
        for i,model in enumerate(self.popset.modelnames):
            labels.append('%s: %.1e' % (model,priors[i]))
        plt.legend(labels,bbox_to_anchor=(-0.25,-0.1),loc='lower left',prop=legendprop)
        plt.title('Priors')
        #p.annotate('*',xy=(0.05,0.41),xycoords='figure fraction')

        ax2 = plt.axes([0.5,0.45,0.35,0.43])
        plt.pie(lhoods/lhoods.sum(),colors=colors)
        labels = []
        #for i,model in enumerate(['eb','heb','bgeb','bgpl','pl']):
        for i,model in enumerate(self.popset.modelnames):
            labels.append('%s: %.1e' % (model,lhoods[i]))
        plt.legend(labels,bbox_to_anchor=(1.25,-0.1),loc='lower right',prop=legendprop)
        plt.title('Likelihoods')

        ax3 = plt.axes([0.3,0.03,0.4,0.5])
        plt.pie(Ls/Ls.sum(),colors=colors)
        labels = []
        #for i,model in enumerate(['eb','heb','bgeb','bgpl','pl']):
        for i,model in enumerate(self.popset.modelnames):
            labels.append('%s: %.3f' % (model,Ls[i]/Ls.sum()))
        plt.legend(labels,bbox_to_anchor=(1.6,0.44),loc='right',prop={'size':14},shadow=True)
        plt.annotate('Final Probability',xy=(0.5,-0.01),ha='center',xycoords='axes fraction',fontsize=18)

        #starpars = 'Star parameters used\nin simulations'
        starpars = ''
        if 'M' in self['heb'].stars.keywords and 'DM_P' in self.keywords:
            starpars += '\n$M/M_\odot = %.2f^{+%.2f}_{-%.2f}$' % (self['M'],self['DM_P'],self['DM_N'])
        else:
            starpars += '\n$(M/M_\odot = %.2f \pm %.2f)$' % (self['M'],0)  #this might not always be right?

        if 'DR_P' in self.keywords:
            starpars += '\n$R/R_\odot = %.2f^{+%.2f}_{-%.2f}$' % (self['R'],self['DR_P'],self['DR_N'])
        else:
            starpars += '\n$R/R_\odot = %.2f \pm %.2f$' % (self['R'],self['DR'])

        if 'FEH' in self.keywords:
            if 'DFEH_P' in self.keywords:
                starpars += '\n$[Fe/H] = %.2f^{+%.2f}_{-%.2f}$' % (self['FEH'],self['DFEH_P'],self['DFEH_N'])
            else:
                starpars += '\n$[Fe/H] = %.2f \pm %.2f$' % (self['FEH'],self['DFEH'])
        for kw in self.keywords:
            if re.search('-',kw):
                try:
                    starpars += '\n$%s = %.2f (%.2f)$ ' % (kw,self[kw],self['COLORTOL'])
                except TypeError:
                    starpars += '\n$%s = %s (%.2f)$ ' % (kw,self[kw],self['COLORTOL'])
                    
        #if 'J-K' in self.keywords:
        #    starpars += '\n$J-K = %.2f (%.2f)$ ' % (self['J-K'],self['COLORTOL'])
        #if 'G-R' in self.keywords:
        #    starpars += '\n$g-r = %.2f (%.2f)$' % (self['G-R'],self['COLORTOL'])
        if starinfo:
            plt.annotate(starpars,xy=(0.03,0.91),xycoords='figure fraction',va='top')

        #p.annotate('Star',xy=(0.04,0.92),xycoords='figure fraction',va='top')

        priorpars = r'$f_{b,short} = %.2f$  $f_{trip} = %.2f$' % (self.priorfactors['fB']*self.priorfactors['f_Pshort'],
                                                                self.priorfactors['ftrip'])
        if 'ALPHA' in self.priorfactors:
            priorpars += '\n'+r'$f_{pl,bg} = %.2f$  $\alpha_{pl,bg} = %.1f$' % (self.priorfactors['fp'],self['ALPHA'])
        else:
            priorpars += '\n'+r'$f_{pl,bg} = %.2f$  $\alpha_1,\alpha_2,r_b = %.1f,%.1f,%.1f$' % \
                         (self.priorfactors['fp'],self['bgpl'].stars.keywords['ALPHA1'],
                          self['bgpl'].stars.keywords['ALPHA2'],
                          self['bgpl'].stars.keywords['RBREAK'])
            
        rbin1,rbin2 = self['RBINCEN']-self['RBINWID'],self['RBINCEN']+self['RBINWID']
        priorpars += '\n$f_{pl,specific} = %.2f, \in [%.2f,%.2f] R_\oplus$' % (self.priorfactors['fp_specific'],rbin1,rbin2)
        priorpars += '\n$r_{confusion} = %.1f$"' % sqrt(self.priorfactors['area']/pi)
        if self.priorfactors['multboost'] != 1:
            priorpars += '\nmultiplicity boost = %ix' % self.priorfactors['multboost']
        if priorinfo:
            plt.annotate(priorpars,xy=(0.03,0.4),xycoords='figure fraction',va='top')
        

        sigpars = ''
        sigpars += '\n$P = %s$ d' % self['P']
        depth,ddepth = self.trsig.depthfit
        sigpars += '\n$\delta = %i^{+%i}_{-%i}$ ppm' % (depth*1e6,ddepth[1]*1e6,ddepth[0]*1e6)
        dur,ddur = self.trsig.durfit
        sigpars += '\n$T = %.2f^{+%.2f}_{-%.2f}$ h' % (dur*24.,ddur[1]*24,ddur[0]*24)
        slope,dslope = self.trsig.slopefit
        sigpars += '\n'+r'$T/\tau = %.1f^{+%.1f}_{-%.1f}$' % (slope,dslope[1],dslope[0])
        sigpars += '\n'+r'$(T/\tau)_{max} = %.1f$' % (self.trsig.maxslope)
        if siginfo:
            plt.annotate(sigpars,xy=(0.81,0.91),xycoords='figure fraction',va='top')
        
            #p.annotate('${}^a$Not used for FP population simulations',xy=(0.02,0.02),
            #           xycoords='figure fraction',fontsize=9)

        constraints = 'Constraints:'
        for c in self.popset.constraints:
            try:
                constraints += '\n  %s' % self['heb'].constraints[c]
            except KeyError:
                constraints += '\n  %s' % self['beb'].constraints[c]                
        if constraintinfo:
            plt.annotate(constraints,xy=(0.03,0.22),xycoords='figure fraction',
                         va='top',color='red')
            
        odds = 1./self.FPP()
        if odds > 1e6:
            fppstr = 'FPP: < 1 in 1e6' 
        else:
            fppstr = 'FPP: 1 in %i' % odds
            
        plt.annotate('$f_{pl,V} = %.3f$\n%s' % (self.fpV(),fppstr),xy=(0.7,0.02),
                     xycoords='figure fraction',fontsize=16,va='bottom')

        plt.suptitle(self.trsig.name,fontsize=22)

        if not simple:
            plt.annotate('n = %.0e' % self.n,xy=(0.5,0.85),xycoords='figure fraction',
                         fontsize=14,ha='center')

        if saveplot:
            if tag is not None:
                plt.savefig('%s/FPPsummary_%s.%s' % (folder,tag,figformat))
            else:
                plt.savefig('%s/FPPsummary.%s' % (folder,figformat))
            plt.close()


    def lhoodplots(self,folder='.',tag=None,figformat='png',**kwargs):
        Ltot = 0
        #print('backend being used for lhoodplots: %s' % matplotlib.rcParams['backend'])
        #for model in ['eb','heb','bgeb','bgpl','pl']:
        for model in self.popset.modelnames:
            Ltot += self.prior(model)*self.lhood(model)
        
        #for model in ['eb','heb','bgeb','bgpl','pl']:
        for model in self.popset.shortmodelnames:
            self.lhoodplot(model,Ltot=Ltot,**kwargs)
            if tag is None:
                plt.savefig('%s/%s.%s' % (folder,model,figformat))
            else:
                plt.savefig('%s/%s_%s.%s' % (folder,model,figformat))
            plt.close()

    def lhoodplot(self,model,suptitle='',**kwargs):
        if suptitle=='':
            suptitle = self[model].model
        self[model].lhoodplot(self.trsig,colordict=self.popset.colordict,
                              suptitle=suptitle,cachefile=self.lhoodcachefile,**kwargs)

    def calc_lhoods(self,verbose=True,**kwargs):
        if verbose:
            logging.info('Calculating likelihoods...')
        for pop in self.popset.poplist:
            L = pop.lhood(self.trsig,**kwargs)
            if verbose:
                logging.info('%s: %.2e' % (pop.model,L))

    def __hash__(self):
        key = 0
        key += hash(self.popset)
        key += hash(self.trsig)
        return key

    def __getitem__(self,model):
        return self.popset[model]

    def piecharts(self,suptitle=None,**kwargs):
        if suptitle is None:
            suptitle = self.trsig.name
        self.popset.piecharts(suptitle=suptitle,**kwargs)
    
    def prior(self,model):
        return self[model].prior

    def lhood(self,model,**kwargs):
        return self[model].lhood(self.trsig,
                                 **kwargs)

    def Pval(self,skipmodels=None,verbose=False):
        Lfpp = 0
        if skipmodels is None:
            skipmodels = []
        if verbose:
            logging.info('evaluating likelihoods for %s' % self.trsig.name)
        
        #for model in ['eb','heb','beb','bgpl']:
        for model in self.popset.modelnames:
            if model=='Planets':
                continue
            if model not in skipmodels:
                prior = self.prior(model)
                lhood = self.lhood(model)
                Lfpp += prior*lhood
                if verbose:
                    logging.info('%s: %.2e = %.2e (prior) x %.2e (lhood)' % (model,prior*lhood,prior,lhood))
        prior = self.prior('pl')
        lhood = self.lhood('pl')
        Lpl = prior*lhood
        if verbose:
            logging.info('planet: %.2e = %.2e (prior) x %.2e (lhood)' % (prior*lhood,prior,lhood))
        return Lpl/Lfpp/self['pl'].priorfactors['fp_specific']

    def fpV(self,FPPV=0.005,skipmodels=None,verbose=False):
        P = self.Pval(skipmodels=skipmodels,verbose=verbose)
        return (1-FPPV)/(P*FPPV)

    def FPP(self,skipmodels=None,verbose=False):
        Lfpp = 0
        if skipmodels is None:
            skipmodels = []
        if verbose:
            logging.info('evaluating likelihoods for %s' % self.trsig.name)
        for model in self.popset.modelnames:
            if model=='Planets':
                continue
            if model not in skipmodels:
                prior = self.prior(model)
                lhood = self.lhood(model)
                Lfpp += prior*lhood
                if verbose:
                    logging.info('%s: %.2e = %.2e (prior) x %.2e (lhood)' % (model,prior*lhood,prior,lhood))
        prior = self.prior('pl')
        lhood = self.lhood('pl')
        Lpl = prior*lhood
        if verbose:
            logging.info('planet: %.2e = %.2e (prior) x %.2e (lhood)' % (prior*lhood,prior,lhood))
        return 1 - Lpl/(Lpl + Lfpp)

    def _get_keywords(self):
        self.popset._get_keywords()
        self.keywords = self.popset.keywords

    def _get_priorfactors(self):
        self.popset._get_priorfactors()
        self.priorfactors = self.popset.priorfactors

    def set_maxrad(self,*args,**kwargs):
        self.popset.set_maxrad(*args,**kwargs)
        self._get_priorfactors()

    def change_prior(self,**kwargs):
        self.popset.change_prior(**kwargs)
        self._get_priorfactors()

    def ruleout_model(self,*args,**kwargs):
        self.popset.ruleout_model(*args,**kwargs)
        self._get_priorfactors()

    def apply_multicolor_transit(self,*args,**kwargs):
        self.popset.apply_multicolor_transit(*args,**kwargs)

    def apply_dmaglim(self,*args,**kwargs):
        self.popset.apply_dmaglim(*args,**kwargs)
        self.dmaglim = self.popset.dmaglim

    def apply_secthresh(self,*args,**kwargs):
        self.popset.apply_secthresh(*args,**kwargs)
        self.secthresh = self.popset.secthresh

    def apply_trend_constraint(self,*args,**kwargs):
        self.popset.apply_trend_constraint(*args,**kwargs)
        self.trend_limit = self.popset.trend_limit
        self.trend_dt = self.popset.trend_dt

    def constrain_Teff(self,T,dT=80,**kwargs):
        self.constrain_property('Teff',measurement=(T,dT),selectfrac_skip=True,**kwargs)

    def constrain_logg(self,g,dg=0.15,**kwargs):
        self.constrain_property('logg',measurement=(g,dg),selectfrac_skip=True,**kwargs)

    def constrain_property(self,*args,**kwargs):
        self.popset.constrain_property(*args,**kwargs)

    def replace_constraint(self,*args,**kwargs):
        self.popset.replace_constraint(*args,**kwargs)

    def remove_constraint(self,*args,**kwargs):
        self.popset.remove_constraint(*args,**kwargs)

    def apply_cc(self,cc):
        logging.info('applying %s band contrast curve to %s.' % (cc.band,self.name))
        self.popset.apply_cc(cc)

    def apply_vcc(self,*args,**kwargs):
        self.popset.apply_vcc(*args,**kwargs)

class PopulationSet(object):
    def __init__(self,poplist,constraints=None,lhoodcachefile=None):
        self.poplist = poplist
        if constraints is None:
            constraints = []
        self.constraints = constraints
        self.hidden_constraints = []
        self.modelnames = []
        self.shortmodelnames = []
        self.lhoodcachefile = lhoodcachefile
        for pop in self.poplist:
            if pop.model in self.modelnames:
                raise ValueError('cannot have more than one model of the same name in PopulationSet')
            self.modelnames.append(pop.model)
            self.shortmodelnames.append(pop.modelshort)

        self.n = len(poplist[0].alldata)

        self.apply_dmaglim()  #a bit of a hack here; this should be slicker...

        self._get_priorfactors()
        self._get_keywords()
        self._set_constraintcolors()

    def add_population(self,pop):
        if pop.model in self.modelnames:
            raise ValueError('%s model already in PopulationSet.' % pop.model)
        self.modelnames.append(pop.model)
        self.shortmodelnames.append(pop.modelshort)
        self.poplist.append(pop)
        self.apply_dmaglim()
        self._get_priorfactors()
        self._get_keywords()
        self._set_constraintcolors()

    def remove_population(self,pop):
        iremove=None
        for i in range(len(self.poplist)):
            if self.modelnames[i]==self.poplist[i].model:
                iremove=i
        if iremove is not None:
            self.modelnames.pop(i)
            self.shortmodelnames.pop(i)
            self.poplist.pop(i)

        self._get_priorfactors()
        self._get_keywords()
        self._set_constraintcolors()

    def __hash__(self):
        key = 0
        for pop in self.poplist:
            key += hash(pop)
        return key

    def __getitem__(self,name):
        if name.upper() in self.keywords:
            return self.keywords[name]
        name = name.lower()
        if name in ['pl','pls']:
            name = 'planets'
        elif name in ['eb','ebs']:
            name = 'ebs'
        elif name in ['heb','hebs']:
            name = 'hebs'
        elif name in ['beb','bebs','bgeb','bgebs']:
            name = 'bebs'
        elif name in ['bpl','bgpl','bpls','bgpls']:
            name = 'blended planets'
        elif name in ['sbeb','sbgeb','sbebs','sbgebs']:
            name = 'specific beb'
        elif name in ['sheb','shebs']:
            name = 'specific heb'
        for pop in self.poplist:
            if name==pop.model.lower():
                return pop
        raise ValueError('%s not in modelnames: %s' % (name,self.modelnames))

    def _set_constraintcolors(self,colors=['g','r','c','m','y','b']):
        self.colordict = {}
        i=0
        n = len(self.constraints)
        for c in self.constraints:
            #self.colordict[c] = colors[i % 6]
            self.colordict[c] = cm.jet(1.*i/n)
            i+=1

    def _get_keywords(self):
        keywords = {}
        for pop in self.poplist:
            for key in pop.stars.keywords:
                val = pop.stars.keywords[key]
                if key.lower() != 'prob' and key.lower() != 'dprob':
                    if key in keywords and keywords[key] != val:
                        logging.warning('keyword %s inconsistent (%s)' % (key,pop.model))
                    keywords[key] = val
        self.keywords = keywords

    def _get_priorfactors(self):
        priorfactors = {}
        for pop in self.poplist:
            for f in pop.priorfactors:
                if f in priorfactors:
                    if pop.priorfactors[f] != priorfactors[f]:
                        raise ValueError('prior factor %s is inconsistent!' % f)
                else:
                    priorfactors[f] = pop.priorfactors[f]
        self.priorfactors = priorfactors

    def piecharts(self,fig=None,suptitle='',**kwargs):
        pu.setfig(fig,figsize=(11,11))
        plt.subplot(221)
        self['eb'].constraint_piechart(fig=0,colordict=self.colordict,title=self['eb'].model,**kwargs)
        plt.subplot(222)
        self['heb'].constraint_piechart(fig=0,colordict=self.colordict,title=self['heb'].model,**kwargs)
        plt.subplot(223)
        self['beb'].constraint_piechart(fig=0,colordict=self.colordict,title=self['beb'].model,**kwargs)
        plt.subplot(224)
        self['bpl'].constraint_piechart(fig=0,colordict=self.colordict,title=self['bpl'].model,**kwargs)
        plt.suptitle(suptitle)

    def ruleout_model(self,model,factor):
        k = {'%s_ruledout' % model:factor}
        self[model].add_priorfactor(**k)
        self._get_priorfactors()

    def change_prior(self,**kwargs):
        for kw,val in kwargs.iteritems():
            if kw=='area':
                logging.warning('cannot change area in this way--use change_maxrad instead')
                continue
            for pop in self.poplist:
                k = {kw:val}
                pop.change_prior(**k)
        self._get_priorfactors()

    def apply_multicolor_transit(self,band,pct):
        if band=='K':
            band = 'Ks'
        if band in ('kep','kepler','Kep'):
            band = 'Kepler'
        if '%s band transit' not in self.constraints:
            self.constraints.append('%s band transit' % band)
        for pop in self.poplist:
            pop.apply_multicolor_transit(band,pct)
        self._set_constraintcolors()

    def set_maxrad(self,newrad):
        if 'Rsky' not in self.constraints:
            self.constraints.append('Rsky')
        for pop in self.poplist:
            if hasattr(pop,'set_maxrad') and not pop.is_specific:
                pop.set_maxrad(newrad)
        self._set_constraintcolors()
        self._get_priorfactors()

    def apply_dmaglim(self,dmaglim=None):
        if 'bright blend limit' not in self.constraints:
            self.constraints.append('bright blend limit')
        for pop in self.poplist:
            if not hasattr(pop,'dmaglim') or pop.is_specific:
                continue
            if dmaglim is None:
                dmag = pop.dmaglim
            else:
                dmag = dmaglim
            pop.set_dmaglim(dmag)
        self.dmaglim = dmaglim
        self._set_constraintcolors()

    def apply_trend_constraint(self,limit,dt):
        if 'RV monitoring' not in self.constraints:
            self.constraints.append('RV monitoring')
        for pop in self.poplist:
            if not hasattr(pop,'dRV'):
                continue
            pop.apply_trend_constraint(limit,dt)
        self.trend_limit = limit
        self.trend_dt = dt
        self._set_constraintcolors()

    def apply_secthresh(self,secthresh):
        if 'secondary depth' not in self.constraints:
            self.constraints.append('secondary depth')
        for pop in self.poplist:
            pop.apply_secthresh(secthresh)
        self.secthresh = secthresh
        self._set_constraintcolors()

    def constrain_property(self,prop,**kwargs):
        if prop not in self.constraints:
            self.constraints.append(prop)
        for pop in self.poplist:
            try:
                pop.constrain_property(prop,**kwargs)
            except AttributeError:
                logging.info('%s model does not have property stars.%s (constraint not applied)' % (pop.model,prop))
        self._set_constraintcolors()

    def replace_constraint(self,name,**kwargs):
        for pop in self.poplist:
            pop.replace_constraint(name,**kwargs)
        if name not in self.constraints:
            self.constraints.append(name)
        self._set_constraintcolors()

    def remove_constraint(self,*names):
        for name in names:
            for pop in self.poplist:
                if name in pop.constraints:
                    pop.remove_constraint(name)
                else:
                    logging.info('%s model does not have %s constraint' % (pop.model,name))
            if name in self.constraints:
                self.constraints.remove(name)
        self._set_constraintcolors()
                    
    def apply_cc(self,cc):
        #if '%s band' % cc.band not in self.constraints:
        #    self.constraints.append('%s band' % cc.band)
        if cc.name not in self.constraints:
            self.constraints.append(cc.name)
        for pop in self.poplist:
            if not pop.is_specific:
                try:
                    pop.apply_cc(cc)
                except AttributeError:
                    logging.info('%s cc not applied to %s model' % (cc.name,pop.model))
        self._set_constraintcolors()

    def apply_vcc(self,vcc):
        if 'secondary spectrum' not in self.constraints:
            self.constraints.append('secondary spectrum')
        for pop in self.poplist:
            if not pop.is_specific:
                try:
                    pop.apply_vcc(vcc)
                except:
                    logging.info('VCC constraint not applied to %s model' % (pop.model))
        self._set_constraintcolors()

class ConstraintDict(dict):
    def __hash__(self):
        hashint = 0
        for name in self:
            hashint += self[name].__hash__()
        return hashint


class Constraint(object):
    def __init__(self,mask,name='',**kwargs):
        self.name = name
        self.ok = mask
        self.wok = where(self.ok)[0]
        self.frac = float(self.ok.sum())/size(mask)
        for kw in kwargs:
            setattr(self,kw,kwargs[kw])

    def __eq__(self,other):
        return hash(self) == hash(other)

    def __ne__(self,other):
        return not self.__eq__(other)

    def __hash__(self):
        key = 0
        key += hash(self.name)
        key += hash(self.wok[0:100].__str__())
        key += hash(self.ok.sum())
        return key

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<%s: %s>' % (type(self),str(self))

class JointConstraintAnd(Constraint):
    def __init__(self,c1,c2,name='',**kwargs):
        self.name = name
        mask = ~(~c1.ok & ~c2.ok)
        Constraint.__init__(self,mask,name=name,**kwargs)

class JointConstraintOr(Constraint):
    def __init__(self,c1,c2,name='',**kwargs):
        self.name = name
        mask = ~(~c1.ok | ~c2.ok)
        Constraint.__init__(self,mask,name=name,**kwargs)

class RangeConstraint(Constraint):
    def __init__(self,vals,lo,hi,name='',**kwargs):
        self.lo = lo
        self.hi = hi
        Constraint.__init__(self,(vals > lo) & (vals < hi),name=name,vals=vals,lo=lo,hi=hi,**kwargs)

    def __str__(self,fmt='%.1e'): #implement default string formatting better.....TODO
        return '%s < %s < %s' % (fmt,self.name,fmt) % (self.lo,self.hi)

class UpperLimit(RangeConstraint):
    def __init__(self,vals,hi,name='',**kwargs):
        RangeConstraint.__init__(self,vals,name=name,lo=-inf,hi=hi,**kwargs)
        
    def __str__(self,fmt='%.1e'):
        return '%s < %s' % (self.name,fmt) % (self.hi)    

class LowerLimit(RangeConstraint):
    def __init__(self,vals,lo,name='',**kwargs):
        RangeConstraint.__init__(self,vals,name=name,lo=lo,hi=inf,**kwargs)

    def __str__(self,fmt='%.1e'):
        return '%s > %s' % (self.name,fmt) % (self.lo)

class MeasurementConstraint(RangeConstraint):
    def __init__(self,vals,val,dval,thresh=3,name='',**kwargs):
        lo = val - thresh*dval
        hi = val + thresh*dval
        RangeConstraint.__init__(self,vals,lo,hi,name=name,val=val,dval=dval,thresh=thresh,**kwargs)

class FunctionLowerLimit(Constraint):
    def __init__(self,xs,ys,fn,name='',**kwargs):
        Constraint.__init__(self,ys > fn(xs),name=name,xs=xs,ys=ys,fn=fn,**kwargs)
    
class FunctionUpperLimit(Constraint):
    def __init__(self,xs,ys,fn,name='',**kwargs):
        Constraint.__init__(self,ys < fn(xs),name=name,xs=xs,ys=ys,fn=fn,**kwargs)
    
class ContrastCurveConstraint(FunctionLowerLimit):
    def __init__(self,rs,dmags,cc,name='CC',**kwargs):
        self.rs = rs
        self.dmags = dmags
        self.cc = cc
        FunctionLowerLimit.__init__(self,rs,dmags,cc,name=name,**kwargs)
        
    def __str__(self):
        return '%s contrast curve' % self.name

    def update_rs(self,rs):
        self.rs = rs
        FunctionLowerLimit.__init__(self,rs,self.dmags,self.cc,name=self.name)
        logging.info('%s updated with new rsky values.' % self.name)

class VelocityContrastCurveConstraint(FunctionLowerLimit):
    def __init__(self,vels,dmags,vcc,name='VCC',**kwargs):
        self.vels = vels
        self.dmags = dmags
        self.vcc = vcc
        FunctionLowerLimit.__init__(self,vels,dmags,vcc,name=name,**kwargs)
        
class VelocityContrastCurve(object):
    def __init__(self,vs,dmags,band='g'):
        self.vs = vs
        self.dmags = dmags
        self.band = band
        if size(vs) > 1:
            self.contrastfn = interpolate(vs,dmags,s=0)
            self.vmax = vs.max()
            self.vmin = vs.min()
        else: #simple case; one v, one dmag
            def cfn(v):
                v = atleast_1d(abs(v))
                dmags = zeros(v.shape)
                dmags[where(v>=self.vs)] = self.dmags
                dmags[where(v<self.vs)] = 0
                return dmags
            self.contrastfn = cfn
            self.vmax = self.vs
            self.vmin = self.vs

    def __call__(self,v):
        v = atleast_1d(abs(v))
        dmags = atleast_1d(self.contrastfn(v))
        dmags[where(v >= self.vmax)] = self.contrastfn(self.vmax)
        dmags[where(v < self.vmin)] = 0
        #put something in here to "extend" beyond rmax?
        return dmags

class ContrastCurve(object):
    def __init__(self,rs,dmags,band,mag=None,name=None):
        """band is self-explanatory; 'mag' is mag of the primary in 'band' """
        if band=='K' or band=="K'":
            band = 'Ks'
        rs = atleast_1d(rs)
        dmags = atleast_1d(dmags)
        self.rs = rs
        self.dmags = dmags
        self.band = band
        self.mag = mag
        self.contrastfn = interpolate(rs,dmags,s=0)
        self.rmax = rs.max()
        self.rmin = rs.min()
        if name is None:
            self.name = '%s band' % self.band
        else:
            self.name = name

    def plot(self,fig=None,**kwargs):
        pu.setfig(fig)
        plt.plot(self.rs,self.dmags,**kwargs)
        plt.title('%s band contrast curve' % self.band)
        plt.gca().invert_yaxis()
        plt.xlabel('Separation [arcsec]')
        plt.ylabel('$\Delta %s$' % self.band)

    def __eq__(self,other):
        return hash(self)==hash(other)

    def __ne__(self,other):
        return not self.__eq__(other)

    def __hash__(self):
        key = 0
        key += hash(str(self.rs[0:20]))
        key += hash(str(self.dmags[0:20]))
        key += hash(self.band)
        key += hash(self.mag)
        return key

    def __call__(self,r):
        r = atleast_1d(r)
        dmags = atleast_1d(self.contrastfn(r))
        dmags[r >= self.rmax] = self.contrastfn(self.rmax)
        dmags[r < self.rmin] = 0
        #put something in here to "extend" beyond rmax?
        return dmags

    def __add__(self,other):
        if type(other) not in [type(1),type(1.),type(self)]:
            raise ValueError('Can only add a number or another ContrastCurve.')
        if type(other) in [type(1),type(1.)]:
            dmags = self.dmags + other
            return ContrastCurve(self.rs,dmags,self.band,self.mag)
            
    def __repr__(self):
        return '<%s: %s>' % (type(self),self.name)

    def power(self,floor=10,rmin=0.1,use_quad=False):
        if use_quad:
            return quad(self,rmin,self.rmax)[0]/((self.rmax-rmin)*floor)
        else:
            rs = np.linspace(rmin,self.rmax,100)
            return np.trapz(self(rs),rs)

class ContrastCurveFromFile(ContrastCurve):
    def __init__(self,filename,band,mag=None):
        rs,dmags = loadtxt(filename,unpack=True)
        if rs[0] > 2:
            rs /= 1000.
        ContrastCurve.__init__(self,rs,dmags,band,mag)
        self.filename = filename

def generic_cc(Kmag=10,dmag=8,band='K'):
    if dmag==8:
        return ContrastCurveFromFile('%s/data/contrast_curves/ex8_K.txt' % KEPLERDIR,band,Kmag)
    elif dmag==4.5:
        return ContrastCurveFromFile('%s/data/contrast_curves/ex4.5_K.txt' % KEPLERDIR,band,Kmag)


class StarPopulation(object):
    def __init__(self,stars,constraints=None,selectfrac_skip=None,distribution_skip=None,
                 **kwargs):
        self.stars = stars
        self.N = len(stars)
        if constraints is None:
            self.constraints = ConstraintDict()
        self.hidden_constraints = ConstraintDict()
        if selectfrac_skip is None:
            self.selectfrac_skip = []
        if distribution_skip is None:
            self.distribution_skip = []
        self._apply_all_constraints()

        if hasattr(self,'maxrad'):
            self.change_maxrad(self.maxrad)

        for kw in kwargs:
            setattr(self,kw,kwargs[kw])

    def __getitem__(self,prop):
        return self.selected[prop]

    def prophist2d(self,propx,propy,logx=False,logy=False,inds=None,fig=None,selected=False,**kwargs):
        pu.setfig(fig)
        if inds is None:
            if selected:
                inds = arange(len(self.selected))
            else:
                inds = arange(len(self.stars))
        if selected:
            xvals = self[propx][inds]
            yvals = self[propy][inds]
        else:
            xvals = self.stars[propx][inds]
            yvals = self.stars[propy][inds]
        if logx:
            xvals = log10(xvals)
        if logy:
            yvals = log10(yvals)
        plot2dhist(xvals,yvals,**kwargs)
        plt.xlabel(propx)
        plt.ylabel(propy)
        #if stylestr is not None:
        #    plt.plot(xvals,yvals,stylestr,**kwargs)
        #else:
        #    plt.plot(xvals,yvals,**kwargs)
        

    def prophist(self,prop,fig=None,log=False,inds=None,selected=False,**kwargs):
        pu.setfig(fig)
        if inds is None:
            if selected:
                inds = arange(len(self.selected))
            else:
                inds = arange(len(self.stars))
        if selected:
            try:
                vals = self[prop][inds]
            except ValueError:
                vals = self.data[prop][inds]
        else:
            try:
                vals = self.stars[prop][inds]
            except ValueError:
                vals = self.alldata[prop][inds]
        if log:
            h = plt.hist(log10(vals),**kwargs)
        else:
            h = plt.hist(vals,**kwargs)
        
        plt.xlabel(prop)

    def constraint_stats(self,primarylist=None):
        if primarylist is None:
            primarylist = []
        n = len(self.stars)
        primaryOK = ones(n).astype(bool)
        tot_reject = zeros(n)
        for name in self.constraints:
            if name in self.selectfrac_skip:
                continue
            c = self.constraints[name]
            if name in primarylist:
                primaryOK &= c.ok
            tot_reject += ~c.ok
        primary_rejected = ~primaryOK
        secondary_rejected = tot_reject - primary_rejected
        lone_reject = {}
        for name in self.constraints:
            if name in primarylist or name in self.selectfrac_skip:
                continue
            c = self.constraints[name]
            lone_reject[name] = ((secondary_rejected==1) & (~primary_rejected) & (~c.ok)).sum()/float(n)
        mult_rejected = (secondary_rejected > 1) & (~primary_rejected)
        not_rejected = ~(tot_reject.astype(bool))
        primary_reject_pct = primary_rejected.sum()/float(n)
        mult_reject_pct = mult_rejected.sum()/float(n)
        not_reject_pct = not_rejected.sum()/float(n)
        tot = 0
        results = {}
        results['pri'] = primary_reject_pct
        tot += primary_reject_pct
        for name in lone_reject:
            results[name] = lone_reject[name]
            tot += lone_reject[name]
        results['multiple constraints'] = mult_reject_pct
        tot += mult_reject_pct
        results['remaining'] = not_reject_pct
        tot += not_reject_pct
        if tot != 1:
            logging.warning('total adds up to: %.2f (%s)' % (tot,self.model))
        return results

    def fraction_complete(self):
        return 1 - self.constraint_stats()
    
    def constraint_piechart(self,primarylist=['secondary depth'],fig=None,title='',colordict=None,legend=True,nolabels=False):
        pu.setfig(fig,figsize=(6,6))
        stats = self.constraint_stats(primarylist=primarylist)
        if len(primarylist)==1:
            primaryname = primarylist[0]
        else:
            primaryname = ''
            for name in primarylist:
                primaryname += '%s,' % name
            primaryname = primaryname[:-1]
        fracs = []
        labels = []
        explode = []
        colors = []
        fracs.append(stats['remaining']*100)
        labels.append('remaining')
        explode.append(0.05)
        colors.append('b')
        if 'pri' in stats and stats['pri']>=0.005:
            fracs.append(stats['pri']*100)
            labels.append(primaryname)
            explode.append(0)
            if colordict is not None:
                colors.append(colordict[primaryname])
        for name in stats:
            if name == 'pri' or name == 'multiple constraints' or name == 'remaining':
                continue
            #if stats[name] < 0.005:
            #    continue
            fracs.append(stats[name]*100)
            labels.append(name)
            explode.append(0)
            if colordict is not None:
                colors.append(colordict[name])
            
        if stats['multiple constraints'] >= 0.005:
            fracs.append(stats['multiple constraints']*100)
            labels.append('multiple constraints')
            explode.append(0)
            colors.append('w')

        autopct = '%1.1f%%'

        if nolabels:
            labels = None
        if legend:
            legendlabels = []
            for i,l in enumerate(labels):
                legendlabels.append('%s (%.1f%%)' % (l,fracs[i]))
            labels = None
            autopct = ''
        if colordict is None:
            plt.pie(fracs,labels=labels,autopct=autopct,explode=explode)
        else:
            plt.pie(fracs,labels=labels,autopct=autopct,explode=explode,colors=colors)
        if legend:
            plt.legend(legendlabels,bbox_to_anchor=(-0.05,0),loc='lower left',prop={'size':10})
        plt.title(title)

    def SED(self,ind,fig=None,bands=['g','r','i','z','J','H','Ks'],**kwargs):
        totmags = {}
        for b in bands:
            if '%s_tot' % b in self.stars.names:  #relevant for EB, HEB
                totmags[b] = self.stars['%s_tot' % b][ind]
            elif '%s_1' % b in self.stars.names: #relevant for BEB
                totmags[b] = addmags(self.stars['%s_1' % b][ind],self.stars['%s_2' % b][ind])
            elif b in self.stars.names: #BGPL
                totmags[b] = self.stars[b][ind]
        return totmags

    def _apply_all_constraints(self):
        #distinguish between which stars are kept for distribution and which are counted, etc.
        n = float(len(self.stars))
        self.distok = ones(len(self.stars)).astype(bool)
        self.countok = ones(len(self.stars)).astype(bool)

        for name in self.constraints:
            c = self.constraints[name]
            if c.name not in self.distribution_skip:
                self.distok &= c.ok
            if c.name not in self.selectfrac_skip:
                self.countok &= c.ok

        self.selected = self.stars.rows(where(self.distok)[0])
        self.selectfrac = self.countok.sum()/n

    def apply_trend_constraint(self,limit,dt):
        """Only works if object has dRV method and plong attribute; limit in km/s"""
        dRVs = absolute(self.dRV(dt))
        c1 = UpperLimit(dRVs, limit)
        c2 = LowerLimit(self.stars.Plong, dt*4)
        #self.apply_constraint(UpperLimit(dRVs, limit,name='RV trend'))
        #self.apply_constraint(LowerLimit(self.stars.Plong, dt*4,name='P(EB) < RV dt*4'))
        self.apply_constraint(JointConstraintOr(c1,c2,name='RV monitoring',Ps=self.stars.Plong,dRVs=dRVs))

    def apply_cc(self,cc):
        """Only works if object has rsky, dmags attributes
        """
        rs = self.rsky
        dmags = self.dmags(cc.band)
        self.apply_constraint(ContrastCurveConstraint(rs,dmags,cc,name=cc.name))

    def apply_vcc(self,vcc):
        """only works if has dmags and RV attributes"""
        if type(vcc)==type((1,)):
            vcc = VelocityContrastCurve(*vcc)
        dmags = self.dmags(vcc.band)
        rvs = self.RV
        self.apply_constraint(VelocityContrastCurveConstraint(rvs,dmags,vcc,name='secondary spectrum'))
        
    def apply_constraint(self,constraint,selectfrac_skip=False,distribution_skip=False,overwrite=False):
        if constraint.name in self.constraints and not overwrite:
            logging.info('constraint already applied: %s' % constraint.name)
            return
        self.constraints[constraint.name] = constraint
        if selectfrac_skip:
            self.selectfrac_skip.append(constraint.name)
        if distribution_skip:
            self.distribution_skip.append(constraint.name)

        self._apply_all_constraints()

    def replace_constraint(self,name,selectfrac_skip=False,distribution_skip=False):
        if name in self.hidden_constraints:
            c = self.hidden_constraints[name]
            self.apply_constraint(c,selectfrac_skip=selectfrac_skip,distribution_skip=distribution_skip)
            del self.hidden_constraints[name]

    def remove_constraint(self,name):
        if name in self.constraints:
            self.hidden_constraints[name] = self.constraints[name]
            del self.constraints[name]
            if name in self.distribution_skip:
                self.distribution_skip.remove(name)
            if name in self.selectfrac_skip:
                self.selectfrac_skip.remove(name)
            self._apply_all_constraints()
        else:
            logging.info('Constraint %s does not exist.' % (name))

    def constrain_property(self,prop,lo=-inf,hi=inf,measurement=None,selectfrac_skip=False,distribution_skip=False,thresh=3):
        if prop in self.constraints:
            logging.info('re-doing %s constraint' % prop)
            self.remove_constraint(prop)
        if measurement is not None:
            val,dval = measurement
            self.apply_constraint(MeasurementConstraint(getattr(self.stars,prop),val,dval,name=prop,thresh=thresh),
                                                        selectfrac_skip=selectfrac_skip,
                                                        distribution_skip=distribution_skip)
        else:
            self.apply_constraint(RangeConstraint(getattr(self.stars,prop),lo=lo,hi=hi,name=prop),
                                  selectfrac_skip=selectfrac_skip,distribution_skip=distribution_skip)

    def set_dmaglim(self,dmaglim):
        if not (hasattr(self,'blendmag') and hasattr(self,'dmaglim')):
            return
        self.dmaglim = dmaglim
        self.apply_constraint(LowerLimit(self.dmags(),self.dmaglim,name='bright blend limit'),overwrite=True)
        self._apply_all_constraints()  #not necessary?

    def set_maxrad_old(self,maxrad):
        if not hasattr(self,'maxrad'):
            return
        else:
            print('%s: setting maxrad: %.2f' % (self.model,maxrad))
            self.maxrad = maxrad
            if hasattr(self,'d'):
                self.rsky = self.stars.rad/self.d #self.stars.rad in projected AU
            else:
                self.rsky = self.stars.rad*maxrad #self.stars.rad random from 0 to 1
            #self.rsky = randpos_in_circle(len(self.stars),rad=maxrad,return_rad=True)
        for name in self.constraints:
            c = self.constraints[name]
            if type(c) is ContrastCurveConstraint:
                c.update_rs(self.rsky)
        self.apply_constraint(UpperLimit(self.rsky,maxrad,name='Rsky'),overwrite=True)
        self._apply_all_constraints()   #not necessary?

    #visualize constraints
    def visualize_constraints(self,fig=None):
        pass


class BGStarPopulation(StarPopulation):
    def __init__(self,stars,mags=None,maxrad=4,density=None):
        """mags should be a dictionary of the magnitudes of the "primary" star.
        """
        self.mags = mags
        if density is None:
            self.density = len(stars)/(3600.**2) #default is for TRILEGAL sims to be 1deg^2
        else:
            self.density = density

        self.rsky = randpos_in_circle(len(stars),maxrad,return_rad=True)
        self.maxrad = maxrad
        StarPopulation.__init__(self,stars)

    def change_maxrad(self,maxrad):
        self.rsky *= maxrad/self.maxrad
        self.maxrad = maxrad

    def dmags(self,band):
        if self.mags is None:
            raise ValueError('dmags not defined because primary mags are not defined for this population.')
        return self.stars[band] - self.mags[band]

    def dmag_lhood(self,dmag,band):
        kde = self.dmags(band)
        return kde(dmag)

    def mag_lhood(self,mag,band):
        wok = where(~isnan(self.stars[band]) & ~isinf(self.stars[band])) #why should there be nans or infs?
        kde = gaussian_kde(self.stars[band][wok])
        return kde(mag)

    def mag_color_lhood(self,mag,band,color,bands):
        m = re.search('(\w+)-(\w+)',bands)
        if m:
            b1 = m.group(1)
            b2 = m.group(2)

        colors = self.stars[b1] - self.stars[b2]
        mags = self.stars[band]
        pts = array([mags,colors])
        kde = gaussian_kde(pts)
        return kde([mag,color])
        

    def rsky_lhood(self,rsky):
        return self.density*2*pi*rsky



class BinaryPopulation(StarPopulation):
    def __init__(self,stars,cc=None,d=None,maxrad=inf,Kmag=None,vcc=None,M=None,dM=0.05,Teff=None,logg=None,mags=None):    
        
        self.stars = stars

        self.mags = mags

        if d is None:
            if Kmag is None:
                if 'Ks' not in mags:
                    Kmag=12.
                else:
                    Kmag = mags['Ks']
            MK = stars.Ks_tot
            mK = Kmag
            d = 10**(1+(mK-MK)/5.)

        self.d = d
        self.distmod = 5*log10(d) - 5

        self.rsky = self.stars.rsky/self.d
        self.RV = self.stars.RV

        self.orbpop = ou.OrbitPopulation(stars.MA,stars.MB,stars.P,eccs=stars.ecc,
                                         mean_anomalies=stars.Manomaly,
                                         obsx=stars.obsx,obsy=stars.obsy,obsz=stars.obsz)

        StarPopulation.__init__(self,stars)

        
        return


        self.rsky = self.stars.rad/self.d
        self.RV = self.stars.RV

        constraints = {}
        selectfrac_skip = []
        distribution_skip = []

        if Teff is not None:
            T,dT = Teff
            constraints['Teff'] = MeasurementConstraint(stars.Teff,T,dT,name='Teff')
            selectfrac_skip.append('Teff')
        if logg is not None:
            g,dg = logg
            constraints['logg'] = MeasurementConstraint(stars.logg,g,dg,name='logg')
            selectfrac_skip.append('logg')

        if cc is not None:
            if size(cc)==1 and type(cc) != type([]):
                cc = [cc]
            for c in cc:
                dmags = stars['%s_B' % c.band] - stars['%s_A' % c.band]
                constraints['%s band' % c.band] = ContrastCurveConstraint(self.rsky,dmags,c,name='%s band' % c.band)
                
        if vcc is not None:
            if type(vcc)==type((1,)):
                vcc = VelocityContrastCurve(*vcc)
            dmags = stars['%s_B' % vcc.band] - stars['%s_A' % vcc.band]
            constraints['secondary spectrum'] = VelocityContrastCurveConstraint(self.RV,dmags,vcc,name='secondary spectrum')

        constraints['Rsky'] = UpperLimit(self.rsky,maxrad,name='Rsky')
        distribution_skip.append('Rsky')

        StarPopulation.__init__(self,stars,constraints,selectfrac_skip=selectfrac_skip,distribution_skip=distribution_skip)


    #def apply_cc(self,cc):
    #    """apply a cc"""
    #    dmags = self.stars['%s_B' % cc.band] - self.stars['%s_A' % cc.band]
    #    self.apply_constraint(ContrastCurveConstraint(self.rsky,dmags,cc,name='%s band' % cc.band))

    #def apply_vcc(self,vcc):
    #    """apply a vcc"""
    #    if type(vcc)==type((1,)):
    #        vcc = VelocityContrastCurve(*vcc)

    #    dmags = self.stars['%s_B' % vcc.band] - self.stars['%s_A' % vcc.band]
    #    self.apply_constraint(VelocityContrastCurveConstraint(self.RV,dmags,vcc,name='secondary spectrum'))

    def dRV(self,dt):
        return self.orbpop.dRV(dt)

    def set_maxrad(self,maxrad):
        self.maxrad = maxrad
        self.apply_constraint(UpperLimit(self.rsky,maxrad,name='Rsky'),overwrite=True)
        self._apply_all_constraints()

    def dmags(self,band):
        mag2 = self.stars['%s_B' % band]
        mag1 = self.stars['%s_A' % band]
        return mag2-mag1

    def rsky_lhood(self,rsky,rmax=None,dr=0.02,smooth=0.1):
        if rmax is None:
            inds = argsort(self.rsky)
            rmax = self.rsky[inds][int(0.99*len(self.rsky))]
        rs = arange(0,rmax,dr)
        dist = utils.Hist_Distribution(self.rsky,bins=rs,maxval=rmax,smooth=smooth)
        return dist(rsky)

    def mag_color_lhood(self,mag,band,color,bands):
        m = re.search('(\w+)-(\w+)',bands)
        if m:
            b1 = m.group(1)
            b2 = m.group(2)
        colors = self.stars['%s_B'% b1] - self.stars['%s_B' % b2] #finish this
        mags = self.stars['%s_B' % band] + self.distmod
        pts = array([mags,colors])
        kde = gaussian_kde(pts)
        return kde([mag,color])

    def mag_lhood(self,mag,band):
        mags = self.stars['%s_B' % band] + self.distmod
        kde = gaussian_kde(mags)
        return kde(mag)


    def rp_dilutedpdf(self,r=1,dr=0.1,fb=0.4,fig=None,plot=True,band='Kepler',rmax=20,
                      allpdfs=False,rmaxplot=None,npts=200,**kwargs):
        n = len(self.selected)
        fb *= self.selectfrac

        simr = rand.normal(size=n)*dr + r
        mainpdf = utils.gaussian(r,dr,norm=1-fb)
    
        dmag = self.selected['%s_B' % band] - self.selected['%s_A' % band]
        diluted1 = utils.kde(dilutedradius(simr,dmag),norm=fb/2,adaptive=False)
        diluted2 = utils.kde(dilutedradius(simr,-dmag),norm=fb/2,adaptive=False)
        diluted2.renorm((fb/2)**2/quad(diluted2,0,rmax)[0])

        totpdf = mainpdf + diluted1 + diluted2

        if plot:
            pu.setfig(fig)
            if rmaxplot is None:
                rmaxplot = r*2
            rs = linspace(0,rmaxplot,npts)
            plt.plot(rs,totpdf(rs),**kwargs)
            if allpdfs:
                plt.plot(rs,mainpdf(rs))
                plt.plot(rs,diluted1(rs))
                plt.plot(rs,diluted2(rs))
        if allpdfs:
            return totpdf,mainpdf,diluted1,diluted2
        else:
            return totpdf

class EclipsePopulation(StarPopulation):
    def __init__(self,stars,data,P=None,model='',priorfactors=None,lhoodcachefile=None):
        self.alldata = data  #data is trapezoidal parameters + secondary eclipse depths
        self.data = data
        self.P = P
        self.model = model
        try:
            self.modelshort = SHORT_MODELNAMES[model]
            if hasattr(self,'index'):
                self.modelshort += '-{}'.format(self.index)
        except KeyError:
            raise KeyError('No short name for model: %s' % model)
        if priorfactors is None:
            priorfactors = {}
        self.priorfactors = priorfactors

        self.lhoodcachefile = lhoodcachefile

        self.is_specific = False

        StarPopulation.__init__(self,stars)

        self.is_ruled_out = False

        if len(self.data)==0:
            raise EmptyPopulationError('Zero elements in %s population.' % model)

        self.make_kdes()
        
        #self.lhoods = {}

    def apply_multicolor_transit(self,band,pct):
        depthratio = self.transitdepth_in_band(band)/self.alldata['depth']
        constraint = RangeConstraint(depthratio,1-pct,1+pct,name='%s band transit' % band)
        self.apply_constraint(constraint,overwrite=True)

    def fluxfrac_eclipsing(self,band=None):
        return ones(len(self.alldata))

    #def fluxfrac_blended(self,band=None):
    #    return zeros(self.stars.N)

    def transitdepth_in_band(self,band):
        return self.alldata['depth'] * (self.fluxfrac_eclipsing(band)/self.fluxfrac_eclipsing(self.band))

    def __getitem__(self,prop):
        if prop in self.stars.names:
            return StarPopulation.__getitem__(self,prop)
        else:
            return self.data[prop]

    def __hash__(self):
        key = 0
        #not using priorfactors to uniquely identify population, because it has nothing to do with likelihood calculation
        #for name in self.priorfactors:
        #    key += hash(name)
        #    key += hash(self.priorfactors[name])
        key += hash(self.constraints)
        for name in self.alldata.names:
            key += hash(self.alldata[name][0:100].__str__())
        return key

    def prophist(self,prop,dur_range=(-inf,inf),dep_range=(-5,0),slope_range=(2,inf),**kwargs):
        inds = self.select_region(dur_range=dur_range,dep_range=dep_range,slope_range=slope_range,return_inds=True)
        StarPopulation.prophist(self,prop,inds=inds,**kwargs)

    def prophist2d(self,propx,propy,dur_range=(-inf,inf),dep_range=(-5,0),slope_range=(2,inf),**kwargs):
        inds = self.select_region(dur_range=dur_range,dep_range=dep_range,slope_range=slope_range,return_inds=True)
        StarPopulation.prophist2d(self,propx,propy,inds=inds,**kwargs)

    def select_best(self,trsig,sig=1,dursig=None,depsig=None,slopesig=None,debug=False,plot=False,fig=None):
        dur,ddur = trsig.durfit
        depth,ddepth = trsig.depthfit
        slope,dslope = trsig.slopefit
        if dursig is None:
            dursig = sig
        if depsig is None:
            depsig = sig
        if slopesig is None:
            slopesig = sig
        dur_range = (dur-dursig*ddur[0],dur+dursig*ddur[1])
        dep_range = (log10(depth-dursig*ddepth[0]),log10(depth+dursig*ddepth[1]))
        slope_range = (slope-slopesig*dslope[0],slope+slopesig*dslope[1])
        if debug:
            print(dur_range)
            print(dep_range)
            print(slope_range)            
        logging.debug(dur_range)
        logging.debug(dep_range)
        logging.debug(slope_range)
        return self.select_region(dur_range=dur_range,dep_range=dep_range,slope_range=slope_range)

    def select_region(self,prop=None,dur_range=(-inf,inf),dep_range=(-5,0),slope_range=(2,inf),return_inds=True):
        dep = log10(self.alldata.depth)
        dur = self.alldata.duration
        slope = self.alldata.slope
        w = where((dep > dep_range[0]) & (dep < dep_range[1]) & (dur > dur_range[0]) &
                  (dur < dur_range[1]) & (slope > slope_range[0]) & (slope < slope_range[1]) &
                  self.distok)[0]
        if return_inds:
            return w
        else:
            return self[prop][w]

    def apply_secthresh(self,secthresh):
        #print 'enforcing secdepth < %.2e for %s model.' % (secthresh,self.model)
        constraint = UpperLimit(self.alldata.secdepth,secthresh,name='secondary depth')
        self.secthresh = secthresh
        self.apply_constraint(constraint,overwrite=True)

    def add_priorfactor(self,**kwargs):
        for kw in kwargs:
            if kw in self.priorfactors:
                logging.error('%s already in prior factors for %s.  use change_prior function instead.' % (kw,self.model))
                continue
            else:
                self.priorfactors[kw] = kwargs[kw]
                logging.info('%s added to prior factors for %s' % (kw,self.model))
        self._calculate_prior()

    def change_prior(self,**kwargs):
        for kw in kwargs:
            if kw == 'area':
                logging.error('cannot change area in this way--use change_maxrad instead')
            else:
                if kw in self.priorfactors:
                    self.priorfactors[kw] = kwargs[kw]
                    logging.info('{0} changed to {1} for {2} model'.format(kw,kwargs[kw],self.model))
        self._calculate_prior()

    #def set_maxrad(self,maxrad):
    #    StarPopulation.set_maxrad(self,maxrad)
    #    self.priorfactors['area'] = pi*maxrad**2
    #    self._calculate_prior()

    def _calculate_prior(self):
        prior = self.stars.keywords['PROB']*self.selectfrac
        for f in self.priorfactors:
            prior *= self.priorfactors[f]
        self.prior = prior

    def _apply_all_constraints(self):
        StarPopulation._apply_all_constraints(self)
        self.data = self.alldata.rows(where(self.distok)[0])
        if len(self.data) < 2:
            #raise EmptyPopulationError('Zero elements in %s population.' % self.model)
            self.is_ruled_out = True

        else:
            self.is_ruled_out = False
            self.make_kdes()
        #self.lhoods = {}
        self._calculate_prior()

    def make_kdes(self,add_sig=None,method='scott'):

        if add_sig is not None:
            extra_dur = [add_sig.dur]
            extra_depth = [add_sig.depth]
            extra_slope = [add_sig.slope]
        else:
            extra_dur,extra_depth,extra_slope = ([],[],[])

        durs = concatenate((absolute(self.data.duration),extra_dur))
        deps = concatenate((self.data.depth,extra_depth))
        deps[where(deps<=0)] = 1e-10
        logdeps = log10(deps)
        slopes = concatenate((self.data.slope,extra_slope))

        if self.P is not None:
            wok = where((slopes > 0) & (durs > 0) & (durs < self.P))
        else: 
            wok = where((slopes > 0) & (durs > 0))

        if size(wok) < 2:
            raise EmptyPopulationError('< 2 valid systems in population.')

        self.depths = deps[wok]
        self.durs = durs[wok]
        self.slopes = slopes[wok]

        self.durkde = stats.gaussian_kde(durs[wok])
        self.depkde = stats.gaussian_kde(deps[wok])
        self.logdepkde = stats.gaussian_kde(logdeps[wok])
        self.slopekde = stats.gaussian_kde(slopes[wok])
        self.logslopekde = stats.gaussian_kde(log10(slopes[wok]))

        points = array([durs[wok],deps[wok]])
        self.durdepthkde = stats.gaussian_kde(points)
        points = array([durs[wok],log10(deps[wok])])
        self.durlogdepthkde = stats.gaussian_kde(points)

        points = array([durs[wok],slopes[wok]])
        self.durslopekde = stats.gaussian_kde(points)
        points = array([durs[wok],log10(slopes[wok])])
        self.durlogslopekde = stats.gaussian_kde(points)

        points = array([durs[wok],log10(deps[wok]),slopes[wok]])
        self.fullkde = stats.gaussian_kde(points)
        points = array([durs[wok],log10(deps[wok]),log10(slopes[wok])])
        self.fullkde_logslope = stats.gaussian_kde(points)

        self.kdes = [self.durkde,self.depkde,self.logdepkde,self.slopekde,self.logslopekde,self.durdepthkde,self.durlogdepthkde,self.durslopekde,self.durlogslopekde,self.fullkde,self.fullkde_logslope]
        for kde in self.kdes:
            if method=='silverman':
                kde.covariance_factor = kde.silverman_factor


    def prilc(self,i):
        return Modelfrompop(self,i)

    def seclc(self,i):
        return Modelfrompop(self,i,sec=True)

    def plotlc(self,i,plotpp=False,plot_tt=True,fig=None):
        diluted = hasattr(self,'dilution_factor')
        mod = Modelfrompop(self,i,diluted=diluted)
        smod = Modelfrompop(self,i,sec=True,diluted=diluted)
        pu.setfig(fig)
        mod.plot(plotpp=plotpp,plot_tt=plot_tt,fig=0,color='b')
        smod.plot(plotpp=plotpp,plot_tt=plot_tt,fig=0,color='g')

    def plot(self,*args,**kwargs):
        self.plotdurdepthkde(*args,**kwargs)

    def hist2d(self,ax1,ax2,xmax=None,ymax=None,xmin=None,ymin=None,npts=20,fig=None,logscale=True,interpolation='bicubic',mask=None,**kwargs):
        if mask is None:
            mask = self.distok
        w = where(mask)

        data = {'dur':self.alldata.duration[w],'depth':self.alldata.depth[w],'slope':self.alldata.slope[w],
                'logdepth':log10(self.alldata.depth[w]),'logslope':log10(self.alldata.slope[w])}
        minval = {'dur':0.,'depth':0.,'slope':2.,'logdepth':-5.,'logslope':log10(2.)}
        maxval = {'logdepth':0}

        xdata = data[ax1]
        ydata = data[ax2]
        if xmax is None:
            if ax1=='logdepth':
                xmax=0
            else:
                xmax = statu.pctile(xdata,0.95)*2
        if ymax is None:
            if ax2=='logdepth':
                ymax=0
            else:
                ymax = statu.pctile(ydata,0.95)*2
            
        xbins = linspace(minval[ax1],xmax,npts)
        ybins = linspace(minval[ax2],ymax,npts)
        #print(ax1,xbins)
        #print(ax2,ybins)

        plot2dhist(xdata,ydata,xbins=xbins,ybins=ybins,fig=fig,logscale=logscale,interpolation=interpolation,**kwargs)


    def plotdurdepthkde(self,fig=None,durs=None,deps=None,npts=50,maxdur=None,hist=True,logscale=True):
        if hist:
            self.hist2d('dur','logdepth',xmax=maxdur,npts=npts,fig=fig,logscale=logscale)
        else:
            if maxdur is None:
                maxdur = statu.pctile(self.durs,0.99)*1.5
            if durs is None:
                durs = linspace(0,maxdur,npts)
            if deps is None:
                deps = linspace(-5,0,npts)

            plot2dkde(durs,deps,self.durlogdepthkde,fig=fig,contour=False)
            plt.xlim(xmax=maxdur)

    def plotdurslopekde(self,fig=None,durs=None,slopes=None,maxslope=None,npts=50,maxdur=None,logslope=False,hist=True,logscale=True):
        if hist:
            if logslope:
                self.hist2d('dur','logslope',xmax=maxdur,logscale=logscale,fig=fig)
            else:
                self.hist2d('dur','slope',xmax=maxdur,logscale=logscale,fig=fig)
        else:

            if maxslope is None:
                maxslope = statu.pctile(self.slopes,0.99)*2
            if maxdur is None:
                maxdur = statu.pctile(self.durs,0.99)*1.5

            if durs is None:
                durs = linspace(0,maxdur,npts)

            if slopes is None:
                if logslope:
                    slopes = linspace(0,log10(maxslope),npts)
                else:
                    slopes = linspace(1.5,maxslope,npts)


            if logslope:
                plot2dkde(durs,slopes,self.durlogslopekde,fig=fig,contour=False,logscale=logscale)
            else:
                plot2dkde(durs,slopes,self.durslopekde,fig=fig,contour=False,logscale=logscale)            

            plt.xlim(xmax=maxdur)

    def plotslopekde(self,fig=None,slopes=None,npts=200,logslope=False):
        pu.setfig(fig)
        if slopes is None:
            if logslope:
                slopes = linspace(-1,2,npts)
            else:
                slopes = linspace(0,20,npts)
        if logslope:
            plt.plot(slopes,self.logslopekde(slopes))
        else:
            plt.plot(slopes,self.slopekde(slopes))

    def depdurlhood(self,trsig,full=False):
        """p(dep,dur) integrated over trsig posterior (needs MCMC)"""
        if not trsig.hasMCMC:
            raise RuntimeError('Need to run MCMC on transit signal')
        if not full:
            dep,ddep = trsig.logdepthfit
            dur,ddur = trsig.durfit
            point = array([[dur,dep]])
            cov = array([[ddur,0],[0,ddep]])
            return self.durlogdepthkde.integrate_gaussian(point,cov)
        else:
            return self.durlogdepthkde.integrate_kde(trsig.durlogdepthkde)
            
    def slopelhood(self,trsig,full=True,logslope=False):
        """p(slope) integrated over trsig posterior (needs MCMC)"""
        if not trsig.hasMCMC:
            raise RuntimeError('Need to run MCMC on transit signal')
        if not full:
            if logslope:
                slope,dslope = trsig.logslopefit
                return self.logslopekde.integrate_gaussian(slope,dslope)
            else:
                slope,dslope = trsig.slopefit
                return self.slopekde.integrate_gaussian(slope,dslope)
            #logslope,dlogslope = trsig.logslopefit
            #return self.logslopekde.integrate_gaussian(logslope,dlogslope)
        else:
            if logslope:
                return self.logslopekde.integrate_kde(trsig.logslopekde)
            else:
                return self.slopekde.integrate_kde(trsig.slopekde)
            #return self.logslopekde.integrate_kde(trsig.logslopekde)

    def lhood(self,trsig,full=True,depdurfull=False,slopefull=True,logslope=False,
              useslope=True,recalc=False,cachefile=None):
        """returns likelihood of given transit signal
        
        trsig needs to have had MCMC run on it.
        """

        #This key business technically doesn't track the keywords at all...
        if cachefile is None:
            cachefile = self.lhoodcachefile
            if cachefile is None:
                cachefile = LHOODCACHEFILE

        key = hash(self) + hash(trsig)
        lhoodcache = loadcache(cachefile)
        if key in lhoodcache and not recalc:
            return lhoodcache[key]

        #if key in self.lhoods and not recalc:
        #    return self.lhoods[key]
        
        #if trsig.name in self.lhoods and not recalc:
        #    return self.lhoods[trsig.name]
        
        if self.is_ruled_out:
            return 0.

        if full:
            if logslope:
                #lh = self.fullkde_logslope.integrate_kde(trsig.fullkde_logslope)
                lh = self.fullkde_logslope(trsig.fullkde_logslope.dataset).sum()
            else:
                #lh = self.fullkde.integrate_kde(trsig.fullkde)
                lh = self.fullkde(trsig.fullkde.dataset).sum()
        else:
            dur,ddur = trsig.durfit
            dep,ddep = trsig.depthfit
            slope,dslope = trsig.slopefit
            logslope,dlogslope = trsig.logslopefit
            if logslope:
                cov = array([[ddur**2,0,0],[0,ddep**2,0],[0,0,dlogslope**2]])
                means = [dur,dep,logslope]
                #print means
                lh = self.fullkde_logslope.integrate_gaussian(means,cov)
            else:
                cov = array([ddur**2,0,0],[0,ddep**2,0],[0,0,dslope**2])
                means = [dur,dep,slope]
                #print(means)
                lh =  self.fullkde.integrate_gaussian(means,cov)

            useslope = useslope and trsig.slopeOK
            if useslope:
                lh = self.depdurlhood(trsig,full=depdurfull)*self.slopelhood(trsig,full=slopefull,logslope=logslope)
            else:
                lh = self.depdurlhood(trsig,full=depdurfull)

        key = hash(self) + hash(trsig)
        #self.lhoods[key] = lh

        if key not in lhoodcache:
            lhoodcache[key] = lh
            fout = open(cachefile,'a')
            fout.write('%i %g\n' % (key,lh))
            fout.close()
        #writecache(lhoodcache)
        return lh

    def constraintplot(self,*constraints,**kwargs):
        pass

    def lhoodplot(self,trsig=None,fig=None,label='',lhood=None,plotsignal=False,details=True,suptitle='',
                  maxdur=None,maxslope=None,plain=False,Ltot=None,constraints='all',inverse=False,piechart=True,colordict=None,cachefile=None,**kwargs):
        if piechart:
            details = False
        if plain:
            details=False
            suptitle=False
        pu.setfig(fig)

        if trsig is not None:
            dep,ddep = trsig.logdepthfit
            dur,ddur = trsig.durfit
            slope,dslope = trsig.slopefit

            ddep = ddep.reshape((2,1))
            ddur = ddur.reshape((2,1))
            dslope = dslope.reshape((2,1))
            
            if maxdur is None:
                maxdur = dur*2
            if maxslope is None:
                maxslope = slope*2

        if constraints == 'all':
            mask = self.distok
        elif constraints == 'none':
            mask = ones(len(self.alldata)).astype(bool)
        else:
            mask = ones(len(self.alldata)).astype(bool)
            for c in constraints:
                if c not in self.distribution_skip:
                    mask &= self.constraints[c].ok
        if inverse:
            mask = ~mask

        
        if piechart:
            a = plt.axes([0.07,0.5,0.4,0.5])
            #p.subplot(221)
            self.constraint_piechart(fig=0,colordict=colordict)
            

        ax1 = plt.subplot(222)
        if not self.is_ruled_out:
            self.hist2d('dur','logdepth',fig=0,xmax=maxdur,mask=mask,**kwargs)
        if trsig is not None:
            plt.errorbar(dur,dep,xerr=ddur,yerr=ddep,color='w',marker='x',ms=12,mew=3,lw=3,capsize=3,mec='w')  
            plt.errorbar(dur,dep,xerr=ddur,yerr=ddep,color='r',marker='x',ms=10,mew=1.5)
        plt.ylabel(r'log($\delta$)')
        yt = ax1.get_yticks()
        plt.yticks(yt[1:])
        #ax1.yaxis.tick_right()
        xt = ax1.get_xticks()
        plt.xticks(xt[2:-1:2])
        plt.xlim(xmax=maxdur)

        ax3 = plt.subplot(223)
        if not self.is_ruled_out:
            self.hist2d('logdepth','slope',fig=0,ymax=maxslope,mask=mask,**kwargs)
        if trsig is not None:
            plt.errorbar(dep,slope,xerr=ddep,yerr=dslope,color='w',marker='x',ms=12,mew=3,lw=3,capsize=3,mec='w')
            plt.errorbar(dep,slope,xerr=ddep,yerr=dslope,color='r',marker='x',ms=10,mew=1.5)        
        plt.ylabel(r'$T/\tau$')
        plt.xlabel(r'log($\delta$)')
        yt = ax3.get_yticks()
        plt.yticks(yt[:-1])
        plt.ylim(ymin=2)
        plt.ylim(ymax=maxslope)

        ax4 = plt.subplot(224)
        if not self.is_ruled_out:
            self.hist2d('dur','slope',fig=0,xmax=maxdur,ymax=maxslope,mask=mask,**kwargs)
        if trsig is not None:
            plt.errorbar(dur,slope,xerr=ddur,yerr=dslope,color='w',marker='x',ms=12,mew=3,lw=3,capsize=3,mec='w')   
            plt.errorbar(dur,slope,xerr=ddur,yerr=dslope,color='r',marker='x',ms=10,mew=1.5)        
        plt.xlabel(r'$T$ [days]')
        xt = ax4.get_xticks()
        #p.xticks(xt[1:-1])
        plt.xticks(xt[2:-1:2])
        plt.yticks(ax3.get_yticks())
        plt.ylim(ymin=2)
        plt.ylim(ymax=maxslope)
        plt.xlim(xmax=maxdur)

        ticklabels = ax1.get_xticklabels() + ax4.get_yticklabels()
        plt.setp(ticklabels,visible=False)
        
        plt.subplots_adjust(hspace=0.001,wspace=0.001)

        if suptitle and trsig is not None:
            plt.suptitle(suptitle,fontsize=20)

        if trsig is None:
            details = False
        if details:
            plt.annotate(self.model,xy=(0.1,0.85),xycoords='figure fraction',fontsize=24)
            plt.annotate('prior: %.1e (%i%% selected)' % (self.prior,self.selectfrac*100),xy=(0.1,0.8),xycoords='figure fraction',fontsize=14)

            if lhood is None:
                lhood = self.lhood(trsig,logslope=False)
            plt.annotate('likelihood: %.1e' % lhood,xy=(0.1,0.75),xycoords='figure fraction',color='red',fontsize=14)
            if Ltot is not None:
                plt.annotate('Probability\nof scenario: %.3f' % (self.prior*lhood/Ltot),xy=(0.5,0.5),ha='center',va='center',
                           bbox=dict(fc='w'),xycoords='figure fraction',fontsize=16)
            else:
                plt.annotate(r'$\pi \times L$' + '= %.1e' % (self.prior*lhood),xy=(0.1,0.70),xycoords='figure fraction',fontsize=14)
                
        if Ltot is not None:
            if lhood is None:
                lhood = self.lhood(trsig,logslope=False)
            plt.annotate('%s:\nProbability\nof scenario: %.3f' % (trsig.name,self.prior*lhood/Ltot),xy=(0.5,0.5),ha='center',va='center',
                       bbox=dict(boxstyle='round',fc='w'),xycoords='figure fraction',fontsize=15)

        if plotsignal:
            fig = plt.gcf()
            fig.add_axes([0.1,0.4,0.5,0.2])

    def lhoodplot_old(self,trsig,fig=None,insetloc=[0.6,0.20,0.27,0.27],labelxy=(0.65,0.85),title='',titlexy=(0.65,0.97),logslope=False,maxslope=None,maxdur=0.3,useslope=True,name=True):
        useslope = useslope and trsig.slopeOK #don't if unconstrained

        pu.setfig(fig)
        self.plot(fig=0,maxdur=3*trsig.dur)
        if name:
            plt.title(trsig.name)
        if maxslope is None:
            maxslope = trsig.slope * 2
        
        dep,ddep = trsig.logdepthfit
        dur,ddur = trsig.durfit
        
        plt.errorbar(dur,dep,xerr=ddur,yerr=ddep,color='w',marker='x',ms=12,mew=3,lw=3,capsize=3,mec='w')            
        plt.errorbar(dur,dep,xerr=ddur,yerr=ddep,color='r',marker='x',ms=10,mew=1.5)

        plt.xlabel('Duration [days]')
        plt.ylabel('log (depth)')

        lhd = self.lhood(trsig,logslope=logslope,useslope=useslope)
        try:
            name = trsig.name
        except:
            name = 'KOI'
        
        plt.annotate(title,titlexy,fontsize=20,xycoords='axes fraction',va='top')
        plt.annotate('Likelihood: %.1e' % (lhd),xy=(dur,dep),xytext=(labelxy),textcoords='axes fraction',color='r')
                   #arrowprops=dict(arrowstyle='->',relpos=(0,0.5)))

        if useslope:
            fig = plt.gcf()
            fig.add_axes(insetloc)
            if logslope:
                xs = arange(0,2,0.01)
                modelys = self.logslopekde(xs)
                sigys = trsig.logslopekde(xs)
            else:
                xs = arange(0,maxslope,0.1)
                modelys = self.slopekde(xs)
                sigys = trsig.slopekde(xs)
            plt.plot(xs,modelys,'k--')
            plt.plot(xs,sigys,'r')
            plt.ylim(ymax=1.1*modelys.max())
            plt.yticks([])
            if logslope:
                plt.xlabel(r'log($T/\tau$)')
            else:
                plt.xlabel(r'$T/\tau$')

    def analyze_constraints(self,plot=True,fig=None,nbins=100,attrs=['magOK','massOK','radOK','priOK','secOK','ccOK','vccOK'],linestyles=None,band='K',RV=False):
        ntot = float(len(self.priOK))
        n = ntot
        cond = arange(ntot) > -1
        if plot:
            pu.setfig(fig)
        for attr in attrs:
            if not hasattr(self,attr):
                continue
            if getattr(self,attr) is None:
                continue
            nprev = n
            cond &= getattr(self,attr)
            n = float(cond.sum())
            loggin.info('%s condition allows %.3f of remaining.' % (attr,n/nprev))
            if plot:
                if RV:
                    plt.hist(self.RV[where(cond)],bins=linspace(-30,30,nbins),histtype='step',label=attr)
                else:
                    plt.hist(self.dmag[band][where(cond)],bins=linspace(-1,11,nbins),histtype='step',label=attr)                    
        if plot:
            if RV:
                plt.xlabel(r'$\Delta_{RV}$')
            else:
                plt.xlabel(r'$\Delta_{%s}$' % band)
            plt.legend(loc='upper left')

        logging.info('total selected fraction: %.3f' % self.selectfrac)

class EBpopulation(EclipsePopulation,StarPopulation):
    def __init__(self,starfile='ebs.fits',parfile='ebs_params.fits',band='Kepler',prithresh=1e-5,secthresh=5e-5,f_Pshort=0.15,fB=0.375,M=None,dM=0.05,Teff=None,logg=None,vcc=None,noconstraints=False,verbose=False): #remember to implement vcc!
        stars = atpy.Table(starfile,verbose=False) 
        pars = atpy.Table(parfile,verbose=False)

        self.stars = stars

        self.starfile = starfile
        self.parfile = parfile

        self.band = band

        priorfactors = {'f_Pshort':f_Pshort,'fB':fB}

        self.rsky = zeros(len(stars))
        self.RV = stars.RV                        

        self.orbpop = ou.OrbitPopulation(stars.MA,stars.MB,stars.P,eccs=stars.ecc,
                                         mean_anomalies=stars.Manomaly,
                                         obsx=stars.obsx,obsy=stars.obsy,obsz=stars.obsz)

        EclipsePopulation.__init__(self,stars,pars,P=stars.P[0],model='EBs',priorfactors=priorfactors)
        if noconstraints:
            if verbose:
                logging.info('no constraints applied to EB population.')
            return

        if Teff is not None:
            self.constrain_property('Teff',measurement=Teff,selectfrac_skip=True)
        if logg is not None:
            self.constrain_property('logg',measurement=logg,selectfrac_skip=True)

        self.apply_constraint(LowerLimit(pars.depth,prithresh,name='primary depth'))
        self.apply_constraint(UpperLimit(pars.secdepth,secthresh,name='secondary depth'))
        self.constrain_property('MB',lo=0.072)

        if vcc is not None:
            if type(vcc)==type((1,)):
                vcc = VelocityContrastCurve(*vcc)
            mag2 = stars['%s_B' % vcc.band]
            mag1 = stars['%s_A' % vcc.band]
            dmags = mag2 - mag1
            self.apply_constraint(VelocityContrastCurveConstraint(self.RV,dmags,vcc,name='secondary spectrum'))

    def dRV(self,dt):
        return self.orbpop.dRV(dt)

    def apply_trend_constraint(self,limit,dt):
        """Only works if object has dRV method and plong attribute; limit in km/s"""
        dRVs = absolute(self.dRV(dt))
        c1 = UpperLimit(dRVs, limit)
        c2 = LowerLimit(self.stars.P, dt*4)
        #self.apply_constraint(UpperLimit(dRVs, limit,name='RV trend'))
        #self.apply_constraint(LowerLimit(self.stars.Plong, dt*4,name='P(EB) < RV dt*4'))
        self.apply_constraint(JointConstraintOr(c1,c2,name='RV monitoring',Ps=self.stars.P,dRVs=dRVs))

    def dmags(self,band=None):
        if band is None:
            band = self.band
        mag2 = self.stars['%s_B' % band]
        mag1 = self.stars['%s_A' % band]
        return mag2-mag1

    def mstar_hist(self,fig=None,maxm=5,name=True):
        pu.setfig(fig)
        plt.hist(self.stars.MA,lw=3,histtype='step',label=r'$M_A$',normed=True,bins=arange(0,maxm,0.1))
        plt.hist(self.stars.MB,lw=3,histtype='step',label=r'$M_B$',normed=True,bins=arange(0,maxm,0.1))
        plt.legend()
        plt.xlabel('mass')
        plt.yticks([])
        plt.ylabel('Probability density')
        if name:
            if type(name)!=type(''):
                name = self.name
            plt.annotate(name,xy=(0.7,0.2),xycoords='axes fraction',fontsize=20)

class BGEBpopulation(EclipsePopulation,StarPopulation):
    def __init__(self,starfile='bebs.fits',parfile='bebs_params.fits',blendmag=14.71,band='Kepler',prithresh=1e-5,secthresh=5e-5,maxrad=2.,cc=None,fB=0.375,f_Pshort=0.15,dmaglim=1,noconstraints=False,blendmags=None,
                 modelname='BEBs',verbose=False):
        stars = atpy.Table(starfile,verbose=False) #e.g. *bebs.fits file
        pars = atpy.Table(parfile,verbose=False) #e.g. *bebs_params.fits file
        self.starfile = os.path.abspath(starfile)
        self.parfile = os.path.abspath(parfile)
        self.band = band
        self.blendmags = blendmags
        if blendmags is None:
            blendmags = {band:blendmag}
        self.blendmags = blendmags
        self.blendmag = blendmag

        F1 = 10**(-0.4*blendmag)
        mag = stars['%s_mag_tot' % band]
        F2 = 10**(-0.4*mag)
        dilution_factor = F2/(F1+F2)
        pars.depth *= dilution_factor
        pars.secdepth *= dilution_factor
        self.dilution_factor = dilution_factor


        priorfactors = {'fB':fB,'f_Pshort':f_Pshort,'area':pi*maxrad**2}
        EclipsePopulation.__init__(self,stars,pars,P=stars.P[0],model=modelname,priorfactors=priorfactors)

        self.set_maxrad(maxrad)  #sets maxrad, rsky, centroid_shift
        self.RV = self.stars.RV

        self.dmaglim = dmaglim

        #self.rsky = randpos_in_circle(len(stars),rad=maxrad,return_rad=True)
        #self.RV = rand_dRV(len(stars))

        #self.set_dmaglim(dmaglim)

        if noconstraints:
            if verbose:
                logging.info('no constraints applied to BGEB population.')
            return

        #self.set_dmaglim(dmaglim)
        self.apply_constraint(LowerLimit(addmags(stars['%s_1' % band],stars['%s_2' % band]),blendmag+dmaglim,name='bright blend limit'))


        if cc is not None:
            if size(cc)==1 and type(cc) != type([]):
                cc = [cc]
            for c in cc:
                bgmag = addmags(stars['%s_1' % c.band],stars['%s_2' % c.band])
                dmags = bgmag - c.mag
                self.apply_constraint(ContrastCurveConstraint(self.rsky,dmags,c,name='%s band' % c.band))

        self.apply_constraint(LowerLimit(pars.depth,prithresh,name='primary depth'))
        self.apply_constraint(UpperLimit(pars.secdepth,secthresh,name='secondary depth'))

    def fluxfrac_eclipsing(self,band=None):
        if band is None:
            band = self.band
        if band=='K':
            band = 'Ks'
        if band=='kep' or band=='Kep':
            band = 'Kepler'

        blendmag = self.blendmags[band]
        F1 = 10**(-0.4*blendmag)
        mag = addmags(self.stars['%s_1' % band],self.stars['%s_2' % band])
        F2 = 10**(-0.4*mag)
        return F2/(F1+F2)

    def set_maxrad(self,maxrad):
        self.maxrad = maxrad
        self.rsky = self.stars.rad*maxrad #self.stars.rad random from 0 to 1
        self.centroid_shift = self.rsky * self.alldata.depth

        for name in self.constraints:
            c = self.constraints[name]
            if type(c) is ContrastCurveConstraint:
                c.update_rs(self.rsky)
        #self.priorfactors['area'] = pi*maxrad**2
        self.apply_constraint(UpperLimit(self.rsky,maxrad,name='Rsky'),overwrite=True)
        self._apply_all_constraints()   #not necessary?
        self.priorfactors['area'] = pi*maxrad**2
        self._calculate_prior()
        

    def set_maxrad_old(self,maxrad):
        self.maxrad = maxrad
        self.rsky = randpos_in_circle(len(stars),rad=maxrad,return_rad=True)
        self.priorfactors['area'] = pi*maxrad**2
        #need to reset any cc constraints

    def dmags(self,band=None,primag=None):
        if band is None:
            band = self.band
        bgmag = addmags(self.stars['%s_1' % band],self.stars['%s_2' % band])
        if primag is None:
            primag = self.blendmag
        return bgmag - primag

class Specific_BGEBpopulation(BGEBpopulation):
    def __init__(self,starfile='bebs_specific1.fits',parfile='bebs_specific1_params.fits',**kwargs):
        BGEBpopulation.__init__(self,starfile,parfile,modelname='Specific BEB',**kwargs)
        
        self.is_specific = True
        self.remove_constraint('Rsky')
        self.remove_constraint('bright blend limit')

        m = re.search('(\d)\.fits',starfile)
        if m:
            self.index = int(m.group(1))

        del self.priorfactors['area']
        #del self.priorfactors['fB'] #don't know why this was here
        self._calculate_prior()

        #sky-projected separation needs to be implemented (at least vs. Specific_HEBpopulation)
        # ...or does it?? (should make prior relative b/w SHEB and SBEB?

class HEBpopulation(EclipsePopulation,StarPopulation):
    def __init__(self,starfile='hebs.fits',parfile='hebs_params.fits',band='Kepler',prithresh=1e-5,secthresh=5e-5,cc=None,d=None,maxrad=2,Kmag=None,vcc=None,M=None,dM=0.05,Teff=None,logg=None,ftrip=0.12,noconstraints=False,verbose=False,modelname='HEBs'):
        stars = atpy.Table(starfile,verbose=False) #e.g. *hebs.fits file
        pars = atpy.Table(parfile,verbose=False) #e.g. *hebs_params.fits file
        self.starfile = os.path.abspath(starfile)
        self.parfile = os.path.abspath(parfile)
        self.band = band
        self.stars = stars

        A = (stars.which_eclipse=='A')
        B = (stars.which_eclipse=='B')

        dilution_factor = stars['%s_fluxfrac_C' % band] + (A*stars['%s_fluxfrac_A' % band] + B*stars['%s_fluxfrac_B' % band])
        pars.depth *= dilution_factor
        pars.secdepth *= dilution_factor

        self.dilution_factor = dilution_factor

        if d is None:
            if Kmag is None:
                Kmag=12.
            MK = stars.Ks_tot
            mK = Kmag
            d = 10**(1+(mK-MK)/5.)

        self.d = d


        priorfactors = {'ftrip':ftrip}
        EclipsePopulation.__init__(self,stars,pars,P=stars.P[0],model=modelname,priorfactors=priorfactors)

        self.rsky = self.stars.rad/self.d  #self.stars.rad is in projected AU
        self.centroid_shift = self.rsky * self.alldata.depth
        self.set_maxrad(maxrad)
        #self.maxrad = maxrad
        self.RV = self.stars.RV  #this is RV difference between star A and B

        M1s = (self.stars.MA*B + self.stars.MB*A)
        M2s = (self.stars.MB*B + self.stars.MA*A)
        M3s = self.stars.MC

        self.orbpop = ou.TripleOrbitPopulation(M1s,M2s,M3s,self.stars.Plong,self.stars.P,
                                               ecclong=self.stars.ecclong,eccshort=self.stars.ecc,
                                               mean_anomalies_long=self.stars.Manomaly_long,
                                               mean_anomalies_short=self.stars.Manomaly_short,
                                               obsx_long=self.stars.obsx_long,
                                               obsy_long=self.stars.obsy_long,
                                               obsz_long=self.stars.obsz_long,
                                               obsx_short=self.stars.obsx_short,
                                               obsy_short=self.stars.obsy_short,
                                               obsz_short=self.stars.obsz_short)

            #self.orbpop = ou.OrbitPopulation(self.stars.orb_M1,self.stars.orb_M2,self.stars.Plong,
            #eccs=self.stars.ecclong,mean_anomalies=self.stars.Manomaly,
            #                             obsx=self.stars.obsx,obsy=self.stars.obsy,obsz=self.stars.obsz)

        if noconstraints:
            if verbose:
                logging.info('no constraints applied to HEB population.')
            return

        ##################################
        # everything below not done when noconstraints is True, which is usually, in practice

        if Teff is not None:
            self.constrain_property('Teff',measurement=Teff,selectfrac_skip=True)
        if logg is not None:
            self.constrain_property('logg',measurement=logg,selectfrac_skip=True)

        if cc is not None:
            if size(cc)==1 and type(cc) != type([]):
                cc = [cc]
            for c in cc:
                mag2 = addmags(stars['%s_B' % c.band],stars['%s_C' % c.band]*B)
                mag1 = addmags(stars['%s_A' % c.band] + (stars['%s_B' % c.band])*A)
                dmags = mag2 - mag1
                self.apply_constraint(ContrastCurveConstraint(self.rsky,dmags,c,name='%s band' % c.band))

        if vcc is not None:
            if type(vcc)==type((1,)):
                vcc = VelocityContrastCurve(*vcc)
            mag2 = addmags(stars['%s_B' % vcc.band],stars['%s_C' % vcc.band]*B) 
            mag1 = stars['%s_A' % vcc.band] + (stars['%s_B' % vcc.band])*A
            dmags = mag2 - mag1
            self.apply_constraint(VelocityContrastCurveConstraint(self.RV,dmags,vcc,name='secondary spectrum'))

        self.constrain_property('MC',lo=0.072)
        self.apply_constraint(LowerLimit(pars.depth,prithresh,name='primary depth'))
        self.apply_constraint(UpperLimit(pars.secdepth,secthresh,name='secondary depth'))

        self.maxrad = maxrad
        self.apply_constraint(UpperLimit(self.rsky,maxrad,name='Rsky'))

        #self.ftrip = ftrip
        #self.prior = stars.keywords['PROB']*self.selectfrac*ftrip
        #wok = where(self.distok)[0]
        #EclipsePopulation.__init__(self,pars.rows(wok),stars.keywords['P'],'HEBs')

    def dRV(self,dt):
        """dt in days
        """
        return self.orbpop.dRV_1(dt)

    def fluxfrac_eclipsing(self,band):
        if band is None:
            band = self.band
        if band=='K':
            band = 'Ks'
        if band=='kep' or band=='Kep':
            band = 'Kepler'

        return self.stars['%s_dilution_factor' % band]        

    def set_maxrad(self,maxrad):
        self.maxrad = maxrad
        self.apply_constraint(UpperLimit(self.rsky,maxrad,name='Rsky'),overwrite=True)
        self._apply_all_constraints()

    def dmags(self,band=None):
        if band is None:
            band = self.band
        CwA = (self.stars.which_eclipse=='A')
        CwB = (self.stars.which_eclipse=='B')
        magA = self.stars['%s_A' % band]
        magB = self.stars['%s_B' % band]
        magC = self.stars['%s_C' % band]

        #mag2 = addmags(self.stars['%s_B' % band],self.stars['%s_C' % band]*B)
        #mag1 = addmags(self.stars['%s_A' % band] + (self.stars['%s_B' % band])*A)
        mag1 = magA*CwB + addmags(magA,magC)*CwA
        mag2 = addmags(magB,magC)*CwB + magB*CwA

        return mag2-mag1

    def mstar_hist(self,fig=None,maxm=5,name='',namexy=(0.65,0.2)):
        pu.setfig(fig)
        plt.hist(self.stars.MA,lw=3,histtype='step',label=r'$M_A$',normed=True,bins=arange(0,maxm,0.1))
        plt.hist(self.stars.MB,lw=3,histtype='step',label=r'$M_B$',normed=True,bins=arange(0,maxm,0.1))
        plt.hist(self.stars.MC,lw=3,histtype='step',label=r'$M_C$',normed=True,bins=arange(0,maxm,0.1))
        plt.legend()
        plt.xlabel('mass')
        plt.yticks([])
        plt.ylabel('Normalized fraction')
        plt.annotate(name,xy=namexy,xycoords='axes fraction',fontsize=20,ha='center')

class Specific_HEBpopulation(HEBpopulation):
    def __init__(self,starfile='hebs_specific1.fits',parfile='hebs_specific1_params.fits',**kwargs):
        HEBpopulation.__init__(self,starfile,parfile,modelname='Specific HEB',**kwargs)

        self.is_specific = True
        self.remove_constraint('Rsky')
        self.remove_constraint('bright blend limit')

        m = re.search('(\d)\.fits',starfile)
        if m:
            self.index = int(m.group(1))

        del self.priorfactors['ftrip']
        self.add_priorfactor(ftripbin=0.5)

class Transitpopulation(EclipsePopulation,StarPopulation):
    def __init__(self,starfile='pls.fits',parfile='pls_params.fits',band='Kepler',
                 prithresh=1e-5,fp=0.01,noconstraints=False,
                 verbose=False,multboost=1):
        stars = atpy.Table(starfile,verbose=False) #e.g. *planets.fits file
        pars = atpy.Table(parfile,verbose=False) #e.g. *planets_params.fits file
        self.starfile = os.path.abspath(starfile)
        self.parfile = os.path.abspath(parfile)
        self.band = band

        priorfactors = {'fp_specific':fp,'multboost':multboost}
        EclipsePopulation.__init__(self,stars,pars,P=stars.keywords['P'],model='Planets',priorfactors=priorfactors)

        if noconstraints:
            if verbose:
                logging.info('no constraints applied to Transit population.')
            return
        
        self.apply_constraint(LowerLimit(pars.depth,prithresh,name='primary depth'))


class BGTransitpopulation(EclipsePopulation,StarPopulation):
    def __init__(self,starfile='bgpls.fits',parfile='bgpls_params.fits',band='Kepler',blendmag=14.71,prithresh=1e-5,maxrad=2.,cc=None,fp=0.4,dmaglim=1,noconstraints=False,verbose=False,blendmags=None):
        stars = atpy.Table(starfile,verbose=False) #e.g. *planets.fits file
        pars = atpy.Table(parfile,verbose=False) #e.g. *planets_params.fits file
        self.starfile = os.path.abspath(starfile)
        self.parfile = os.path.abspath(parfile)
        self.blendmag = blendmag
        self.blendmags = blendmags
        self.band = band

        F1 = 10**(-0.4*blendmag)
        mag = stars['%s_mag_tot' % band]
        F2 = 10**(-0.4*mag)
        dilution_factor = F2/(F1+F2)
        pars.depth *= dilution_factor
        pars.secdepth *= dilution_factor
        self.dilution_factor = dilution_factor

        #self.rsky = randpos_in_circle(len(stars),rad=maxrad,return_rad=True)
        #self.RV = rand_dRV(len(stars))

        priorfactors = {'fp':fp,'area':pi*maxrad**2}
        EclipsePopulation.__init__(self,stars,pars,P=stars.keywords['P'],model='Blended Planets',priorfactors=priorfactors)

        self.set_maxrad(maxrad)
        #self.maxrad = maxrad
        #self.rsky = self.stars.rad*maxrad #self.stars.rad is random radius from 0 to 1
        self.RV = self.stars.RV

        self.dmaglim = dmaglim
        #self.set_dmaglim(dmaglim)

        if noconstraints:
            if verbose:
                logging.info('no constraints applied to BG Transit population.')
            return    

        self.apply_constraint(LowerLimit(stars[band],blendmag+dmaglim,name='bright'))
        if cc is not None:
            if size(cc)==1 and type(cc) != type([]):
                cc = [cc]
            for c in cc:
                bgmag = stars[c.band]
                dmags = bgmag - c.mag
                self.apply_constraint(ContrastCurveConstraint(self.rsky,dmags,c,name='%s band' % c.band))

        self.apply_constraint(LowerLimit(pars.depth,prithresh,name='primary depth'))

    def fluxfrac_eclipsing(self,band=None):
        if band is None:
            band = self.band
        if band=='K':
            band = 'Ks'
        if band=='kep' or band=='Kep':
            band = 'Kepler'

        blendmag = self.blendmags[band]
        F1 = 10**(-0.4*blendmag)
        mag = self.stars['%s' % band]
        F2 = 10**(-0.4*mag)
        return F2/(F1+F2)

    def set_maxrad(self,maxrad):
        self.maxrad = maxrad
        self.rsky = self.stars.rad*maxrad #self.stars.rad random from 0 to 1
        self.centroid_shift = self.rsky * self.alldata.depth

        for name in self.constraints:
            c = self.constraints[name]
            if type(c) is ContrastCurveConstraint:
                c.update_rs(self.rsky)
        self.apply_constraint(UpperLimit(self.rsky,maxrad,name='Rsky'),overwrite=True)
        self._apply_all_constraints()   #not necessary?
        self.priorfactors['area'] = pi*maxrad**2
        self._calculate_prior()
        

    def set_maxrad_old(self,maxrad):
        self.maxrad = maxrad
        self.rsky = randpos_in_circle(len(stars),rad=maxrad,return_rad=True)
        self.priorfactors['area'] = pi*maxrad**2
        #need to reset any cc constraints

    def dmags(self,band=None,primag=None):
        if band is None:
            band = self.band
        bgmag = self.stars[band]
        if primag is None:
            primag = self.blendmag
        return bgmag - primag

class ObservedBlendedEclipse(EclipsePopulation,StarPopulation):
    pass

#############################################


class Transitsignal(object):
    """a phased transit signal with the epoch of the transit at 0, and 'continuum' set at 1
    """
    def __init__(self,ts,fs,dfs=None,P=10,p0=None,name='',maxslope=None):
        inds = ts.argsort()
        self.ts = ts[inds]
        self.fs = fs[inds]
        self.name = name
        self.P = P
        if maxslope is None:
            maxslope = 30
        self.maxslope = maxslope
        if type(P) == type(array([1])):
            self.P = P[0]
        if p0 is None:
            depth = 1 - fs.min()
            duration = (fs < (1-0.01*depth)).sum()/float(len(fs)) * (ts[-1] - ts[0])
            tc0 = ts[fs.argmin()]
            p0 = array([duration,depth,5.,tc0])
        #pfit = tr.fitprotopapas(ts,fs,p0)
        goahead = True
        try:
            tru
        except:
            goahead = False
        if not goahead:
            logging.warning('transit_utils not defined, no leastsq trapezoid fit available.')
        else:
            tfit = tr.fit_traptransit(ts,fs,p0)

            if dfs is None:
                dfs = (self.fs - tru.traptransit(self.ts,tfit)).std()
            if size(dfs)==1:
                dfs = ones(len(self.ts))*dfs
            self.dfs = dfs


        #self.ts -= tfit[3]  #center at fitted epoch
            self.dur,self.depth,self.slope,self.center = tfit
        #self.protopapas = pfit
            self.traptrans = tfit

        logging.debug('trapezoidal leastsq fit: {}'.format(self.traptrans))

        self.hasMCMC=False

    def __eq__(self,other):
        return hash(self) == hash(other)

    def __hash__(self):
        key = 0
        key += hash(self.ts.__str__())
        key += hash(self.ts.sum())
        key += hash(self.fs.__str__())
        key += hash(self.fs.sum())
        key += hash(self.P)
        key += hash(self.maxslope)
        if self.hasMCMC:
            key += hash(self.slopes.sum())
            key += hash(str(self.slopes[:100]))
        return key
        
    def noisylc(self,noise=5e-5):
        return self.ts,self.fs*(1 + rand.normal(size=len(self.ts))*noise)

    def plot(self,fig=None,plotpp=False,plot_tt=False,addnoise=0,name=False,tt_color='g',**kwargs):
        pu.setfig(fig)
        ts,fs = self.noisylc(addnoise)
        plt.plot(ts,fs,'.',**kwargs)
        if plotpp:
            plt.plot(ts,tr.protopapas(ts,self.protopapas),**kwargs)
        if plot_tt and hasattr(self,'traptrans'):
            plt.plot(ts,tr.traptransit(ts,self.traptrans),color=tt_color,**kwargs)
        if name is not None:
            if type(name)==type(''):
                text = name
            else:
                text = self.name
            plt.annotate(text,xy=(0.1,0.1),xycoords='axes fraction',fontsize=22)
        if hasattr(self,'depthfit') and not np.isnan(self.depthfit[0]):
            lo = 1 - 3*self.depthfit[0]
            hi = 1 + 2*self.depthfit[0]
        else:
            lo = 1
            hi = 1
        sig = statu.qstd(self.fs,0.005)
        hi = max(hi,self.fs.mean() + 7*sig)
        lo = min(lo,self.fs.mean() - 7*sig)
        logging.debug('lo={}, hi={}'.format(lo,hi))
        plt.ylim((lo,hi))
        plt.xlabel('time [days]')
        plt.ylabel('Relative flux')

    def MCMC(self,niter=500,nburn=200,nwalkers=200,threads=1,fit_partial=False,width=3,savedir=None,refit=False,thin=10,conf=0.95,maxslope=30,debug=False,p0=None):
        if fit_partial:
            wok = where(absolute(self.ts-self.center) < (width*self.dur))
        else:
            wok = where(~isnan(self.fs))

        alreadydone = True
        alreadydone &= savedir is not None
        alreadydone &= os.path.exists('%s/ts.npy' % savedir)
        alreadydone &= os.path.exists('%s/fs.npy' % savedir)

        if savedir is not None and alreadydone:
            ts_done = load('%s/ts.npy' % savedir)
            fs_done = load('%s/fs.npy' % savedir)
            alreadydone &= all(ts_done == self.ts[wok])
            alreadydone &= all(fs_done == self.fs[wok])
        
        if alreadydone and not refit:
            logging.info('MCMC fit already done for %s.  Loading chains.' % self.name)
            Ts = load('%s/duration_chain.npy' % savedir)
            ds = load('%s/depth_chain.npy' % savedir)
            slopes = load('%s/slope_chain.npy' % savedir)
            tcs = load('%s/tc_chain.npy' % savedir)
        else:
            logging.info('Fitting data to trapezoid shape with MCMC for %s....' % self.name)
            if p0 is None:
                p0 = self.traptrans.copy()
                p0[0] = np.absolute(p0[0])
                if p0[2] < 2:
                    p0[2] = 2.01
                if p0[1] < 0:
                    p0[1] = 1e-5
            logging.debug('p0 for MCMC = {}'.format(p0))
            sampler = tr.traptransit_MCMC(self.ts[wok],self.fs[wok],self.dfs[wok],niter=niter,nburn=nburn,nwalkers=nwalkers,threads=threads,p0=p0,return_sampler=True,maxslope=maxslope)
            Ts,ds,slopes,tcs = (sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3])
            self.sampler = sampler
            if savedir is not None:
                save('%s/duration_chain.npy' % savedir,Ts)
                save('%s/depth_chain.npy' % savedir,ds)
                save('%s/slope_chain.npy' % savedir,slopes)
                save('%s/tc_chain.npy' % savedir,tcs)
                save('%s/ts.npy' % savedir,self.ts[wok])
                save('%s/fs.npy' % savedir,self.fs[wok])

        if debug:
            print(Ts)
            print(ds)
            print(slopes)
            print(tcs)

        N = len(Ts)
        self.Ts_acor = acor.acor(Ts)[1]
        self.ds_acor = acor.acor(ds)[1]
        self.slopes_acor = acor.acor(slopes)[1]
        self.tcs_acor = acor.acor(tcs)[1]
        self.fit_converged = True
        for t in [self.Ts_acor,self.ds_acor,
                  self.slopes_acor,self.tcs_acor]:
            if t > 0.1*N:
                self.fit_converged = False


        ok = (Ts > 0) & (ds > 0) & (slopes > 0) & (slopes < self.maxslope)
        logging.debug('trapezoidal fit has {} good sample points'.format(ok.sum()))
        if ok.sum()==0:
            if (Ts > 0).sum()==0:
                #logging.debug('{} points with Ts > 0'.format((Ts > 0).sum()))
                logging.debug('{}'.format(Ts))
                raise MCMCError('{}: 0 points with Ts > 0'.format(self.name))
            if (ds > 0).sum()==0:
                #logging.debug('{} points with ds > 0'.format((ds > 0).sum()))
                logging.debug('{}'.format(ds))
                raise MCMCError('{}: 0 points with ds > 0'.format(self.name))
            if (slopes > 0).sum()==0:
                #logging.debug('{} points with slopes > 0'.format((slopes > 0).sum()))
                logging.debug('{}'.format(slopes))
                raise MCMCError('{}: 0 points with slopes > 0'.format(self.name))
            if (slopes < self.maxslope).sum()==0:
                #logging.debug('{} points with slopes < maxslope ({})'.format((slopes < self.maxslope).sum(),self.maxslope))
                logging.debug('{}'.format(slopes))
                raise MCMCError('{} points with slopes < maxslope ({})'.format((slopes < self.maxslope).sum(),self.maxslope))


        durs,deps,logdeps,slopes,logslopes = (Ts[ok],ds[ok],log10(ds[ok]),slopes[ok],log10(slopes[ok]))
        
        
        #self.durfit = (durs.mean(),durs.std())
        #self.depthfit = (deps.mean(),deps.std())
        #self.logdepthfit = (logdeps.mean(),logdeps.std())
        #self.logslopefit = (logslopes.mean(),logslopes.std())
        #self.slopefit= (slopes.mean(),slopes.std())


        #print (logslopes > 4).sum(),len(logslopes)

        inds = (arange(len(durs)/thin)*thin).astype(int)
        durs,deps,logdeps,slopes,logslopes = (durs[inds],deps[inds],logdeps[inds],slopes[inds],logslopes[inds])
        self.durs,self.logdeps,self.slopes = (durs,logdeps,slopes)

        points = array([durs,deps])
        self.durdepthkde = stats.gaussian_kde(points)
        points = array([durs,logdeps])
        self.durlogdepthkde = stats.gaussian_kde(points)
        self.durkde = stats.gaussian_kde(durs)
        self.depthkde = stats.gaussian_kde(deps)
        self.slopekde = stats.gaussian_kde(slopes)
        self.logslopekde = stats.gaussian_kde(logslopes)
        self.logdepthkde = stats.gaussian_kde(logdeps)


        if self.fit_converged:
            try:
                durconf = statu.kdeconf(self.durkde,conf)
                depconf = statu.kdeconf(self.depthkde,conf)
                logdepconf = statu.kdeconf(self.logdepthkde,conf)
                logslopeconf = statu.kdeconf(self.logslopekde,conf)
                slopeconf = statu.kdeconf(self.slopekde,conf)
            except:
                raise
                raise MCMCError('Error generating confidence intervals...fit must not have worked.')

            durmed = median(durs)
            depmed = median(deps)
            logdepmed = median(logdeps)
            logslopemed = median(logslopes)
            slopemed = median(slopes)

            self.durfit = (durmed,array([durmed-durconf[0],durconf[1]-durmed]))
            self.depthfit = (depmed,array([depmed-depconf[0],depconf[1]-depmed]))
            self.logdepthfit = (logdepmed,array([logdepmed-logdepconf[0],logdepconf[1]-logdepmed]))
            self.logslopefit = (logslopemed,array([logslopemed-logslopeconf[0],logslopeconf[1]-logslopemed]))
            self.slopefit = (slopemed,array([slopemed-slopeconf[0],slopeconf[1]-slopemed]))

        else:
            self.durfit = (nan,nan,nan)
            self.depthfit = (nan,nan,nan)
            self.logdepthfit = (nan,nan,nan)
            self.logslopefit = (nan,nan,nan)
            self.slopefit = (nan,nan,nan)



        points = array([durs,logdeps,slopes])
        self.fullkde = stats.gaussian_kde(points)
        points = array([durs,logdeps,logslopes])
        self.fullkde_logslope = stats.gaussian_kde(points)

        self.slopelo,self.slopehi = statu.kdeconf(self.logslopekde,-1,5)
        self.slopeOK = self.slopehi < 4
            

        self.hasMCMC = True

    def MCMC_old(self,niter=5e4,nburn=5e3,thin=25,verbose=True,plot=False):
        M = tr.protopapasMCMC(self.ts,self.fs,self.dfs,self.P,
                                   niter=niter,nburn=nburn,thin=thin,verbose=verbose)
        #make a text database to save the chains?

        if plot:
            pm.Matplot.plot(M)
        self.M = M

        durs = squeeze(M.trace('duration')[:])
        deps = squeeze(M.trace('depth')[:])
        logdeps = log10(deps)
        logslopes = squeeze(M.trace('logslope')[:])
        slopes = 10**logslopes

        self.durfit = (durs.mean(),durs.std())
        self.depthfit = (deps.mean(),deps.std())
        self.logdepthfit = (logdeps.mean(),logdeps.std())
        self.logslopefit = (logslopes.mean(),logslopes.std())
        self.slopefit= (slopes.mean(),slopes.std())

        points = array([durs,deps])
        self.durdepthkde = stats.gaussian_kde(points)
        points = array([durs,logdeps])
        self.durlogdepthkde = stats.gaussian_kde(points)
        self.durkde = stats.gaussian_kde(durs)
        self.depthkde = stats.gaussian_kde(deps)
        self.slopekde = stats.gaussian_kde(slopes)
        self.logslopekde = stats.gaussian_kde(logslopes)

        self.hasMCMC = True

    def plotslopekde(self,fig=None,logslope=False,maxslope=30):
        pu.setfig(fig)
        if logslope:
            slopes = arange(-1,log10(maxslope),0.01)
            plt.plot(slopes,self.logslopekde(slopes))
        else:
            slopes = arange(0,maxslope,0.1)
            plt.plot(slopes,self.slopekde(slopes))
                   

    def plotdurdepth(self,fig=None,npts=100,durmin=None,durmax=None,depmin=None,depmax=None,logdepth=False,**kwargs):
        if durmin is None:
            durmin = 0.8*self.dur
        if durmax is None:
            durmax = 1.2*self.dur
        if depmin is None:
            depmin = 0.8*self.depth
        if depmax is None:
            depmax = 1.2*self.depth
        if logdepth:
            depmin = log10(depmin)
            depmax = log10(depmax)

            
        durs = linspace(durmin,durmax,npts)
        deps = linspace(depmin,depmax,npts)
        
        if logdepth:
            plot2dkde(durs,deps,self.durlogdepthkde,fig=fig)
        else:
            plot2dkde(durs,deps,self.durdepthkde,fig=fig)


    def noisy(self,noise=1e-4):
        ts,fs = self.noisylc(noise)
        new = transitsignal(ts,fs,noise,self.P)
        newdict = dict(self.__dict__.items() + new.__dict__.items())
        new.__dict__ = newdict
        return new 


class ModelTransitsignal(Transitsignal):
    def __init__(self,star,planet,P,inc=90,ecc=0,w=90,conv=False,mu1=None,mu2=None,dt=5,npts=100,texp=29.4):
        self.star = star
        self.planet = planet
        self.P = P
        self.inc = inc
        self.ecc = ecc
        self.w = w
        ts, fs = tr.planetlc(self.P,self.planet.m,self.planet.r,self.star.m,self.star.r,inc,ecc,w,
                                            self.star.absmag,conv=conv,mu1=mu1,mu2=mu2,npts=npts,dt=dt)

        Transitsignal.__init__(self,ts,fs,P=P)
        
    def diluted(self,mag):
        """ dilutes light curve; 'thirdlight' is the flux of third light in units of primary
        """
        thirdlight = 10**(0.4*(self.star.mag - mag))
        newfs = (thirdlight + self.fs)/(thirdlight + 1)
        self.fs = newfs
        fs = self.fs
        ts = self.ts

        depth = 1 - fs.min()
        duration = (fs < (1-0.01*depth)).sum()/float(len(fs)) * (ts[-1] - ts[0])
        p0 = (duration,depth,10)
        pfit = tr.fitprotopapas(ts,fs,p0)
        self.protopapas = pfit
        self.dur,self.depth,self.slope = pfit
    
        return self

class Modelfrompop(Transitsignal):
    def __init__(self,pop,i,sec=False,conv=True,diluted=True,MAfn=None):
        #no limb darkening anywhere here...

        data = pop.stars
        P = data.P[i]
        
        p0,b,aR = tr.eclipse_pars(P,data.M1[i],data.M2[i],data.R1[i],data.R2[i],
                                  inc=data.inc[i],ecc=data.ecc[i],w=data.w[i])
        self.P = P
        self.M1 = data.M1[i]
        self.M2 = data.M2[i]
        self.R1 = data.R1[i]
        self.R2 = data.R2[i]
        self.p0 = p0
        self.b = b
        self.aR = aR
        self.ecc=data.ecc[i]
        self.w = data.w[i]

        frac = data.fluxfrac1[i]
        self.frac = frac
        
        if sec:
            b = data.b_sec[i]
            frac = data.fluxfrac2[i]

        try:
            if diluted:
                frac *= pop.dilution_factor[i]
        except:
            pass

        #print p0,b,aR,P,data.ecc[i],data.w[i],sec,frac,conv
        ts,fs = tr.eclipse(p0,b,aR,P=P,ecc=data.ecc[i],w=data.w[i],sec=sec,frac=frac,conv=conv,MAfn=MAfn)

        Transitsignal.__init__(self,ts,fs)

class ASCIItransitsignal(Transitsignal):
    def __init__(self,filename,cols=(0,1)):
        m = re.search('(\S+)\.ascii',filename)
        if m:
            self.name = m.group(1)
        else:
            self.name = filename
        self.filename = filename
        ts,fs = loadtxt('%s/%s' % (KOIDATAFOLDER,filename),usecols=cols,unpack=True)
        #ts *= 24*60 #days to minutes
        Transitsignal.__init__(self,ts,fs)
        self.MCMC()


def plot2dhist(xdata,ydata,cmap='binary',interpolation='nearest',fig=None,logscale=True,xbins=None,ybins=None,nbins=50,pts_only=False,**kwargs):
    pu.setfig(fig)
    if pts_only:
        plt.plot(xdata,ydata,**kwargs)
        return
    if xbins is not None and ybins is not None:
        H,xs,ys = histogram2d(xdata,ydata,bins=(xbins,ybins))
    else:
        H,xs,ys = histogram2d(xdata,ydata,bins=(nbins,nbins))        
    H = H.T
    #H = flipud(H)
    if logscale:
        H = log(H)
    extent = [xs[0],xs[-1],ys[0],ys[-1]]
    plt.imshow(H,extent=extent,interpolation=interpolation,aspect='auto',cmap=cmap,origin='lower')
    


def plot2dkde(xs,ys,kde,cmap='binary',fig=None,vmax=None,contour=True,confs=[0.683,0.95],logscale=False):
    pu.setfig(fig)
    X,Y = meshgrid(xs,ys)

    pts = array([X.ravel(),Y.ravel()])
    Z = kde(pts)
    Z = Z.reshape(X.shape)
    if logscale:
        Z = log10(Z)

    #grid_coords = append(X.reshape(-1,1),Y.reshape(-1,1),axis=1)
    #print grid_coords
    #Z = kde(grid_coords.T)
    #Z = Z.reshape(X.shape)
    if contour:
        levels = []
        for conf in confs:
            levels.append(conf2d(xs,ys,Z,conf))
        plt.contour(X,Y,Z,levels=levels)
    else:
        plt.imshow(Z,origin='lower',extent=(xs[0],xs[-1],ys[0],ys[-1]),aspect='auto',cmap=cmap,vmin=0,vmax=vmax)
    
#def getKOIdata(filename):
#    filename = '%s/%s' % (DATAROOTDIR,filename)
#    return atpy.Table(filename)

def plotlc(data,i,sec=False,fig=None,**kwargs):
    pu.setfig(fig)
    P = data.P[i]
    p0,b,aR = tr.eclipse_pars(P,data.M1[i],data.M2[i],data.R1[i],data.R2[i],
                             inc=data.inc[i],ecc=data.ecc[i],w=data.w[i])
    if sec:
        b = data.b_sec[i]
        frac = data.fluxfrac2[i]
    else:
        frac = data.fluxfrac1[i]
    ts,fs = tr.eclipse(p0,b,aR,P=P,ecc=data.ecc[i],w=data.w[i],sec=sec,frac=frac)
    plt.plot(ts,fs)


def findmatch(pop,trsig,n=10,return_inds=True,plot=False,fig=None,logslope=False,logdepth=True):
    durnorm = pop.data.duration.std()
    if logdepth:
        depnorm = log10(pop.data.depth).std()
    else:
        depnorm = (pop.data.depth).std()        
    if logslope:
        slopenorm = log10(pop.data.slope).std()
    else:
        slopenorm = pop.data.slope.std()
        

    durdists = trsig.dur/durnorm - pop.data.duration/durnorm
    if logdepth:
        depdists = log10(trsig.depth)/depnorm - log10(pop.data.depth)/depnorm
    else:
        depdists = (trsig.depth)/depnorm - (pop.data.depth)/depnorm
    if logslope:
        slopedists = log10(trsig.slope)/slopenorm - log10(pop.data.slope)/slopenorm
    else:
        slopedists = (trsig.slope)/slopenorm - (pop.data.slope)/slopenorm

    totdists = (durdists**2 + depdists**2 + slopedists**2)
    inds = totdists.argsort()[0:n]

    if plot:
        pu.setfig(fig)
        trsig.plot(fig=0)
        for ind in inds:
            try:
                mod = Modelfrompop(pop,ind)
            except tr.NoEclipseError:
                mod = Modelfrompop(pop,ind,sec=True)
            mod.plot(fig=0,marker='None',ls='-',label='%i' % ind)
        plt.legend(loc='lower right')
        return

    if return_inds:
        return inds
    else:
        return pop.data[inds]


def compare_prophists(f1,f2,prop,model='heb',fig=None,**kwargs):
    pu.setfig(fig)
    f1[model].prophist(prop,histtype='step',lw=3,fig=0,**kwargs)    
    f2[model].prophist(prop,histtype='step',lw=3,fig=0,**kwargs)


def prob_bound(rsky,mag,binpop,bgpop,band='Kepler',mag2=None,band2=None,fb=0.4):

    if mag2 is None:
        lhood_bin = binpop.rsky_lhood(rsky)*binpop.mag_lhood(mag,band)
        lhood_bg = bgpop.rsky_lhood(rsky)*bgpop.mag_lhood(mag,band)
    else:
        lhood_bin = binpop.rsky_lhood(rsky)*binpop.mag_color_lhood(mag,band,mag-mag2,
                                                                   '%s-%s' % (band,band2))
        lhood_bg = bgpop.rsky_lhood(rsky)*bgpop.mag_color_lhood(mag,band,mag-mag2,
                                                                   '%s-%s' % (band,band2))

    tot = lhood_bin*fb + lhood_bg

    return lhood_bin*fb/tot

#custom exceptions

class MCMCError(Exception):
    pass

class EmptyPopulationError(Exception):
    pass

