from numpy import *
from scipy.interpolate import UnivariateSpline as interpolate
from astropysics.coords import FK5Coordinates
import subprocess as sp
import os,re,sys,getopt,os.path
import numpy.random as rand
import plotutils as pu
import pylab as p
import statutils as statu
from utils import trapznd
import utils
from consts import *
import orbitutils as ou
import inclination as inc
from scipy.interpolate import LinearNDInterpolator as interpnd
import isochrones_old as iso
from scipy.stats import gaussian_kde

DATAFOLDER = os.environ['ASTROUTIL_DATADIR'] #'/Users/tdm/Dropbox/astroutil/data'

RAGHAVANPERS = recfromtxt('%s/stars/raghavan_periods.dat' % DATAFOLDER)  #raghavan_periods.dat
LOGPERS = log10(RAGHAVANPERS.f1[where(RAGHAVANPERS.f0 == 'Y')])
BINPERS = RAGHAVANPERS.f1[where(RAGHAVANPERS.f0 == 'Y')]
#logperkde = utils.kde(logpers,k=80)
BINPERKDE = utils.kde(BINPERS,adaptive=False)
LOGPERKDE = utils.kde(LOGPERS,adaptive=False)

TRIPDATA = recfromtxt('%s/stars/multiple_pecc.txt' % DATAFOLDER,names=True)
TRIPLEPERS = TRIPDATA.P
TRIPPERKDE = utils.kde(TRIPLEPERS,adaptive=False)
TRIPLOGPERKDE = utils.kde(log10(TRIPLEPERS),adaptive=False)

ECCKDE = gaussian_kde(TRIPDATA.e)

EXTINCTIONFILE = '%s/stars/extinction.txt' % DATAFOLDER
EXTINCTION = dict()
EXTINCTION5 = dict()
for line in open(EXTINCTIONFILE,'r'):
    line = line.split()
    EXTINCTION[line[0]] = float(line[1])
    EXTINCTION5[line[0]] = float(line[2])

EXTINCTION['kep'] = 0.85946
EXTINCTION['V'] = 1.0
EXTINCTION['Ks'] = EXTINCTION['K']
EXTINCTION['Kepler'] = EXTINCTION['kep']

SEDTABLEFILE = '%s/stars/stellarseds.txt' % DATAFOLDER
SEDTABLE = recfromtxt(SEDTABLEFILE,names=True)

PASSBANDS = ['u','g','r','i','z','J','H','K','bol']
WEFFS = {'u':354,'g':477,'r':622,'i':763,'z':905,'J':1235,'H':1662,'K':2159}


MAGFN = dict()
for band in PASSBANDS:
    MAGFN[band] = interpolate(SEDTABLE['teff'][::-1],SEDTABLE[band][::-1],s=0)
MAGFN['kep'] = interpolate(SEDTABLE['teff'][::-1],(SEDTABLE['g'][::-1] + SEDTABLE['r'][::-1])/2.,s=0)

#BARAFFE = iso.baraffe()
#BARAFFEMINAGE = interpolate(array([0.02,0.03,0.04,0.05,0.055,0.06,0.07,0.072,0.075]),
#                       array([0.063,0.159,0.322,0.566,0.808,1,2.8,8.9,10]),k=1,s=0)

DARTBANDS = ['g','r','i','z','J','H','K','kep']
PADBANDS = ['g','r','i','z','J','H','K','kep']
MINFEH_DAR=-0.5
MAXFEH_DAR=0.2
MINFEH_PAD=-0.5
MAXFEH_PAD=0.2
DARTMOUTH = {}
PADOVA = {}
LOWMASS_VALUE = {'R':1.3*RJUP/RSUN,'Teff':1000,'logg':5.2,'mag':15,'logL':-3,'logTe':3}
LOWMASSLIMIT = {'dartmouth':0.117,'padova':0.151}  #implement this below sometime?

BARAFFE_LOWMS = array([ 0.075,  0.08 ,  0.1  ,  0.11 ,  0.13 ])
BARAFFE_LOWTEFFS = array([ 2001.,  2314.,  2813.,  2921.,  3056.])
BARAFFE_LOWTEFF_FN = interpolate(BARAFFE_LOWMS,BARAFFE_LOWTEFFS)

DARTMOUTH_OFFSETS={'M':0.0}
    
for feh in arange(MINFEH_PAD,MAXFEH_PAD+0.01,0.1):
    PADOVA[iso.fehstr(feh,MINFEH_PAD,MAXFEH_PAD)] = iso.padova(feh) #PADBANDS
if '-0.0' in PADOVA:
    PADOVA['0.0'] = PADOVA['-0.0']

for feh in arange(MINFEH_DAR,MAXFEH_DAR+0.01,0.1):
    ic = iso.dartmouth(feh,DARTBANDS)
    for prop in ['R','logg','logL','logTe','Teff']:
        pad = getattr(PADOVA[iso.fehstr(feh,MINFEH_PAD,MAXFEH_PAD)],prop)(LOWMASSLIMIT['padova'],9.3)
        dar = getattr(ic,prop)(LOWMASSLIMIT['padova'],9.3)
        DARTMOUTH_OFFSETS[prop] = pad-dar
    for band in DARTBANDS:
        pad = PADOVA[iso.fehstr(feh,MINFEH_PAD,MAXFEH_PAD)].mag[band](LOWMASSLIMIT['padova'],9.3)
        dar = ic.mag[band](LOWMASSLIMIT['padova'],9.3)
        DARTMOUTH_OFFSETS[band] = pad-dar
            

    DARTMOUTH['%.1f' % feh] = ic
    #find offsets to padvoa isochrones at 0.15 msun

#DARTMOUTH_OFFSETS['kep'] = -1.2   #fix this!

if '-0.0' in DARTMOUTH:
    DARTMOUTH['0.0'] = DARTMOUTH['-0.0']

AGES_PAD,MAXMS_PAD = loadtxt('%s/stars/padova_maxage.txt' % DATAFOLDER,unpack=True)
PADOVAMAXMASS = interpolate(AGES_PAD,MAXMS_PAD,s=0)
PADOVAMAXAGE = interpolate(MAXMS_PAD[::-1],AGES_PAD[::-1],s=0,k=1)

BARAFFE = iso.baraffe()

#at some point, add interpolation here between metallicity isochrones?


def model_property(prop,m,age=9.3,feh=0,ic=None,models='padova',mag=False,verbose=False):
    if re.match('pad',models):
        MINFEH,MAXFEH = (MINFEH_PAD,MAXFEH_PAD)
        models='padova'
    elif re.match('dar',models):
        MINFEH,MAXFEH = (MINFEH_DAR,MAXFEH_DAR)
        models='dartmouth'
    else:
        raise ValueError('unknown stellar models %s' % models)

    m = atleast_1d(m).copy()
    age = atleast_1d(age).copy()
    feh = atleast_1d(feh).copy()
    feh[where(feh<MINFEH)] = MINFEH
    feh[where(feh>MAXFEH)] = MAXFEH

    if size(feh)==1:
        ic_pad = PADOVA[iso.fehstr(feh,MINFEH,MAXFEH)]
        ic_dar = DARTMOUTH[iso.fehstr(feh,MINFEH,MAXFEH)]
        if ic is None:
            if models=='padova':
                ic = ic_pad
            else:
                ic = ic_dar
        age[where(age<ic.minage)] = ic.minage
        age[where(age>ic.maxage)] = ic.maxage
        if mag:
            x = ic.mag[prop](m,age)
        else:
            x = getattr(ic,prop)(m,age)
        if models=='padova':
            lo = (m < LOWMASSLIMIT['padova'])
            if size(lo)>0:
                age[where(lo & (age<ic_dar.minage))] = ic_dar.minage
                if mag:
                    darval = ic_dar.mag[prop](m,age)
                else:
                    darval = getattr(ic_dar,prop)(m,age)
                ok = (~isnan(darval))
                w = (lo & ok)
                x[w] = darval[w] + DARTMOUTH_OFFSETS[prop]
        wlo = where(m < LOWMASSLIMIT['dartmouth'])
        if mag:
            #teffs = BARAFFE_LOWTEFF_FN(m[wlo])
            #x[wlo] = MAGFN[prop](teffs)
            x[wlo] = BARAFFE.mag[prop](m[wlo],9) # Baraffe age fixed to 1gyr
        elif prop == 'M':
            x[wlo] = m[wlo]
        elif prop == 'Teff':
            #x[wlo] = BARAFFE_LOWTEFF_FN(m[wlo])
            x[wlo] = BARAFFE.Teff(m[wlo],9)
        else:
            #x[wlo] = LOWMASS_VALUE[prop]
            x[wlo] = getattr(BARAFFE,prop)(m[wlo],9)
    else:
        fehs = arange(MINFEH-0.1,MAXFEH+0.1,0.1)
        bins = fehs+0.05
        inds = digitize(feh,bins)

        x = zeros(size(feh))

        lo = (m < LOWMASSLIMIT['padova'])
        for i,f in enumerate(fehs):
            if i==0:
                continue
            ic_pad = PADOVA[iso.fehstr(f,MINFEH,MAXFEH)]
            ic_dar = DARTMOUTH[iso.fehstr(f,MINFEH,MAXFEH)]
            if models=='padova':
                ic = ic_pad
            else:
                ic = ic_dar
            inbin = (i==inds)
            if verbose:
                print 'feh = %.1f, inbin = %i' % (f,inbin.sum())
            winbin = where(inbin)
            age[where(inbin & (age<ic.minage))] = ic.minage
            age[where(inbin & (age>ic.maxage))] = ic.maxage
            if mag:
                val = ic.mag[prop](m,age)
            else:
                val = getattr(ic,prop)(m,age)
            x[winbin] = val[winbin]
            #print f
            #print inbin
            #print x
            if models=='padova':
                age[where(lo & (age<ic_dar.minage))] = ic_dar.minage
                if mag:
                    darval = ic_dar.mag[prop](m,age)
                else:
                    darval = getattr(ic_dar,prop)(m,age)
                ok = ~isnan(darval)
                w = where(lo & inbin & ok)
                x[w] = darval[w] + DARTMOUTH_OFFSETS[prop]
        wlo = where(m < LOWMASSLIMIT['dartmouth'])
        if mag:
            #teffs = BARAFFE_LOWTEFF_FN(m[wlo])
            #x[wlo] = MAGFN[prop](teffs)
            x[wlo] = BARAFFE.mag[prop](m[wlo],9) # Baraffe age fixed to 1gyr
        elif prop == 'M':
            x[wlo] = m[wlo]
        elif prop == 'Teff':
            #x[wlo] = BARAFFE_LOWTEFF_FN(m[wlo])
            x[wlo] = BARAFFE.Teff(m[wlo],9)
        else:
            #x[wlo] = LOWMASS_VALUE[prop]
            x[wlo] = getattr(BARAFFE,prop)(m[wlo],9)
    wtoo_old = where(m > PADOVAMAXMASS(age))
    x[wtoo_old] = nan
    if mag:
        x[where(x>35)] = 35.
    return x

def model_mag(band,m,age=9.3,feh=0,ic=None,models='padova',verbose=False):
    if band=='Ks':
        band='K'
    if band=='Kepler':
        band='kep'
    return model_property(band,m,age=age,feh=feh,ic=ic,models=models,mag=True,verbose=verbose)

def model_M(m,age=9.3,feh=0,ic=None,models='padova'):
    return model_property('M',m,age,feh,ic,models)
            
def model_R(m,age=9.3,feh=0,ic=None,models='padova'):
    return model_property('R',m,age,feh,ic,models)

def model_Teff(m,age=9.3,feh=0,ic=None,models='padova'):
    return model_property('Teff',m,age,feh,ic,models)

def model_logg(m,age=9.3,feh=0,ic=None,models='padova'):
    return model_property('logg',m,age,feh,ic,models)

def model_logL(m,age=9.3,feh=0,ic=None,models='padova'):
    return model_property('logL',m,age,feh,ic,models)

def model_logTe(m,age=9.3,feh=0,ic=None,models='padova'):
    return log10(model_Teff(m,age,feh,ic,models))


def rochelobe(q):
    """returns r1/a; q = M1/M2"""
    return 0.49*q**(2./3)/(0.6*q**(2./3) + log(1+q**(1./3)))

def withinroche(semimajors,M1,R1,M2,R2):
    q = M1/M2
    return ((R1+R2)*RSUN) > (rochelobe(q)*semimajors*AU)
    
def semimajor(P,mstar=1):
    return ((P*DAY/2/pi)**2*G*mstar*MSUN)**(1./3)/AU

def period_from_a(a,mstar):
    return sqrt(4*pi**2*(a*AU)**3/(G*mstar*MSUN))/DAY

def addmags(mag1,mag2):
    F1 = 10**(-0.4*mag1)
    F2 = 10**(-0.4*mag2)
    try:
        F2 = F2[:,newaxis]
    except:
        pass
    return -2.5*log10(F1+F2)

def plotSED(teff=5770,d=10,AV=0.6,unc=0,fig=None,**kwargs):
    sed = simulateSED(teff,d,AV,unc)
    sed.plot(fig=fig,**kwargs)

def plotbinarySED(teff1=5770,teff2=4500,d=10,AV=0.6,unc=0,fig=None,**kwargs):
    sed = simulatebinarySED(teff1,teff2,d,AV,unc)
    sed.plot(fig=fig,**kwargs)


def plotSEDfits(sed,res1,res2,fig=None):
    pu.setfig(fig)
    sed.plot(fig=0,label='data',psym='o')
    AV1,d1,T1 = res1
    AV2,d2,T21,T22 = res2
    plotSED(T1,d1,AV1,fig=0,label='single')
    plotbinarySED(T21,T22,d2,AV1,fig=0,label='binary',psym='x')
    p.legend(loc='lower right')

class SED(object):
    def __init__(self,mags=None,unc=None,ra=None,dec=None):
        self.mags = mags
        if unc is None or unc==0.04:
            unc = dict()
            for band in mags:
                if band in ('u','g','r','i','z'):
                    unc[band] = 0.04
                if band in ('J','H','K'):
                    unc[band] = 0.02
        self.unc = unc
        self.ra = ra
        self.dec = dec
        todelete = []
        for key in mags:
            if isnan(mags[key]):
                todelete.append(key)
        for band in todelete:
            del mags[band]


    def plot(self,fig=None,psym='+',**kwargs):
        pu.setfig(fig)
        ws = []
        ms = []
        for band in self.mags.keys():
            ws.append(WEFFS[band])
            ms.append(self.mags[band])
        ws = array(ws)
        ms = array(ms)
        inds = argsort(ws)
        p.plot(ws[inds],ms[inds],psym,**kwargs)
        ax = p.gca()
        lims = ax.get_ylim()
        if lims[0] < lims[1]:
            ax.set_ylim(lims[1],lims[0])

    def __add__(self,other):
        newmags = dict()
        for band in self.mags.keys():
            try:
                newmags[band] = addmags(self.mags[band],other.mags[band])
            except:
                pass
        return SED(newmags)

    def __radd__(self,other):
        return self.__add__(other)

    def fit(self,teffs=None,ds=None,AVs=None,minteff=3000,maxteff=8000,mind=50,maxd=2e3,
            dteff=50,nds=200,dAV=0.02,plot=False,fig=None,figsize=(6,6),plotfit=False,
            return_Tpost=False,
            **kwargs):
        
        if teffs is None:
            teffs = arange(minteff,maxteff,dteff)
        if ds is None:
            ds = logspace(log10(mind),log10(maxd),nds)
        if AVs is None:
            if self.ra is not None and self.dec is not None:
                maxAV = getAV(self.ra,self.dec)
            else:
                maxAV = 1.
            AVs = arange(0.,maxAV,dAV)
        #print 'max AV is %.2f' % maxAV
        

        logl,Ts,Ds,As = teff_dm_logl(self.mags,dmags=self.unc,teffs1=teffs,ds=ds,AVs=AVs)
        L = exp(logl)

        Tprior = 1./(maxteff-minteff)
        #Dprior = 1./(Ds * (log(maxd)-log(mind)))
        Dprior = 1./(maxd-mind)
        AVprior = 1./maxAV
        evidence = trapznd(L,As,Ds,Ts) * Tprior * Dprior * AVprior

        Tpost = trapz(trapz(L,As,axis=0),Ds,axis=0)
        if plot:
            pu.setfig(fig,figsize=figsize)
            statu.plot_posterior(Ts,Tpost,'Teff',fig=fig,fmt='%i',**kwargs)
            
        wmax = where(logl==logl.max())
        
        if return_Tpost:
            return Ts,Tpost
        else:
            return (As[wmax[0]][0],Ds[wmax[1]][0],Ts[wmax[2]][0]),evidence
    

    def fitbinary(self,teffs1=None,teffs2=None,ds=None,AVs=None,
                  minteff=3000,maxteff=8000,dteff=50,mind=50,maxd=2e3,nds=200,dAV=0.02,
                  plot=False,fig=None,figsize=(10,10),**kwargs):

        if teffs1 is None:
            teffs1 = arange(minteff,maxteff,dteff)
        if teffs2 is None:
            teffs2 = arange(minteff,6000,dteff)
        if ds is None:
            ds = logspace(log10(mind),log10(maxd),nds)
        if AVs is None:
            if self.ra is not None and self.dec is not None:
                maxAV = getAV(self.ra,self.dec)
            else:
                maxAV = 1.
            AVs = arange(0.,maxAV,dAV)
        #print 'max AV is %.2f' % maxAV

        logl,Ts1,Ts2,Ds,As = teff_dm_logl(self.mags,dmags=self.unc,teffs1=teffs1,
                                          teffs2=teffs2,ds=ds,AVs=AVs,binary=True)
        L = exp(logl)

        #n0,n1,n2,n3 = L.shape
        #foo1,foo2 = meshgrid(Ts1,Ts2)
        #T2frac = sum(foo2 > foo1) / float(size(foo1))
        #w2,w3 = where(foo2 > foo1)
        #nw = len(w2)
        #w0 = concatenate([arange(n0)]*(nw*n1))
        #w1 = concatenate([repeat(arange(n1),n0)]*nw)
        #w = (w0,w1,repeat(w2,(n0*n1)),repeat(w3,(n0*n1)))
        #L[w] = 0

        T1prior = 1./(maxteff-minteff)
        T2prior = 1./(6000-minteff) #* (1./T2frac)
        Dprior = 1./(maxd-mind)
        AVprior = 1./maxAV
        evidence = trapznd(L,As,Ds,Ts2,Ts1) * T1prior * T2prior * Dprior * AVprior

        wmax = where(logl==logl.max())

        Tpost = trapz(trapz(L,As,axis=0),Ds,axis=0)
        if plot:
            pu.setfig(fig,figsize=figsize)
            statu.plot_posterior2d(Ts1,Ts2,Tpost,'Teff_1','Teff_2',fig=0,symmetric=True,
                                   fmt1='%i',fmt2='%i',**kwargs)
        
        return  (As[wmax[0]][0],Ds[wmax[1]][0],Ts2[wmax[2]][0],Ts1[wmax[3]][0]),evidence

    def fitboth(self,plot=False,singleplotfile=None,binaryplotfile=None,title=''):
        singlefit,ev1 = self.fit(plot=plot)
        if plot:
            p.title(title)
            if singleplotfile is not None:
                p.savefig(singleplotfile)
                p.close()
        doublefit,ev2 = self.fitbinary(plot=plot)
        R = log10(ev2/ev1)
        if plot:
            p.annotate('R = %.2f' % R, xy=(0.2,0.6), xycoords='figure fraction')
            p.suptitle(title)
            if binaryplotfile is not None:
                p.savefig(binaryplotfile)
                p.close()

        return singlefit,doublefit,R
            
class BinarySED(SED):
    def __init__(self,teff1,teff2,d=10,AV=0.6,unc=0.,ra=None,dec=None):
        sed = ModelSED(teff1,d,AV,unc) + ModelSED(teff2,d,AV,unc)
        self.mags = sed.mags
        self.unc = unc
        self.AV = AV
        self.d = d
        self.teff1 = teff1
        self.teff2 = teff2
        self.ra = ra
        self.dec = dec

class ModelSED(SED):
    def __init__(self,teff,d=10,AV=0.6,unc=0.,ra=None,dec=None):
        self.mags = simulatemags(teff,d,AV,unc)
        self.unc = unc
        self.AV = AV
        self.d = d
        self.teff = teff
        self.ra = ra
        self.dec = dec

def simulatebinarySED(teff1,teff2,d=10,AV=0.,unc=0.):
    mags = simulatemags_binary(teff1,teff2,d,AV,unc)
    return SED(mags,unc)

def simulateSED(teff,d=10,AV=0.0,unc=0.):
    mags = simulatemags(teff,d,AV,unc)
    return SED(mags,unc)

def simulatemags(teff,d=10,AV=0.,unc=0.):
    mags = dict()
    for band in PASSBANDS:
        try:
            mags[band] = modelmag(teff,band,d,AV)
            if unc != 0:
                mags[band] += rand.normal()*unc
        except:
            continue
    return mags

def simulatemags_binary(teff1,teff2,d=10,AV=0.,unc=0.):
    mags = dict()
    for band in PASSBANDS:
        try:
            mags[band] = addmags(modelmag(teff1,band,d,AV),modelmag(teff2,band,d,AV))
            if unc != 0:
                mags[band] += rand.normal()*unc
        except:
            continue
    return mags

def getAV(ra,dec):
    """gets AV from NED for given RA/Dec using wget!--- represents max AV in a particular direction
    """
    coords = FK5Coordinates(ra,dec)
    rah,ram,ras = coords.ra.hms
    decd,decm,decs = coords.dec.dms
    url = 'http://ned.ipac.caltech.edu/cgi-bin/nph-calc?in_csys=Equatorial&in_equinox=J2000.0&obs_epoch=2010&lon='+'%i' % rah + \
        '%3A'+'%i' % ram + '%3A' + '%05.2f' % ras + '&lat=%2B' + '%i' % decd + '%3A' + '%i' % decm + '%3A' + '%05.2f' % decs + \
        '&pa=0.0&out_csys=Equatorial&out_equinox=J2000.0'
    tmpfile = '/tmp/nedsearch.html'
    cmd = 'wget \'%s\' -O %s -q' % (url,tmpfile)
    sp.Popen(cmd,shell=True).wait()
    for line in open(tmpfile,'r'):
        m = re.search('V \(0.54\)\s+(\S+)',line)
        if m:
            AV = float(m.group(1))
    os.remove(tmpfile)
    return AV

def dfromdm(dm):
    if size(dm)>1:
        dm = atleast_1d(dm)
    return 10**(1+dm/5)

def distancemodulus(d):
    """d in parsec
    """
    if size(d)>1:
        d = atleast_1d(d)
    return 5*log10(d/10)

def modelmag_binary(teff1,teff2,band,distance=10,AV=0.0,RV=3):
    """ can call with multiple Teffs, distances, AVs

    return axis order:  AV, distance, teff2, teff1 
    """
    if band not in PASSBANDS:
        raise ValueError('%s is unrecognized bandpass.' % band)

    T1 = atleast_1d(teff1)
    T2 = atleast_1d(teff2)

    distance = atleast_1d(distance)
    AV = atleast_1d(AV)
    #AV = AV * distance/1000.

    if RV==5:
        A = AV*EXTINCTION5[band]
    else:
        A = AV*EXTINCTION[band]
    
    dm = atleast_1d(distancemodulus(distance))

    M1 = MAGFN[band](T1)
    M2 = MAGFN[band](T2)
    M = addmags(M1,M2)

    #tot = zeros((len(dm),len(T1),len(T2)))
    #for i in arange(len(dm)):
    #    tot[i,:,:] += M + dm[i] + A[i]
    #return tot
    
    D = dm[:,newaxis,newaxis]
    A = A[:,newaxis,newaxis,newaxis]
    #A = A[:,newaxis,newaxis]

    #print M.shape
    #print D.shape
    #print A.shape

    res = M+D+A
    if size(res)==1:
        return res[0]
    else:
        return squeeze(res)

def modelmag(teff,band,distance=10,AV=0.0,RV=3):
    """distance in parsecs; can call with multiple distances and AVs

    return axis order:  AV, distance, teff
    """
    if band not in PASSBANDS:
        raise ValueError('%s is unrecognized bandpass.' % band)

    distance = atleast_1d(distance)
    AV = atleast_1d(AV)
    #AV = AV * distance/1000.
    

    if RV==5:
        A = AV*EXTINCTION5[band]
    else:
        A = AV*EXTINCTION[band]

    if size(distance) > 1 or size(AV) > 1:
        teff = atleast_1d(teff)
        dm = distancemodulus(distance)
        M = MAGFN[band](teff)
        D = dm[:,newaxis]
        A = A[:,newaxis,newaxis]
        #A = resize(A,(M.shape[1],M.shape[0])).T
        #A = A[:,newaxis]
    else:
        M = MAGFN[band](teff)
        D = distancemodulus(distance)

    
    res = M+D+A
    if size(res) == 1:
        return res[0]
    else:
        return res


def teffposterior(mags,ra=None,dec=None,dmags=0.04,RV=3,AV=.601,teffprior=None,dmprior=None,minteff=2200,maxteff=11900,mindm=0,maxdm=15):
    if ra is not None and dec is not None:
        AV = getAV(ra,dec)
    logl,teffs,ds = teff_dm_logl(mags,dmags=dmags,RV=RV,AVs=AV)
    posterior = sum(exp(logl),axis=0)
    posterior /= trapz(posterior,teffs)
    return posterior,teffs

def teff_dm_logl(mags,ra=None,dec=None,dmags=0.04,teffs1=None,teffs2=None,ds=None,AVs=None,RV=3,binary=False,
                 ic=None):
    if type(mags) != type(dict()):
        raise TypeError('mags must be a dictionary.')
    if dmags is not None and type(mags) != type(dict()):
        if type(dmags) != type(1.):
            raise TypeError('dmags must be either a dictionary or a float.')


    if teffs1 is None:
        teffs1 = arange(3000,8000,50)
    if binary and teffs2 is None:
        teffs2 = arange(3000,6000,50)
    if ds is None:
        ds = logspace(2,log10(3e3),175)
    if AVs is None:
        if ra is not None and dec is not None:
            maxAV = getAV(ra,dec)
        else:
            maxAV = 1.
        AVs = arange(0.,maxAV,0.04)

    if binary:
        logl = zeros((len(AVs),len(ds),len(teffs2),len(teffs1)))
    else:
        logl=zeros((len(AVs),len(ds),len(teffs1)))
    
    for band in mags.keys():
        mag = mags[band]
        if type(dmags)==type(dict()):
            dmag = dmags[band]
        else:
            dmag = dmags
        if binary:
            modelmags = modelmag_binary(teffs1,teffs2,band,ds,RV=RV,AV=AVs)
        else:
            modelmags = modelmag(teffs1,band,ds,RV=RV,AV=AVs)
        logl += -0.5*(modelmags-mag)**2/dmag**2
    if binary:
        return logl,teffs1,teffs2,ds,AVs
    else:
        return logl,teffs1,ds,AVs
    

    

def draw_eccs(n,per=10,percirc=12,binsize=0.1,fuzz=0.05,maxecc=0.97):
    if size(per) == 1 or std(atleast_1d(per))==0:
        if size(per)>1:
            per = per[0]
        ne=0
        while ne<10:
            #if per > 25:
            #    w = where((TRIPLEPERS>25) & (TRIPLEPERS<300))
            #else:
            #    w = where(abs(TRIPLEPERS-per)<binsize/2.)
            w = where(absolute(log10(TRIPLEPERS)-log10(per))<binsize/2.)
            es = TRIPDATA.e[w]
            ne = len(es)
            if ne<10:
                binsize*=1.1
        inds = rand.randint(ne,size=n)
        es = es[inds] * (1 + rand.normal(size=n)*fuzz)
    
    #print '%i in period bin, binsize=%.2f' % (ne,binsize)

    else:
        wlong = where(per > 25)
        wshort = where(per <= 25)
        es = zeros(size(per))

        elongs = TRIPDATA.e[where(TRIPLEPERS > 25)]
        eshorts = TRIPDATA.e[where(TRIPLEPERS <= 25)]

        n = size(per)
        nlong = size(wlong[0])
        nshort = size(wshort[0])
        nelongs = size(elongs)
        neshorts = size(eshorts)
        ilongs = rand.randint(nelongs,size=nlong)
        ishorts = rand.randint(neshorts,size=nshort)
        
        es[wlong] = elongs[ilongs]
        es[wshort] = eshorts[ishorts]

    es = es * (1 + rand.normal(size=n)*fuzz)
    es[where(es>maxecc)] = maxecc
    return absolute(es)


    #if per < percirc:
    #    return rand.random(n)*0.05
    #else:
    #    return rand.random(n)*0.6



class BinaryPopulation(ou.OrbitPopulation):
    def __init__(self,M,M2s=None,per=None,ecc=None,n=None):
        M = atleast_1d(M)
        M2s = atleast_1d(M2s)
        if len(M)==1 and len(M2s)==1:
            M = ones(n)*M
            M2s = ones(n)*M2s

        self.M1s = M
        self.M2s = M2s

        if n is None:
            n = len(M2s)
        if per is None:
            pers = 10**LOGPERKDE.draw(n)
        elif size(per)==1:
            pers = per*ones(n)
        else:
            pers = per

        self.Ps = pers

        if ecc is None:
            eccs = draw_eccs(n,pers)
        elif size(ecc) == 1:
            eccs = ones(n)*ecc
        else:
            eccs = ecc
        self.eccs = eccs

        ou.OrbitPopulation.__init__(self,M,M2s,pers,eccs,n)

    
def fbofm(M):
    return 0.45 - (0.7-M)/4



############### old stuff ###################
           #AGE CORRECTION FOR BARAFFE MODELS SHOULD HAPPEN IN ISOCHRONES.PY
def correctage(m,age,feh=0,ic=None):
    if ic is None:
        if feh < -0.5:
            fehstr = '-0.5'
        elif feh > 0.2:
            fehstr = '0.2'
        else:
            fehstr = '%.1f' % feh
            
        ic = PADOVA[fehstr]
    age = atleast_1d(age).astype(float)
    m = atleast_1d(m)
    if size(age)==1:
        age = ones(m.shape)*age
    w = where((m <= 0.075) & (age > BARAFFEMINAGE(m)))
    if size(w) > 0:
        age[w] = BARAFFEMINAGE(m[w])
    age[where(age>ic.maxage)] = ic.maxage
    age[where(age<ic.minage)] = ic.minage
    return age


def IMFdraw(n,A=0.086,mc=0.22,sig=0.57,maxm=100,minm=0.072,single=False):
    if single:
        A = 0.158
        mc = 0.079
        sig = 0.69
        
    logm = rand.normal(size=n)*sig + log10(mc)
    w = where((logm > log10(maxm)) | (logm < log10(minm)))
    nw = size(w)
    while nw > 0:
        logm[w] = rand.normal(size=nw)*sig + mc
        w = where((logm > log10(maxm)) | (logm < log10(minm)))
        nw = size(w)
    return 10**logm

