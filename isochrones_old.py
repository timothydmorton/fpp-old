"""
Compiles stellar model isochrones into an easy-to-access format.  

"""
from numpy import *
from scipy.interpolate import LinearNDInterpolator as interpnd
from consts import *
import os,sys,re
import scipy.optimize
#try:
#    import pymc as pm
#except:
#    print 'isochrones: pymc not loaded!  MCMC will not work'
import numpy.random as rand
import atpy

DATAFOLDER = os.environ['ASTROUTIL_DATADIR'] #'/Users/tdm/Dropbox/astroutil/data'

def gr2B(g,r):
    return gr2V(g,r) + 1.04*(g-r) + 0.19

def gr2V(g,r):
    return r + 0.44*(g-r)-0.02

def BV2r(B,V):
    return (1/(1 + 0.56/0.44))*((1/0.44)*(V+0.02) - (1/1.04)*(B-V-0.19))

def BV2g(B,V):
    return BV2r(B,V) + (1/1.04)*(B-V) - 0.19/1.04


def fehstr(feh,minfeh=-1.0,maxfeh=0.5):
        if feh < minfeh:
            return '%.1f' % minfeh
        elif feh > maxfeh:
            return '%.1f' % maxfeh
        elif (feh > -0.05 and feh < 0):
            return '0.0'
        else:
            return '%.1f' % feh            



class isochrone(object):
    """Generic isochrone class."""
    def __init__(self,age,m_ini,m_act,logL,Teff,logg,mags,fehs=None):
        self.minage = age.min()
        self.maxage = age.max()
        self.minmass = m_act.min()
        self.maxmass = m_act.max()
        
        self.bands = []
        for band in mags:
            self.bands.append(band)
        L = 10**logL
        #R = sqrt(G*m_act*MSUN/10**logg)/RSUN

        if fehs is None:
            points = array([m_ini,age]).T
            self.is3d = False
        else:
            points = array([m_ini,age,fehs]).T
            self.is3d = True

        if self.is3d:
            self.feh = lambda m,age,feh: feh
        else:
            self.feh = lambda m,age: self.isofeh

        self.M = interpnd(points,m_act)
        self.tri = self.M.tri
        #self.R = interpnd(points,R)
        self.logL = interpnd(self.tri,logL)
        self.logg = interpnd(self.tri,logg)
        self.logTe = interpnd(self.tri,log10(Teff))
        def Teff_fn(*pts):
            return 10**self.logTe(*pts)
        #self.Teff = lambda *pts: 10**self.logTe(*pts)
        self.Teff = Teff_fn
        def R_fn(*pts):
            return sqrt(G*self.M(*pts)*MSUN/10**self.logg(*pts))/RSUN
        #self.R = lambda *pts: sqrt(G*self.M(*pts)*MSUN/10**self.logg(*pts))/RSUN
        self.R = R_fn

        self.mag = {}
        for band in self.bands:
            self.mag[band] = interpnd(self.tri,mags[band])

    def __call__(self,*args):
        if self.is3d:
            if len(args) != 3:
                raise ValueError('must call with M, age, and [Fe/H]')
            m,age,feh = args
        else:
            if len(args) != 2:
                raise ValueError('must call with M,age')
            m,age = args
        Ms = self.M(*args)
        Rs = self.R(*args)
        logLs = self.logL(*args)
        loggs = self.logg(*args)
        Teffs = self.Teff(*args)
        mags = {}
        for band in self.bands:
            mags[band] = self.mag[band](*args)
        return {'age':age,'M':Ms,'feh':self.feh(*args),'R':Rs,'logL':logLs,'logg':loggs,'Teff':Teffs,'mag':mags}        

    def evtrack(self,m,minage=6.7,maxage=10,dage=0.05):
        ages = arange(minage,maxage,dage)
        Ms = self.M(m,ages)
        Rs = self.R(m,ages)
        logLs = self.logL(m,ages)
        loggs = self.logg(m,ages)
        Teffs = self.Teff(m,ages)
        mags = {}
        for band in self.bands:
            mags[band] = self.mag[band](m,ages)

        #return array([ages,Ms,Rs,logLs,loggs,Teffs,   #record array?
        return {'age':ages,'M':Ms,'R':Rs,'logL':logLs,'Teff':Teffs,'mag':mags}
            
    def isochrone(self,age,minm=0.1,maxm=2,dm=0.02):
        ms = arange(minm,maxm,dm)
        ages = ones(ms.shape)*age

        Ms = self.M(ms,ages)
        Rs = self.R(ms,ages)
        logLs = self.logL(ms,ages)
        loggs = self.logg(ms,ages)
        Teffs = self.Teff(ms,ages)
        mags = {}
        for band in self.bands:
            mags[band] = self.mag[band](ms,ages)

        return {'M':Ms,'R':Rs,'logL':logLs,'Teff':Teffs,'mag':mags}        
        

class WD(isochrone):
    def __init__(self,composition='H'):
        if composition not in ['H','He']:
            raise ValueError('Unknown composition: %s (must be H or He)' % 
                             composition)
        self.composition = composition
        filename = '%s/stars/WDs_%s.txt' % (DATAFOLDER,composition)
        data = recfromtxt(filename,names=True)
        mags = {'bol':data.Mbol,'U':data.U,'B':data.B,'V':data.V,'R':data.R,'I':data.I,
                'J':data.J,'H':data.H,'K':data.K,'u':data.u,'g':data.g,'r':data.r,
                'i':data.i,'z':data.z,'y':data.y}
        logL = -2.5*(data.Mbol-4.77)

        gr = mags['g']-mags['r']
        
        mags['kep'] = 0.25*mags['g'] + 0.75*mags['r']
        w = where(gr > 0.3)
        mags['kep'][w] = 0.3*mags['g'][w] + 0.7*mags['r'][w]


        isochrone.__init__(self,log10(data.Age),data.mass,data.mass,logL,
                           data.Teff,data.logg,mags)


class padova(isochrone):
    def __init__(self,feh=0.):
        filename = DATAFOLDER + '/stars/padova_%s.dat' % fehstr(feh,-2,0.2)
        self.isofeh = feh
        #filename = 'data/kepisochrones.dat'
        age,m_ini,m_act,logL,logT,logg,mbol,kep,g,r,i,z,dd051,J,H,K = \
            loadtxt(filename,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),unpack=True)
        mags = {'bol':mbol,'kep':kep,'g':g,'r':r,'i':i,'z':z,'dd051':dd051,'J':J,'H':H,'K':K}
        mags['B'] = gr2B(g,r)
        mags['V'] = gr2V(g,r)
        mags['R'] = r  #cheating
        mags['I'] = i  #cheating
        mags['Kepler'] = mags['kep']
        isochrone.__init__(self,age,m_ini,m_act,logL,10**logT,logg,mags)

class padova3d(isochrone):
    def __init__(self,minm=0.9,maxm=1.1,minage=9,maxage=10,minfeh=-0.2,maxfeh=0.2):
        #ages = array([]); m_inis = array([]); m_acts = array([]); logLs = array([]); logTs = array([]); loggs = array([])
        #mbols = array([]); keps = array([]); gs = array([]); rs = array([]); iz = array([]); zs = array([])
        #dd051s = array([]); Js = array([])

        data = None
        fehs = array([])
        fehlist = arange(-2,0.3,0.1)
        fehlist = fehlist[where((fehlist >= minfeh) & (fehlist <= maxfeh+0.001))]
        for feh in fehlist:
            print 'loading isochrone for [Fe/H] = %.1f...' % feh
            filename = DATAFOLDER + '/stars/padova_%.1f.dat' % feh
            if data is None:
                data = loadtxt(filename)
                n = len(data)
            else:
                newdata = loadtxt(filename)
                n = len(newdata)
                data = concatenate((data,newdata))
            fehs = concatenate((fehs,ones(n)*feh))

        inds = where((data[:,2] >= minm) & (data[:,2] <= maxm) & (data[:,0] >= minage) & (data[:,0] <= maxage))[0]
        data = data[inds,:]
        fehs = fehs[inds]
        
        age,m_ini,m_act,logL,logT,logg,mbol,kep,g,r,i,z,dd051,J,H,K = \
            (data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5],data[:,6],data[:,7],data[:,8],
             data[:,9],data[:,10],data[:,11],data[:,12],data[:,13],data[:,14],data[:,15])

        self.minm = m_ini.min()
        self.maxm = m_ini.max()
        self.minage = age.min()
        self.maxage = age.max()
        self.minfeh = fehs.min()
        self.maxfeh = fehs.max()

        mags = {'bol':mbol,'kep':kep,'g':g,'r':r,'i':i,'z':z,'dd051':dd051,'J':J,'H':H,'K':K}
        mags['B'] = gr2B(g,r)
        mags['V'] = gr2V(g,r)
        mags['R'] = r  #cheating
        mags['I'] = i  #cheating

        isochrone.__init__(self,age,m_ini,m_act,logL,10**logT,logg,mags,fehs=fehs)
        

class baraffe(isochrone):
    def __init__(self,feh=0):
        filename = '%s/stars/baraffe0.0.txt' % DATAFOLDER
        data = recfromtxt(filename,names=True)
        self.isofeh = feh
        mags = {'V':data.Mv,'R':data.Mr,'I':data.Mi,'J':data.Mj,'H':data.Mh,
                'K':data.Mk,'L':data.Ml,'M':data.Mm}
        #mags['kep'] = (mags['V'] + mags['R'])/2 - 2
        mags['g'] = mags['V']
        mags['r'] = mags['R']
        mags['i'] = mags['I']
        mags['z'] = 0.8*mags['I'] + 0.2*mags['J']
        mags['kep'] = 0.1*mags['g'] + 0.9*mags['r']
        isochrone.__init__(self,log10(data.age*1e9),data.m,data.m,data.logL,
                           data.Teff,data.g,mags)

class dartmouth(isochrone):
    def __init__(self,feh=0,bands=['U','B','V','R','I','J','H','K','g','r','i','z','Kepler']):
        filename = '%s/stars/dartmouth_%s.fits' % (DATAFOLDER,fehstr(feh,-1.0,0.5))
        t = atpy.Table(filename)
        self.isofeh = feh
        mags = {}
        for band in bands:
            try:
                mags[band] = t[band]
            except:
                if band == 'kep' or band == 'Kepler':
                    mags[band] = t['Kp']
                else:
                    raise

        #Fg = 10**(-0.4*mags['g'])
        #Fr = 10**(-0.4*mags['r'])
        #gr = mags['g']-mags['r']
        
        #mags['kep'] = 0.25*mags['g'] + 0.75*mags['r']
        #w = where(gr > 0.3)
        #mags['kep'][w] = 0.3*mags['g'][w] + 0.7*mags['r'][w]
        #mags['kep'] = (mags['g']+mags['r'])/2 #fix this!

        isochrone.__init__(self,log10(t['age']*1e9),t['M'],t['M'],t['logL'],
                           10**t['logTe'],t['logg'],mags)
        

def write_all_dartmouth_to_fits(fehs=arange(-1,0.51,0.1)):
    for feh in fehs:
        try:
            print feh
            dartmouth_to_fits(feh)
        except:
            raise
            pass

def dartmouth_to_fits(feh):
    filename_2mass = '%s/stars/dartmouth_%s_2massKp.iso' % (DATAFOLDER,fehstr(feh,-1.0,0.5))
    filename_ugriz = '%s/stars/dartmouth_%s_ugriz.iso' % (DATAFOLDER,fehstr(feh,-1.0,0.5))
    data_2mass = recfromtxt(filename_2mass,skiprows=8,names=True)
    data_ugriz = recfromtxt(filename_ugriz,skiprows=8,names=True)
    n = len(data_2mass)
    ages = zeros(n)
    curage = 0
    i=0
    for line in open(filename_2mass):
        m = re.match('#',line)
        if m:
            m = re.match('#AGE=\s*(\d+\.\d+)\s+',line)
            if m:
                curage=m.group(1)
        else:
            if re.search('\d',line):
                ages[i]=curage
                i+=1
    t = atpy.Table()
    t.add_column('age',ages)
    t.add_column('M',data_2mass.MMo)
    t.add_column('logTe',data_2mass.LogTeff)
    t.add_column('logg',data_2mass.LogG)
    t.add_column('logL',data_2mass.LogLLo)
    t.add_column('U',data_2mass.U)
    t.add_column('B',data_2mass.B)
    t.add_column('V',data_2mass.V)
    t.add_column('R',data_2mass.R)
    t.add_column('I',data_2mass.I)
    t.add_column('J',data_2mass.J)
    t.add_column('H',data_2mass.H)
    t.add_column('K',data_2mass.Ks)
    t.add_column('Kp',data_2mass.Kp)
    t.add_column('u',data_ugriz.sdss_u)
    t.add_column('g',data_ugriz.sdss_g)
    t.add_column('r',data_ugriz.sdss_r)
    t.add_column('i',data_ugriz.sdss_i)
    t.add_column('z',data_ugriz.sdss_z)
    t.write('%s/stars/dartmouth_%s.fits' % (DATAFOLDER,fehstr(feh,-1,0.5)),overwrite=True)
    


        

def isofit(iso,p0=None,**kwargs):
    def chisqfn(pars):
        if iso.is3d:
            m,age,feh = pars
        else:
            m,age = pars
        tot = 0
        for kw in kwargs:
            val,err = kwargs[kw]
            fn = getattr(iso,kw)
            tot += (val-fn(*pars))**2/err**2
        return tot
    if iso.is3d:
        if p0 is None:
            p0 = ((iso.minm+iso.maxm)/2,(iso.minage + iso.maxage)/2.,(iso.minfeh + iso.maxfeh)/2.)
    else:
        if p0 is None:
            p0 = (1,9.5)
    pfit = scipy.optimize.fmin(chisqfn,p0,disp=False)
    print pfit
    return iso(*pfit)

def shotgun_isofit(iso,n=100,**kwargs):
    simdata = {}
    for kw in kwargs:
        val,err = kwargs[kw]
        simdata[kw] = rand.normal(size=n)*err + val
    if iso.is3d:
        Ms,ages,fehs = (zeros(n),zeros(n),zeros(n))
    else:
        Ms,ages = (zeros(n),zeros(n))
    for i in arange(n):
        simkwargs = {}
        for kw in kwargs:
            val = simdata[kw][i]
            err = kwargs[kw][1]
            simkwargs[kw] = (val,err)
        fit = isofit(iso,**simkwargs)
        Ms[i] = fit['M']
        ages[i] = fit['age']
        if iso.is3d:
            fehs[i] = fit['feh']

    if iso.is3d:
        res = iso(Ms,ages,fehs)
    else:
        res = iso(Ms,ages)
    return res
    
def isofitMCMCmodel(iso,**kwargs):
    if iso.is3d:
        mass = pm.Uniform('mass',lower=iso.minm,upper=iso.maxm)
        age = pm.Uniform('age',lower=iso.minage,upper=iso.maxage)
        feh = pm.Uniform('feh',lower=iso.minfeh,upper=iso.maxfeh)
        ns = {'pm':pm,'mass':mass,'age':age,'feh':feh}
    else:
        mass = pm.Uniform('mass',lower=0.1,upper=5)
        age = pm.Uniform('age',lower=6.7,upper=10.1)
        ns = {'pm':pm,'mass':mass,'age':age}
        
    for kw in kwargs:
        val,dval = kwargs[kw]
        fn = getattr(iso,kw)
        ns['fn'] = fn
        ns['val'] = val
        ns['dval'] = dval
        if iso.is3d:
            code = "@pm.observed(dtype=float)\ndef %s(value=val,mass=mass,age=age,feh=feh): return max(-1000,-(fn(mass,age,feh) - val)**2/dval**2)" % kw
        else:
            code = "@pm.observed(dtype=float)\ndef %s(value=val,mass=mass,age=age): return max(-1000,-(fn(mass,age) - val)**2/dval**2)" % kw
        exec code in ns
    return ns

def isofitMCMC(iso,niter=5e4,nburn=1e4,thin=100,verbose=True,**kwargs):
    model = isofitMCMCmodel(iso,**kwargs)
    M = pm.MCMC(model)
    M.sample(iter=niter,burn=nburn,thin=thin,verbose=verbose)
    return M
