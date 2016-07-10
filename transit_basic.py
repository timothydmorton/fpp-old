from numpy import *
import numpy as np
import orbitutils as ou #make this faster by c-ifying Efn interpolation?
import scipy.optimize
from scipy.ndimage import convolve1d
from consts import *
from scipy.interpolate import LinearNDInterpolator as interpnd

import os

try:
    import transit_utils as tru
except ImportError:
    print 'transit_basic: did not import transit_utils.'
import emcee
import numpy.random as rand

DATAFOLDER = os.environ['ASTROUTIL_DATADIR']

LDDATA = recfromtxt('%s/keplerld.dat' % DATAFOLDER,names=True)
LDWOK = where((LDDATA.teff < 10000) & (LDDATA.logg > 2.0) & (LDDATA.feh > -2))
#LDPOINTS = array([LDDATA.teff[LDWOK],LDDATA.logg[LDWOK],LDDATA.feh[LDWOK]]).T
LDPOINTS = array([LDDATA.teff[LDWOK],LDDATA.logg[LDWOK]]).T
U1FN = interpnd(LDPOINTS,LDDATA.u1[LDWOK])
U2FN = interpnd(LDPOINTS,LDDATA.u2[LDWOK])

def ldcoeffs(teff,logg=4.5,feh=0):
    teffs = atleast_1d(teff)
    loggs = atleast_1d(logg)
    #fehs = atleast_1d(feh)
    #where statements here
    Tmin,Tmax = (LDPOINTS[:,0].min(),LDPOINTS[:,0].max())
    gmin,gmax = (LDPOINTS[:,1].min(),LDPOINTS[:,1].max())
    #fehmin,fehmax = (LDPOINTS[:,2].min(),LDPOINTS[:,2].max())
    teffs[where(teffs < Tmin)] = Tmin + 1
    teffs[where(teffs > Tmax)] = Tmax - 1
    loggs[where(loggs < gmin)] = gmin + 0.01
    loggs[where(loggs > gmax)] = gmax - 0.01
    #fehs[where(fehs < fehmin)] = fehmin + 0.01
    #fehs[where(fehs > fehmax)] = fehmax - 0.01
    #print teffs,loggs,fehs

    #u1,u2 = (U1FN(teffs,loggs,fehs),U2FN(teffs,loggs,fehs))
    u1,u2 = (U1FN(teffs,loggs),U2FN(teffs,loggs))
    return u1,u2


def correct_fs(fs):
    """ patch-y fix to anything with messed-up fs
    """
    fflat = fs.ravel().copy()
    wbad = np.where(fflat > 1)[0]     

    #identify lowest and highest index of valid flux
    ilowest=0
    while fflat[ilowest] > 1:
        ilowest += 1
    ihighest = len(fflat)-1
    while fflat[ihighest] > 1:
        ihighest -= 1

    wlo = wbad - 1
    whi = wbad + 1


    #find places where wlo index is still in wbad
    ilo = np.searchsorted(wbad,wlo)
    mask = wbad[ilo]==wlo
    while np.any(mask):
        wlo[mask] -= 1
        ilo = np.searchsorted(wbad,wlo)
        mask = wbad[ilo]==wlo
    ihi = np.searchsorted(wbad,whi)
    ihi = np.clip(ihi,0,len(wbad)-1) #make sure no IndexError
    mask = wbad[ihi]==whi

    while np.any(mask):
        whi[mask] += 1
        ihi = np.searchsorted(wbad,whi)
        ihi = np.clip(ihi,0,len(wbad)-1) #make sure no IndexError
        mask = wbad[ihi]==whi
    
    wlo = np.clip(wlo,ilowest,ihighest)
    whi = np.clip(whi,ilowest,ihighest)

    fflat[wbad] = (fflat[whi] + fflat[wlo])/2. #slightly kludge-y, esp. if there are consecutive bad vals
    return fflat.reshape(fs.shape)


class MAInterpolationFunction(object):
    def __init__(self,u1=0.394,u2=0.261,pmin=0.007,pmax=2,nps=200,nzs=200,zmax=None):
    #def __init__(self,pmin=0.007,pmax=2,nps=500,nzs=500):
        self.u1 = u1
        self.u2 = u2
        self.pmin = pmin
        self.pmax = pmax
        if zmax is None:
            zmax = 1+pmax
        self.zmax = zmax
        self.nps = nps

        ps = logspace(log10(pmin),log10(pmax),nps)
        if pmax < 0.5:
            zs = concatenate([array([0]),ps-1e-10,ps,arange(pmax,1-pmax,0.01),
                              arange(1-pmax,zmax,0.005)])
        elif pmax < 1:
            zs = concatenate([array([0]),ps-1e-10,ps,arange(1-pmax,zmax,0.005)])
        else:
            zs = concatenate([array([0]),ps-1e-10,ps,arange(pmax,zmax,0.005)])

        self.nzs = size(zs)
        #zs = linspace(0,zmax,nzs)
        #zs = concatenate([zs,ps,ps+1e-10])

        mu0s = zeros((size(ps),size(zs)))
        lambdads = zeros((size(ps),size(zs)))
        etads = zeros((size(ps),size(zs)))
        fs = zeros((size(ps),size(zs)))
        for i,p0 in enumerate(ps):
            f,res = occultquad(zs,u1,u2,p0,return_components=True)
            mu0s[i,:] = res[0]
            lambdads[i,:] = res[1]
            etads[i,:] = res[2]
            fs[i,:] = f
        P,Z = meshgrid(ps,zs)
        points = array([P.ravel(),Z.ravel()]).T
        self.mu0 = interpnd(points,mu0s.T.ravel())
        
        ##need to make two interpolation functions for lambdad 
        ## b/c it's strongly discontinuous at z=p
        mask = (Z<P)
        pointmask = points[:,1] < points[:,0]

        w1 = where(mask)
        w2 = where(~mask)
        wp1 = where(pointmask)
        wp2 = where(~pointmask)

        self.lambdad1 = interpnd(points[wp1],lambdads.T[w1].ravel())
        self.lambdad2 = interpnd(points[wp2],lambdads.T[w2].ravel())
        def lambdad(p,z):
            #where p and z are exactly equal, this will return nan....
            p = atleast_1d(p)
            z = atleast_1d(z)
            l1 = self.lambdad1(p,z)
            l2 = self.lambdad2(p,z)
            bad1 = isnan(l1)
            l1[where(bad1)]=0
            l2[where(~bad1)]=0
            return l1*~bad1 + l2*bad1
        self.lambdad = lambdad
        
        #self.lambdad = interpnd(points,lambdads.T.ravel())
        self.etad = interpnd(points,etads.T.ravel())        
        self.fn = interpnd(points,fs.T.ravel())

    def __call__(self,ps,zs,u1=.394,u2=0.261,force_broadcast=False,fix=False):
        """  returns array of fluxes; if ps and zs aren't the same shape, then returns array of 
        shape (nps, nzs)
        """
        #return self.fn(ps,zs)

        if size(ps)>1 and (size(ps)!=size(zs) or force_broadcast):
            P = ps[:,newaxis]
            if size(u1)>1 or size(u2)>1:
                if u1.shape != ps.shape or u2.shape != ps.shape:
                    raise ValueError('limb darkening coefficients must be same size as ps')
                U1 = u1[:,newaxis]
                U2 = u2[:,newaxis]
            else:
                U1 = u1
                U2 = u2
        else:
            P = ps
            U1 = u1
            U2 = u2

        if size(u1)>1 or any(u1 != self.u1) or any(u2 != self.u2):
            mu0 = self.mu0(P,zs)
            lambdad = self.lambdad(P,zs)
            etad = self.etad(P,zs)
            fs = 1. - ((1-U1-2*U2)*(1-mu0) + (U1+2*U2)*(lambdad+2./3*(P > zs)) + U2*etad)/(1.-U1/3.-U2/6.)
            if fix:
                fs = correct_fs(fs)
        else:
            fs = self.fn(P,zs)

        return fs

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
    inc = np.pi/2 - b*RSUN/a

    return P*DAY/np.pi*np.arcsin(Rs*RSUN/a * np.sqrt((1-k)**2 - b**2)/np.sin(inc)) *\
        np.sqrt(1-ecc**2)/(1+ecc*np.sin(w*pi/180)) / 3600.#*24*60    

#def transit_T(*args,**kwargs):
#    return (transit_T14(*args,**kwargs) + transit_T23(*args,**kwargs))/2


def eclipse_depth(mafn,Rp,Rs,b,u1=0.394,u2=0.261,max_only=False,npts=100,force_1d=False):
    """ Calculates average (or max) eclipse depth

    ***why does b>1 take so freaking long?...
    """
    k = Rp*REARTH/(Rs*RSUN)

    if max_only:
        return 1 - mafn(k,b,u1,u2)

    if np.size(b) == 1:
        x = np.linspace(0,np.sqrt(1-b**2),npts)
        y = b
        zs = np.sqrt(x**2 + y**2)
        fs = mafn(k,zs,u1,u2) # returns array of shape (nks,nzs)
        depth = 1-fs
    else:
        xmax = np.sqrt(1-b**2)
        x = np.linspace(0,1,npts)*xmax[:,newaxis]
        y = b[:,newaxis]
        zs = np.sqrt(x**2 + y**2)
        fs = mafn(k,zs.ravel(),u1,u2)
        if not force_1d:
            fs = fs.reshape(size(k),*zs.shape)
        depth = 1-fs
    
    meandepth = np.squeeze(depth.mean(axis=depth.ndim-1))

    #if np.ndim(depth)==1:
    #    meandepth = depth.mean()
    #else:
    #    meandepth = depth.mean(axis=depth.ndim-1)

    return meandepth  #array of average depths, shape (nks,nbs)


def rochelobe(q):
    """returns r1/a.  q = M1/M2"""
    return 0.49*q**(2./3)/(0.6*q**(2./3) + log(1+q**(1./3)))

def withinroche(semimajors,M1,R1,M2,R2):
    q = M1/M2
    return ((R1+R2)*RSUN) > (rochelobe(q)*semimajors*AU)
    
def semimajor(P,mstar=1):
    return ((P*DAY/2/pi)**2*G*mstar*MSUN)**(1./3)/AU


def minimum_inclination(P,M1,M2,R1,R2):
    P,M1,M2,R1,R2 = (atleast_1d(P),atleast_1d(M1),atleast_1d(M2),atleast_1d(R1),atleast_1d(R2))
    semimajors = semimajor(P,M1+M2)
    rads = ((R1+R2)*RSUN/(semimajors*AU))
    #wok = where(~isnan(R1) & ~isnan(R2) & ~withinroche(semimajors,M1,R1,M2,R2))
    wok = where(~isnan(rads) & ~withinroche(semimajors,M1,R1,M2,R2))
    if size(wok) == 0:
        print 'P:',P
        print 'M1:',M1
        print 'M2:',M2
        print 'R1:',R1
        print 'R2:',R2
        if all(withinroche(semimajors,M1,R1,M2,R2)):
            raise AllWithinRocheError('All simulated systems within Roche lobe')
        else:
            raise EmptyPopulationError('no valid systems! (see above)')
    mininc = arccos(rads[wok].max())*180/pi
    return mininc

def a_over_Rs(P,R2,M2,M1=1,R1=1,planet=True):
    if planet:
        M2 *= REARTH/RSUN
        R2 *= MEARTH/MSUN
    return semimajor(P,M1+M2)*AU/(R1*RSUN)

class MAModel(object):
    def __init__(self,ts,fs,P=None,tc=None,texp=30,ldcoeffs=None):
        """ts,fs are times, fluxes, texp is exposure time in minutes"""
        self.ts = ts
        self.fs = fs
        self.P = P
        self.tc = tc
        self.texp = texp
        self.ldcoeffs = ldcoeffs

        #self.resampled_ts = 

    def call(self,pars):
        if self.ldcoeffs is None:
            tc,p,b,aR,u1,u2 = pars
        else:
            tc,p,b,aR = pars

        #zs = eclipse_z(self.ts,self.P)

def eclipse_z(ts,P,tc,b,aR,ecc=0,w=0,sec=False,debug=False,approx=True):
    """ returns the zs corresponding to the provided ts for an eclipse of period P and tc
    """
    if ecc != 0 or w != 0:
        raise NotImplementedError('eclipse_z function not implemented for non-circular orbits')
    if sec == True:
        raise NotImplementedError('eclipse_z function not implemented for occulations')
    
    #convert t to true anomaly f (f = pi/2 - w at transit; -pi/2 - w at occultation)
    if approx:
        fs = mod((mod(ts-tc,P)/P + 0.25)*2*pi,(2*pi))  #temporary hack; properly makes tc -> pi/2 (ignoring w for now)        
        inc = arccos(b/aR*(1+ecc*sin(w*pi/180))/(1-ecc**2))

        zs = aR*(1-ecc**2)/(1+ecc*cos(fs))*sqrt(1-(sin(w*pi/180 + fs))**2 * (sin(inc))**2)
        
    else:
        raise ValueError('eclipse_z function not implemented for exact orbit calculation')

    return zs


def eclipse_tz(P,b,aR,ecc=0,w=0,npts=200,width=1.5,sec=False,dt=1,approx=False,new=False,debug=False):
    """Returns ts and zs for an eclipse (npts points right around the eclipse)
    """
    if sec:
        eccfactor = sqrt(1-ecc**2)/(1-ecc*sin(w*pi/180))
    else:
        eccfactor = sqrt(1-ecc**2)/(1+ecc*sin(w*pi/180))
    if eccfactor < 1:
        width /= eccfactor
        #if width > 5:
        #    width = 5
        
    if new:
        Ms = linspace(-pi,pi,2e3)
        if ecc != 0:
            Es = ou.Efn(Ms,ecc) #eccentric anomalies
        else:
            Es = Ms
        zs,in_eclipse = tru.find_eclipse(Es,b,aR,ecc,w,width,sec)

        #if debug:
        #    win = where(in_eclipse)
        #    print win
        #    print zs[win]

        if in_eclipse.sum() < 2:
            raise NoEclipseError

        wecl = where(in_eclipse)
        subMs = Ms[wecl]

        dMs = subMs[1:] - subMs[:-1]

        if any(subMs < 0) and dMs.max()>1: #if there's a discontinuous wrap-around...
            subMs[where(subMs < 0)] += 2*pi


        if debug:
            print subMs


        minM,maxM = (subMs.min(),subMs.max())
        if debug:
            print minM,maxM
        dM = 2*pi*dt/(P*24*60)   #the spacing in mean anomaly that corresponds to dt (minutes)
        Ms = arange(minM,maxM+dM,dM)
        if ecc != 0:
            Es = ou.Efn(Ms,ecc) #eccentric anomalies
        else:
            Es = Ms

        zs,in_eclipse = tru.find_eclipse(Es,b,aR,ecc,w,width,sec)

        #if debug:
        #    print zs
        #    print Ms

        Mcenter = Ms[zs.argmin()]
        phs = (Ms - Mcenter) / (2*pi)
        #if debug:
        #    print Ms-Mcenter
        ts = phs*P
        return ts,zs
    
    if not approx:
        if sec:
            inc = arccos(b/aR*(1-ecc*sin(w*pi/180))/(1-ecc**2))
        else:
            inc = arccos(b/aR*(1+ecc*sin(w*pi/180))/(1-ecc**2))

        Ms = linspace(-pi,pi,2e3) #mean anomalies around whole orbit
        if ecc != 0:
            Es = ou.Efn(Ms,ecc) #eccentric anomalies
            nus = 2 * arctan2(sqrt(1+ecc)*sin(Es/2),sqrt(1-ecc)*cos(Es/2)) #true anomalies
        else:
            nus = Ms

        r = aR*(1-ecc**2)/(1+ecc*cos(nus))  #secondary distance from primary in units of R1

        X = -r*cos(w*pi/180 + nus)
        Y = -r*sin(w*pi/180 + nus)*cos(inc)
        rsky = sqrt(X**2 + Y**2)

        if not sec:
            inds = where((sin(nus + w*pi/180) > 0) & (rsky < width))  #where "front half" of orbit and w/in width
        if sec:
            inds = where((sin(nus + w*pi/180) < 0) & (rsky < width))  #where "front half" of orbit and w/in width
        subMs = Ms[inds].copy()

        if any((subMs[1:]-subMs[:-1]) > pi):
            subMs[where(subMs < 0)] += 2*pi

        if size(subMs)<2:
            print subMs
            raise NoEclipseError

        minM,maxM = (subMs.min(),subMs.max())
        dM = 2*pi*dt/(P*24*60)   #the spacing in mean anomaly that corresponds to dt (minutes)
        Ms = arange(minM,maxM+dM,dM)
        if ecc != 0:
            Es = ou.Efn(Ms,ecc) #eccentric anomalies
            nus = 2 * arctan2(sqrt(1+ecc)*sin(Es/2),sqrt(1-ecc)*cos(Es/2)) #true anomalies
        else:
            nus = Ms
        r = aR*(1-ecc**2)/(1+ecc*cos(nus))
        X = -r*cos(w*pi/180 + nus)
        Y = -r*sin(w*pi/180 + nus)*cos(inc)
        zs = sqrt(X**2 + Y**2)  #rsky
    
        #center = absolute(X).argmin()
        #c = polyfit(Ms[center-1:center+2],X[center-1:center+2],1)
        #Mcenter = -c[1]/c[0]
        if not sec:
            Mcenter = Ms[absolute(X[where(sin(nus + w*pi/180) > 0)]).argmin()]
        else:
            Mcenter = Ms[absolute(X[where(sin(nus + w*pi/180) < 0)]).argmin()]
        phs = (Ms - Mcenter) / (2*pi)
        wmin = absolute(phs).argmin()
        ts = phs*P

        return ts,zs
    else:
        if sec:
            f0 = -pi/2 - (w*pi/180)
            inc = arccos(b/aR*(1-ecc*sin(w*pi/180))/(1-ecc**2))
        else:
            f0 = pi/2 - (w*pi/180)
            inc = arccos(b/aR*(1+ecc*sin(w*pi/180))/(1-ecc**2))
        fmin = -arcsin(1./aR*sqrt(width**2 - b**2)/sin(inc))
        fmax = arcsin(1./aR*sqrt(width**2 - b**2)/sin(inc))
        if isnan(fmin) or isnan(fmax):
            raise NoEclipseError('no eclipse:  P=%.2f, b=%.3f, aR=%.2f, ecc=%0.2f, w=%.1f' % (P,b,aR,ecc,w))
        fs = linspace(fmin,fmax,npts)
        if sec:
            ts = fs*P/2./pi * sqrt(1-ecc**2)/(1 - ecc*sin(w)) #approximation of constant angular velocity
        else:
            ts = fs*P/2./pi * sqrt(1-ecc**2)/(1 + ecc*sin(w)) #approximation of constant ang. vel.
        fs += f0
        rs = aR*(1-ecc**2)/(1+ecc*cos(fs))
        xs = -rs*cos(w*pi/180 + fs)
        ys = -rs*sin(w*pi/180 + fs)*cos(inc)
        zs = aR*(1-ecc**2)/(1+ecc*cos(fs))*sqrt(1-(sin(w*pi/180 + fs))**2 * (sin(inc))**2)
        return ts,zs

def eclipse_pars(P,M1,M2,R1,R2,ecc=0,inc=90,w=0,sec=False):
    """retuns p,b,aR from P,M1,M2,R1,R2,ecc,inc,w"""
    a = semimajor(P,M1+M2)
    if sec:
        b = a*AU*cos(inc*pi/180)/(R1*RSUN) * (1-ecc**2)/(1 - ecc*sin(w*pi/180))
        aR = a*AU/(R2*RSUN)
        p0 = R1/R2
    else:
        b = a*AU*cos(inc*pi/180)/(R1*RSUN) * (1-ecc**2)/(1 + ecc*sin(w*pi/180))
        aR = a*AU/(R1*RSUN)
        p0 = R2/R1
    return p0,b,aR

def eclipse(p0,b,aR,P=1,ecc=0,w=0,xmax=1.5,npts=200,MAfn=None,u1=0.394,u2=0.261,width=3,conv=False,texp=0.0204,frac=1,sec=False,dt=2,approx=False,new=False,debug=False):
    """ frac is fraction of total light in eclipsed object"""
    if sec:
        ts,zs = eclipse_tz(P,b/p0,aR/p0,ecc,w,npts=npts,width=(1+1/p0)*width,sec=sec,dt=dt,approx=approx,new=new,debug=debug)
        if zs.min() > (1 + 1/p0):
            raise NoEclipseError
    else:
        ts,zs = eclipse_tz(P,b,aR,ecc,w,npts=npts,width=(1+p0)*width,sec=sec,dt=dt,approx=approx,new=new,debug=debug)
        if zs.min() > (1+p0):
            raise NoEclipseError
        
    if MAfn is None:
        if sec:
            fs = occultquad(zs,u1,u2,1/p0)
        else:
            fs = occultquad(zs,u1,u2,p0)            
    else:
        if sec:
            fs = MAfn(1/p0,zs,u1,u2)
        else:
            fs = MAfn(p0,zs,u1,u2)
        fs[where(isnan(fs))] = 1.

    if conv:
        dt = ts[1]-ts[0]
        npts = round(texp/dt)
        if npts % 2 == 0:
            npts += 1
        boxcar = ones(npts)/npts
        fs = convolve1d(fs,boxcar)
    fs = 1 - frac*(1-fs)
    return ts,fs #ts are in the same units P is given in.

def calcT14(p0,b,aR,P,ecc=0,w=0,sec=False):
    if sec:
        inc = arccos(b/aR*(1-ecc*sin(w*pi/180))/(1-ecc**2))
        return P/pi*arcsin(1/aR*sqrt((1+p0)**2 - b**2)/sin(inc))*sqrt(1-ecc**2)/(1-ecc*sin(w*pi/180))
    else:
        inc = arccos(b/aR*(1+ecc*sin(w*pi/180))/(1-ecc**2))
        return P/pi*arcsin(1/aR*sqrt((1+p0)**2 - b**2)/sin(inc))*sqrt(1-ecc**2)/(1+ecc*sin(w*pi/180))

def eclipse_tt(p0,b,aR,P=1,ecc=0,w=0,xmax=1.5,npts=200,MAfn=None,u1=0.394,u2=0.261,leastsq=True,conv=False,texp=0.0204,frac=1,sec=False,new=True,debug=False,pars0=None):
    ts,fs = eclipse(p0,b,aR,P,ecc,w,xmax,npts,MAfn,u1,u2,conv=conv,texp=texp,frac=frac,sec=sec,new=new,debug=debug)
    if debug:
        print p0,',',b,',',aR,',',P,',',ecc,',',w,',',xmax,',',npts,',',None,',',u1,',',u2,',',leastsq,',',conv,',',texp,',',frac,',',sec,',',new
        print ts,fs

    #durguess = ((fs < 1).sum()/float(len(fs)) * (ts[-1] - ts[0]))/2.
    #depguess = (1-fs.min())
    #pars = array([durguess,depguess,3.,0.])
    #if debug:
    #    print p0,b,aR,P,ecc,w,'frac=',frac
    #    print pars
    if pars0 is None:
        depth = 1 - fs.min()
        duration = (fs < (1-0.01*depth)).sum()/float(len(fs)) * (ts[-1] - ts[0])
        tc0 = ts[fs.argmin()]
        pars0 = array([duration,depth,5.,tc0])
    
    dur,dep,slope,epoch = fit_traptransit(ts,fs,pars0,debug=debug)
    return dur,dep,slope


def eclipse_pp(p0,b,aR,P=1,ecc=0,w=0,xmax=1.5,npts=200,MAfn=None,u1=0.394,u2=0.261,leastsq=True,conv=False,texp=0.0204,frac=1,sec=False,use_fortran=False,new=False):
    ts,fs = eclipse(p0,b,aR,P,ecc,w,xmax,npts,MAfn,u1,u2,conv=conv,texp=texp,frac=frac,sec=sec,new=new)
    #ts *= 24*60
    #durguess = calcT14(p0,b,aR,P,ecc,w,sec)
    durguess = ((fs < 1).sum()/float(len(fs)) * (ts[-1] - ts[0]))/2.
    #if sec:
    #    depguess = frac
    #else:
    #    depguess = frac*p0**2
    depguess = 1-fs.min()
    pars = array([durguess,depguess,5.,0.])
    if not use_fortran:
        dur,dep,slope,epoch = fitprotopapas(ts,fs,pars)
    else:
        dur,dep,slope,epoch = fit_pp_fortran(ts,fs,pars)
    if dur==-1 and dep==-1:
        raise NoFitError
    return dur,dep,slope


# Computes Hasting's polynomial approximation for the complete
# elliptic integral of the first (ek) and second (kk) kind
def ellke(k):
    m1=1.-k**2
    logm1 = log(m1)

    a1=0.44325141463
    a2=0.06260601220
    a3=0.04757383546
    a4=0.01736506451
    b1=0.24998368310
    b2=0.09200180037
    b3=0.04069697526
    b4=0.00526449639
    ee1=1.+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ee2=m1*(b1+m1*(b2+m1*(b3+m1*b4)))*(-logm1)
    ek = ee1+ee2
        
    a0=1.38629436112
    a1=0.09666344259
    a2=0.03590092383
    a3=0.03742563713
    a4=0.01451196212
    b0=0.5
    b1=0.12498593597
    b2=0.06880248576
    b3=0.03328355346
    b4=0.00441787012
    ek1=a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ek2=(b0+m1*(b1+m1*(b2+m1*(b3+m1*b4))))*logm1
    kk = ek1-ek2
    
    return [ek,kk]

# Computes the complete elliptical integral of the third kind using
# the algorithm of Bulirsch (1965):
def ellpic_bulirsch(n,k):
    kc=sqrt(1.-k**2); p=n+1.
    if(p.min() < 0.):
        print 'Negative p'
    m0=1.; c=1.; p=sqrt(p); d=1./p; e=kc
    while 1:
        f = c; c = d/p+c; g = e/p; d = 2.*(f*g+d)
        p = g + p; g = m0; m0 = kc + m0
        if (absolute(1.-kc/g)).max() > 1.e-8:
            kc = 2*sqrt(e); e=kc*m0
        else:
            return 0.5*pi*(c*m0+d)/(m0*(m0+p))


#   Python translation of IDL code.
#   This routine computes the lightcurve for occultation of a
#   quadratically limb-darkened source without microlensing.  Please
#   cite Mandel & Agol (2002) and Eastman & Agol (2008) if you make use
#   of this routine in your research.  Please report errors or bugs to
#   jdeast@astronomy.ohio-state.edu
def occultquad(z,u1,u2,p0,return_components=False):
    z = atleast_1d(z)
    nz = size(z)
    lambdad = zeros(nz)
    etad = zeros(nz)
    lambdae = zeros(nz)
    omega=1.-u1/3.-u2/6.

    ## tolerance for double precision equalities
    ## special case integrations
    tol = 1e-14

    p = absolute(p0)
    
    z = where(absolute(p-z) < tol,p,z)
    z = where(absolute((p-1)-z) < tol,p-1.,z)
    z = where(absolute((1-p)-z) < tol,1.-p,z)
    z = where(z < tol,0.,z)
               
    x1=(p-z)**2.
    x2=(p+z)**2.
    x3=p**2.-z**2.
    

    def finish(p,z,u1,u2,lambdae,lambdad,etad):
        omega = 1. - u1/3. - u2/6.
        #avoid Lutz-Kelker bias
        if p0 > 0:
            #limb darkened flux
            muo1 = 1 - ((1-u1-2*u2)*lambdae+(u1+2*u2)*(lambdad+2./3*(p > z)) + u2*etad)/omega
            #uniform disk
            mu0 = 1 - lambdae
        else:
            #limb darkened flux
            muo1 = 1 + ((1-u1-2*u2)*lambdae+(u1+2*u2)*(lambdad+2./3*(p > z)) + u2*etad)/omega
            #uniform disk
            mu0 = 1 + lambdae
        if return_components:
            return muo1,(mu0,lambdad,etad)
        else:
            return muo1



    ## trivial case of no planet
    if p <= 0.:
        return finish(p,z,u1,u2,lambdae,lambdad,etad)

    ## Case 1 - the star is unocculted:
    ## only consider points with z lt 1+p
    notusedyet = where( z < (1. + p) )[0]
    if size(notusedyet) == 0:
        return finish(p,z,u1,u2,lambdae,lambdad,etad)

    # Case 11 - the  source is completely occulted:
    if p >= 1.:
        cond = z[notusedyet] <= p-1.
        occulted = where(cond)#,complement=notused2)
        notused2 = where(~cond)
        #occulted = where(z[notusedyet] <= p-1.)#,complement=notused2)
        if size(occulted) != 0:
            ndxuse = notusedyet[occulted]
            etad[ndxuse] = 0.5 # corrected typo in paper
            lambdae[ndxuse] = 1.
            # lambdad = 0 already
            #notused2 = where(z[notusedyet] > p-1)
            if size(notused2) == 0:
                return finish(p,z,u1,u2,lambdae,lambdad,etad)
            notusedyet = notusedyet[notused2]
                
    # Case 2, 7, 8 - ingress/egress (uniform disk only)
    inegressuni = where((z[notusedyet] >= absolute(1.-p)) & (z[notusedyet] < 1.+p))
    if size(inegressuni) != 0:
        ndxuse = notusedyet[inegressuni]
        tmp = (1.-p**2.+z[ndxuse]**2.)/2./z[ndxuse]
        tmp = where(tmp > 1.,1.,tmp)
        tmp = where(tmp < -1.,-1.,tmp)
        kap1 = arccos(tmp)
        tmp = (p**2.+z[ndxuse]**2-1.)/2./p/z[ndxuse]
        tmp = where(tmp > 1.,1.,tmp)
        tmp = where(tmp < -1.,-1.,tmp)
        kap0 = arccos(tmp)
        tmp = 4.*z[ndxuse]**2-(1.+z[ndxuse]**2-p**2)**2
        tmp = where(tmp < 0,0,tmp)
        lambdae[ndxuse] = (p**2*kap0+kap1 - 0.5*sqrt(tmp))/pi
        # eta_1
        etad[ndxuse] = 1./2./pi*(kap1+p**2*(p**2+2.*z[ndxuse]**2)*kap0- \
           (1.+5.*p**2+z[ndxuse]**2)/4.*sqrt((1.-x1[ndxuse])*(x2[ndxuse]-1.)))
    
    # Case 5, 6, 7 - the edge of planet lies at origin of star
    cond = z[notusedyet] == p
    ocltor = where(cond)#, complement=notused3)
    notused3 = where(~cond)
    #ocltor = where(z[notusedyet] == p)#, complement=notused3)
    t = where(z[notusedyet] == p)
    if size(ocltor) != 0:
        ndxuse = notusedyet[ocltor] 
        if p < 0.5:
            # Case 5
            q=2.*p  # corrected typo in paper (2k -> 2p)
            Ek,Kk = ellke(q)
            # lambda_4
            lambdad[ndxuse] = 1./3.+2./9./pi*(4.*(2.*p**2-1.)*Ek+\
                                              (1.-4.*p**2)*Kk)
            # eta_2
            etad[ndxuse] = p**2/2.*(p**2+2.*z[ndxuse]**2)        
            lambdae[ndxuse] = p**2 # uniform disk
        elif p > 0.5:
            # Case 7
            q=0.5/p # corrected typo in paper (1/2k -> 1/2p)
            Ek,Kk = ellke(q)
            # lambda_3
            lambdad[ndxuse] = 1./3.+16.*p/9./pi*(2.*p**2-1.)*Ek-\
                              (32.*p**4-20.*p**2+3.)/9./pi/p*Kk
            # etad = eta_1 already
        else:
            # Case 6
            lambdad[ndxuse] = 1./3.-4./pi/9.
            etad[ndxuse] = 3./32.
        #notused3 = where(z[notusedyet] != p)
        if size(notused3) == 0:
            return finish(p,z,u1,u2,lambdae,lambdad,etad)
        notusedyet = notusedyet[notused3]

    # Case 2, Case 8 - ingress/egress (with limb darkening)
    cond = ((z[notusedyet] > 0.5+absolute(p-0.5)) & \
                       (z[notusedyet] < 1.+p))  | \
                      ( (p > 0.5) & (z[notusedyet] > absolute(1.-p)) & \
                        (z[notusedyet] < p))
    inegress = where(cond)
    notused4 = where(~cond)
    #inegress = where( ((z[notusedyet] > 0.5+abs(p-0.5)) & \
        #(z[notusedyet] < 1.+p))  | \
        #( (p > 0.5) & (z[notusedyet] > abs(1.-p)) & \
        #(z[notusedyet] < p)) )#, complement=notused4)
    if size(inegress) != 0:

        ndxuse = notusedyet[inegress]
        q=sqrt((1.-x1[ndxuse])/(x2[ndxuse]-x1[ndxuse]))
        Ek,Kk = ellke(q)
        n=1./x1[ndxuse]-1.

        # lambda_1:
        lambdad[ndxuse]=2./9./pi/sqrt(x2[ndxuse]-x1[ndxuse])*\
                         (((1.-x2[ndxuse])*(2.*x2[ndxuse]+x1[ndxuse]-3.)-\
                           3.*x3[ndxuse]*(x2[ndxuse]-2.))*Kk+(x2[ndxuse]-\
                           x1[ndxuse])*(z[ndxuse]**2+7.*p**2-4.)*Ek-\
                          3.*x3[ndxuse]/x1[ndxuse]*ellpic_bulirsch(n,q))

        #notused4 = where( ( (z[notusedyet] <= 0.5+abs(p-0.5)) | \
        #                    (z[notusedyet] >= 1.+p) ) & ( (p <= 0.5) | \
        #                    (z[notusedyet] <= abs(1.-p)) | \
        #                    (z[notusedyet] >= p) ))
        if size(notused4) == 0:
            return finish(p,z,u1,u2,lambdae,lambdad,etad)
        notusedyet = notusedyet[notused4]

    # Case 3, 4, 9, 10 - planet completely inside star
    if p < 1.:
        cond = z[notusedyet] <= (1.-p)
        inside = where(cond)
        notused5 = where(~cond)
        #inside = where(z[notusedyet] <= (1.-p))#, complement=notused5)
        if size(inside) != 0:
            ndxuse = notusedyet[inside]

            ## eta_2
            etad[ndxuse] = p**2/2.*(p**2+2.*z[ndxuse]**2)

            ## uniform disk
            lambdae[ndxuse] = p**2

            ## Case 4 - edge of planet hits edge of star
            edge = where(z[ndxuse] == 1.-p)#, complement=notused6)
            if size(edge[0]) != 0:
                ## lambda_5
                lambdad[ndxuse[edge]] = 2./3./pi*arccos(1.-2.*p)-\
                                      4./9./pi*sqrt(p*(1.-p))*(3.+2.*p-8.*p**2)
                if p > 0.5:
                    lambdad[ndxuse[edge]] -= 2./3.
                notused6 = where(z[ndxuse] != 1.-p)
                if size(notused6) == 0:
                    return finish(p,z,u1,u2,lambdae,lambdad,etad)
                ndxuse = ndxuse[notused6[0]]

            ## Case 10 - origin of planet hits origin of star
            origin = where(z[ndxuse] == 0)#, complement=notused7)
            if size(origin) != 0:
                ## lambda_6
                lambdad[ndxuse[origin]] = -2./3.*(1.-p**2)**1.5
                notused7 = where(z[ndxuse] != 0)
                if size(notused7) == 0:
                    return finish(p,z,u1,u2,lambdae,lambdad,etad)
                ndxuse = ndxuse[notused7[0]]
   
            q=sqrt((x2[ndxuse]-x1[ndxuse])/(1.-x1[ndxuse]))
            n=x2[ndxuse]/x1[ndxuse]-1.
            Ek,Kk = ellke(q)    

            ## Case 3, Case 9 - anywhere in between
            ## lambda_2
            lambdad[ndxuse] = 2./9./pi/sqrt(1.-x1[ndxuse])*\
                              ((1.-5.*z[ndxuse]**2+p**2+x3[ndxuse]**2)*Kk+\
                               (1.-x1[ndxuse])*(z[ndxuse]**2+7.*p**2-4.)*Ek-\
                               3.*x3[ndxuse]/x1[ndxuse]*ellpic_bulirsch(n,q))

        ## if there are still unused elements, there's a bug in the code
        ## (please report it)
        #notused5 = where(z[notusedyet] > (1.-p))
        if notused5[0] != 0:
            print "ERROR: the following values of z didn't fit into a case:"

        return finish(p,z,u1,u2,lambdae,lambdad,etad)

def protopapas(t,p,const=1,noper=True,noepoch=True):
    if size(p)==3:
        eta,theta,c = p #dur,depth,slope
        T = eta*1000
        tau = 0
    elif size(p)==4:
        eta,theta,c,tau = p
        T=eta*1000
    else:
        T,eta,theta,tau,c = p
    const = const-theta
    tprime = T*sin(pi*(t-tau)/T)/pi/eta
    return const + 0.5*theta*(2 - tanh(c*(tprime + 0.5)) + tanh(c*(tprime - 0.5)))
    
def traptransit(ts,p):
    return tru.traptransit(ts,p)

def fit_traptransit(ts,fs,p0,debug=False):
    pfit,success = scipy.optimize.leastsq(tru.traptransit_resid,p0,args=(ts,fs))
    if success not in [1,2,3,4]:
        raise NoFitError
    if debug:
        print 'success = %i' % success
    return pfit

def fitprotopapas(ts,fs,p0):
    pfit,success = scipy.optimize.leastsq(tru.protopapas_resid,p0,args=(ts,fs))
    return pfit

def fitprotopapas_old(ts,fs,p0,const=1,leastsq=True,noper=True):
    """Returns fit parameters: dur, depth, slope """
    if leastsq:
        def errfn(p,t,f):
            return protopapas(t,p,const=const,noper=noper) - f
        pfit,success = scipy.optimize.leastsq(errfn,p0,args=(ts,fs))
    else:
        def errfn(p,t,f):
            err = protopapas(t,p,const=const,noper=noper) - f
            if size(err)>1:
                return (err.sum())**2
            else:
                return err**2
        pfit = scipy.optimize.fmin(errfn,p0,args=(ts,fs),disp=0)
    return pfit

def protopapasMCMCmodel(ts,fs,dfs=1e-5,per=10,epoch=0,maxdur=1000,depthguess=0.001): #make dfs reasonable!
    per *= 60*24
    duration = pm.Uniform('duration',lower=0,upper=0.5)
    depth = pm.Uniform('depth',lower=depthguess*0.5,upper=depthguess*2)
    logslope = pm.Uniform('logslope',lower=-1,upper=2)
    epoch = pm.Uniform('epoch',lower=-0.1,upper=0.1)
    #not sure why the line below didn't work
    #flux = pm.Normal('flux',observed=True,value=fs,mu=protopapas(ts,(duration.value,depth.value,slope.value)),tau=dfs)

    @pm.observed(dtype=float)
    def flux(value=fs,duration=duration,depth=depth,logslope=logslope,epoch=epoch):
        modelfs = protopapas(ts,(duration,depth,10**logslope,epoch))
        errs = -0.5*(fs-modelfs)**2/(dfs**2) - 0.5*log(2*pi*dfs)
        return errs.sum()
    return locals()

def protopapasMCMC_old(ts,fs,dfs=1e-5,per=10,epoch=0,maxdur=0.5,depthguess=None,niter=5e4,nburn=5e3,thin=100,verbose=False,add_noise=False):
    if depthguess is None:
        depthguess = 1-fs.min()  #crude approximation for now
    model = protopapasMCMCmodel(ts,fs,dfs,per,epoch,maxdur,depthguess)
    M = pm.MCMC(model)
    M.sample(iter=niter,burn=nburn,thin=thin,verbose=verbose)
    return M
    
 
class traptransit_model(object):
    def __init__(self,ts,fs,sigs=1e-4,maxslope=30):
        self.n = size(ts)
        if size(sigs)==1:
            sigs = ones(self.n)*sigs
        self.ts = ts
        self.fs = fs
        self.sigs = sigs
        self.maxslope = maxslope
        
    def __call__(self,pars):
        pars = array(pars)
        #return tru.traptransit_lhood(pars,self.ts,self.fs,self.sigs)
        return traptransit_lhood(pars,self.ts,self.fs,self.sigs,maxslope=self.maxslope)

def traptransit_lhood(pars,ts,fs,sigs,maxslope=30):
    if pars[0] < 0 or pars[1] < 0 or pars[2] < 2 or pars[2] > maxslope:
        return -inf
    resid = tru.traptransit_resid(pars,ts,fs)
    return (-0.5*resid**2/sigs**2).sum()

def traptransit_MCMC(ts,fs,dfs=1e-5,nwalkers=200,nburn=300,niter=1000,threads=1,p0=[0.1,0.1,3,0],return_sampler=False,verbose=False,maxslope=30):
    model = traptransit_model(ts,fs,dfs,maxslope=maxslope)
    sampler = emcee.EnsembleSampler(nwalkers,4,model,threads=threads)
    T0 = p0[0]*(1+rand.normal(size=nwalkers)*0.1)
    d0 = p0[1]*(1+rand.normal(size=nwalkers)*0.1)
    slope0 = p0[2]*(1+rand.normal(size=nwalkers)*0.1)
    ep0 = p0[3]+rand.normal(size=nwalkers)*0.0001

    p0 = array([T0,d0,slope0,ep0]).T

    pos, prob, state = sampler.run_mcmc(p0, nburn)
    sampler.reset()
    sampler.run_mcmc(pos, niter, rstate0=state)
    if return_sampler:
        return sampler
    else:
        return sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3]


##### Custom Exceptions

class NoEclipseError(Exception):
    pass

class NoFitError(Exception):
    pass

class EmptyPopulationError(Exception):
    pass

class NotImplementedError(Exception):
    pass

class AllWithinRocheError(Exception):
    pass
