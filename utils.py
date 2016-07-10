import scipy
from numpy import *
from scipy.integrate import *
from consts import *
from numpy.random import randint,random,normal,shuffle
from scipy.stats import gaussian_kde
#from pickleutils import *
try:
    from astropysics.coords import ICRSCoordinates,GalacticCoordinates,FK5Coordinates
except ImportError:
    pass
import numpy as np
import pylab as p
from scipy.optimize import leastsq
from scipy.interpolate import UnivariateSpline as interpolate

def iclosest(arr,val):
    ind = ((arr-val).abs()).argmin()
    if size(ind) > 1:
        ind = ind[0]
    return ind

def gr2B(g,r):
    return gr2V(g,r) + 1.04*(g-r) + 0.19

def gr2V(g,r):
    return r + 0.44*(g-r)-0.02

def keckSNR(vmag,t):
    # mV=8, t=88s, SNR=188
    return 188*sqrt(2.51**(8-vmag)*(t/88.))

def kecktexp(vmag,snr):
    return 88*2.51**(vmag-8)*(snr/188.)**2


def deriv(f,c,dx=0.0001):
    """
    deriv(f,c,dx)  --> float
    
    Returns f'(x), computed as a symmetric difference quotient.
    """
    return (f(c+dx)-f(c-dx))/(2*dx)

def fuzzyequals(a,b,tol=0.0001):
    return abs(a-b) < tol

def newton(f,c,tol=0.0001,restrict=None):
    """
    newton(f,c) --> float
    
    Returns the x closest to c such that f(x) = 0
    """
    #print c
    if restrict:
        lo,hi = restrict
        if c < lo or c > hi:
            print c
            c = random*(hi-lo)+lo

    if fuzzyequals(f(c),0,tol):
        return c
    else:
        try:
            return newton(f,c-f(c)/deriv(f,c,tol),tol,restrict)
        except:
            return None

def trapznd(arr,*axes):
    n = len(arr.shape)
    if len(axes) != n:
        raise ValueError('must provide same number of axes as number of dimensions!')
    val = trapz(arr,axes[0],axis=0)
    for i in arange(1,n):
        val = trapz(val,axes[i],axis=0)
    return val
        
    

def epkernel(u):
    x = atleast_1d(u)
    y = 3./4*(1-x*x)
    y[where((x>1) | (x < -1))] = 0
    return y

def gausskernel(u):
    return 1/sqrt(2*pi)*exp(-0.5*u*u)

def tricubekernel(u):
    x = atleast_1d(u)
    y = 35./32*(1-x*x)**3
    y[where((x > 1) | (x < -1))] = 0
    return y

def kernelfn(kernel='tricube'):
    if kernel=='ep':
        #def fn(u):
        #    x = atleast_1d(u)
        #    y = 3./4*(1-x*x)
        #    y[where((x>1) | (x<-1))] = 0
        #    return y
        #return fn
        return epkernel

    elif kernel=='gauss':
        #return lambda x: 1/sqrt(2*pi)*exp(-0.5*x*x)
        return gausskernel

    elif kernel=='tricube':
        #def fn(u):
        #    x = atleast_1d(u)
        #    y = 35/32.*(1-x*x)**3
        #    y[where((x>1) | (x<-1))] = 0
        #    return y
        #return fn
        return tricubekernel

def kerneldraw(size=1,kernel='tricube'):
    if kernel=='tricube':
        fn = lambda x: 1./2 + 35./32*x - 35./32*x**3 + 21./32*x**5 - 5./32*x**7
        u = random(size=size)
        rets = zeros(size)
        for i in arange(size):
            f = lambda x: u[i]-fn(x)
            rets[i] = newton(f,0,restrict=(-1,1))
        return rets

class composite_kde(object):
    def __init__(self,kde1,kde2,operation='add'):
        self.operation = operation
        if self.operation == 'add':
            self.comp1 = kde1
            self.comp2 = kde2
            self.norm = self.comp1.norm + self.comp2.norm
        prop = self.comp1.properties.copy()
        prop.update(self.comp2.properties)
        self.properties = prop

    def __call__(self,x):
        if self.operation == 'add':
            return (self.comp1(x) + self.comp2(x))/self.norm

    def integrate_box(self,lo,hi,forcequad=False):
        return self.comp1.integrate_box(lo,hi,forcequad=forcequad) + self.comp2.integrate_box(lo,hi,forcequad=forcequad)

    def resample(self,size=1):
        f1 = float(self.comp1.norm)/(self.comp1.norm+self.comp2.norm)
        n1 = sum(random(size=size) < f1)
        n2 = size-n1
        samples = concatenate((self.comp1.resample(n1),self.comp2.resample(n2)))
        shuffle(samples)
        return samples

class kde(object):
    

    def __init__(self,dataset,kernel='tricube',adaptive=True,k=None,lo=None,hi=None,\
                     fast=None,norm=None,bandwidth=None,weights=None):
        self.dataset = atleast_1d(dataset)
        self.weights = weights
        self.n = size(dataset)
        self.kernel = kernelfn(kernel)
        self.kernelname = kernel
        self.bandwidth = bandwidth
        if k:
            self.k = k
        else:
            self.k = self.n/4

        if not norm:
            self.norm=1.
        else:
            self.norm=norm


        self.adaptive = adaptive
        self.fast = fast
        if adaptive:
            if fast==None:
                fast = self.n < 5001

            if fast:
                d1,d2 = meshgrid(self.dataset,self.dataset)
                diff = abs(d1-d2)
                diffsort = sort(diff,axis=0)
                self.h = diffsort[self.k,:]

        ##Attempt to handle larger datasets more easily:
            else:
                sortinds = argsort(self.dataset)
                x = self.dataset[sortinds]
                h = zeros(len(x))
                for i in arange(len(x)):
                    lo = i - self.k
                    hi = i + self.k + 1
                    if lo < 0:
                        lo = 0
                    if hi > len(x):
                        hi = len(x)
                    diffs = abs(x[lo:hi]-x[i])
                    h[sortinds[i]] = sort(diffs)[self.k]
                self.h = h
        else:
            self.gauss_kde = gaussian_kde(self.dataset)
            
        self.properties=dict()

        self.lo = lo
        self.hi = hi

    def shifted(self,x):
        new = kde(self.dataset+x,self.kernel,self.adaptive,self.k,self.lo,self.hi,self.fast,self.norm)
        return new

    def renorm(self,norm):
        self.norm = norm

    def evaluate(self,points):
        if not self.adaptive:
            return self.gauss_kde(points)*self.norm
        points = atleast_1d(points).astype(self.dataset.dtype)
        k = self.k

        npts = size(points)

        h = self.h
        
        X,Y = meshgrid(self.dataset,points)
        H = resize(h,(npts,self.n))

        U = (X-Y)/H.astype(float)

        result = 1./self.n*1./H*self.kernel(U)
        return sum(result,axis=1)*self.norm
            
    __call__ = evaluate
            
    def __imul__(self,factor):
        self.renorm(factor)
        return self

    def __add__(self,other):
        return composite_kde(self,other)

    __radd__ = __add__

    def integrate_box(self,low,high,npts=500,forcequad=False):
        if not self.adaptive and not forcequad:
            return self.gauss_kde.integrate_box_1d(low,high)*self.norm
        pts = linspace(low,high,npts)
        return quad(self.evaluate,low,high)[0]

    def draw(self,size=None):
        return self.resample(size)

    def resample(self,size=None):
        size=int(size)
        if not self.adaptive:
            return squeeze(self.gauss_kde.resample(size=size))
        if size is None:
            size = self.n
        indices = randint(0,self.n,size=size)
        means = self.dataset[indices]
        h = self.h[indices]
        fuzz = kerneldraw(size,self.kernelname)*h
        return squeeze(means + fuzz)

class generalpdf(object):
    def __add__(self,other):
        return compositepdf(self,other)

    __radd__ = __add__

    def __mul__(self,scale):
        return scaledpdf(self,scale)

    __rmul__ = __mul__

    def renorm(self,factor=None):
        self.norm *= factor

    def __imul__(self,factor):
        self.renorm(factor)
        return self

class compositepdf(generalpdf):
    def __init__(self,comp1,comp2):
        self.comp1 = comp1
        self.comp2 = comp2
        self.norm = self.comp1.norm + self.comp2.norm

    def __call__(self,x):
        return self.comp1(x) + self.comp2(x)

    def draw(self,size=1):
        f1 = float(self.comp1.norm)/(self.comp1.norm+self.comp2.norm)
        n1 = sum(random(size=size) < f1)
        n2 = size-n1
        samples = concatenate((self.comp1.draw(n1),self.comp2.draw(n2)))
        shuffle(samples)
        return samples
        

class scaledpdf(generalpdf):
    def __init__(self,pdf,scale):
        self.pdf = pdf
        self.scale = scale
        self.norm = scale * pdf.norm

    def __call__(self,x):
        return self.scale * self.pdf(x)

    def draw(self,size=1):
        return self.pdf.draw(size)

class powerlaw(generalpdf):
    def __init__(self,alpha,xmin=0.5,xmax=10,norm=1.0):
        self.alpha = alpha
        self.xmin = xmin
        self.xmax = xmax
        self.norm = norm
        self.plnorm = powerlawnorm(alpha,xmin,xmax)

    def __call__(self,inpx):
        x = atleast_1d(inpx)
        y = self.norm*self.plnorm*x**self.alpha
        y[where((x < self.xmin) | (x > self.xmax))] = 0
        return y

    def draw(self,size=1):
        u = random(size=size)
        a = self.alpha
        if a==-1:
            a = -1.00001 #hack to avoid -1...
        C = self.plnorm
        return ((u*(a+1))/C + self.xmin**(a+1))**(1./(a+1))

class polynomial(generalpdf):
    def __init__(self,c,xmin=0.5,xmax=20,norm=1.0):
        self.c = c
        self.xmin = xmin
        self.xmax = xmax
        self.norm = norm

    def __call__(self,x):
        return np.polyval(self.c,x)

        

class triple_powerlaw(generalpdf):
    def __init__(self,alpha1,alpha2,alpha3,xbreak1,xbreak2,xmin=0.5,xmax=20,norm=1.0):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.xbreak1 = xbreak1
        self.xbreak2 = xbreak2
        self.xmin = xmin
        self.xmax = xmax
        self.norm = norm

        x1 = xbreak1; x2 = xbreak2
        a1 = alpha1; a2 = alpha2; a3 = alpha3
        if a1==-1:
            a1 = -1.000001
        if a2==-1:
            a2 = -1.000001
        if a3==-1:
            a3 = -1.000001

        self.A = (self.norm)/((x1**(a1 + 1) - xmin**(a1 + 1))/(a1 + 1) +
                              (x1**(a1 - a2)*(x2**(a2 +1) - x1**(a2+1)))/(a2 + 1) + 
                              (x1**(a1 - a2)*(x2**(a2 - a3))*(xmax**(a3 + 1) - 
                                                              x2**(a3 + 1)))/(a3 + 1))

        self.B = self.A * x1**(a1 - a2)
        self.C = self.B * x2**(a2 - a3)



        self.f1 = quad(self,xmin,x1)[0]/self.norm
        self.f2 = quad(self,x1,x2)[0]/self.norm
        self.f3 = quad(self,x2,xmax)[0]/self.norm

        self.plaw1 = powerlaw(alpha1,xmin,xbreak1)
        self.plaw2 = powerlaw(alpha2,xbreak1,xbreak2)
        self.plaw3 = powerlaw(alpha3,xbreak2,xmax)

    def __call__(self,inpx):
        x = atleast_1d(inpx)
        lo = (x < self.xbreak1)
        mid = (x >= self.xbreak1) & (x < self.xbreak2)
        hi = (x >= self.xbreak2)
        x1 = self.xbreak1; x2 = self.xbreak2
        a1 = self.alpha1; a2 = self.alpha2; a3 = self.alpha3

        return (lo * self.A * x**self.alpha1 + 
                mid * self.B * x**self.alpha2 +
                hi * self.C * x**self.alpha3)

    def draw(self,size=1):
        u = random(size=size)
        lo = (u < self.f1)
        mid = (u >= self.f1) & (u < self.f2)
        hi = (u >= self.f2)
        return (self.plaw1.draw(size)*lo + 
                self.plaw2.draw(size)*mid +
                self.plaw3.draw(size)*hi)
    
        

class broken_powerlaw(generalpdf):
    def __init__(self,alpha1,alpha2,xbreak,xmin=0.5,xmax=20,norm=1.0):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.xbreak = xbreak
        self.xmin = xmin
        self.xmax = xmax
        self.norm = norm

        def fn(x):
            if x < xbreak:
                return (x/xbreak)**alpha1
            else:
                return (x/xbreak)**alpha2

        self.plawnorm = quad(fn,xmin,xmax)[0]/self.norm

        self.f1 = quad(self,xmin,xbreak)[0]/self.norm
        self.f2 = quad(self,xbreak,xmax)[0]/self.norm

        self.plaw1 = powerlaw(alpha1,xmin,xbreak)
        self.plaw2 = powerlaw(alpha2,xbreak,xmax)

    def __call__(self,inpx):
        x = atleast_1d(inpx)
        lo = (x < self.xbreak)
        hi = (x >= self.xbreak)
        xb = self.xbreak
        return 1./self.plawnorm * (lo*(x/xb)**self.alpha1 + hi*(x/xb)**self.alpha2)

    def draw(self,size=1):
        u = random(size=size)
        lo = (u < self.f1)
        hi = (u >= self.f1)
        return self.plaw1.draw(size)*lo + self.plaw2.draw(size)*hi

class lognorm(generalpdf):
    def __init__(self,mu,sig):
        self.mu = mu*log(10)
        self.sig = sig*log(10)
        self.norm = 1.

    def __call__(self,inpx):
        mu,sig = (self.mu,self.sig)
        x = atleast_1d(inpx)
        return 1/(x*sig*sqrt(2*pi))*exp(-(log(x)-mu)**2/(2*sig*sig))

    def draw(self,size=1):
        rand = normal(size=size) * self.sig + self.mu
        return exp(rand)

class uniform(generalpdf):
    def __init__(self,xmin,xmax):
        self.xmin=xmin
        self.xmax=xmax
        self.norm=1.0

    def __call__(self,inpx):
        x = atleast_1d(inpx)
        return x*1./(xmax-xmin)

    def draw(self,size=1):
        rand = random(size)
        return rand*(xmax-xmin)+xmin

class gaussian(generalpdf):
    def __init__(self,mu,sig,norm=1):
        self.mu = mu
        self.sig = sig
        self.norm = norm

    def __call__(self,inpx):
        x = atleast_1d(inpx)
        return self.norm*1/sqrt(2*pi*self.sig**2)*exp(-(x-self.mu)**2/(2*self.sig**2))

    #needs draw() written!

#class uniform_gausscutoffhi(generalpdf):
#    def __init__(self,xmin,xmax,sig=0.1):
#        self.xmin=xmin
#        self.xmax=xmax
#        self.sig=sig
#        self.norm=1.0
        

#    def __call__(self,inpx):
#        x = atleast_1d(inpx)
        


def powerlawfn(alpha,xmin=.01,xmax=50,normed=True):
#    if alpha == -1:
#        C = 1/log(xmax/xmin)
#    else:
#        C = (1+alpha)/(xmax**(1+alpha)-xmin**(1+alpha))
#    return C*x**(alpha)
    if normed:
        C = powerlawnorm(alpha,xmin,xmax)
    else:
        C=1
    def fn(inpx):
        x = atleast_1d(inpx)
        y = C*x**(alpha)
        y[where((x < xmin) | (x > xmax))] = 0
        return y
    return fn

def powerlawnorm(alpha,xmin,xmax):
    if size(alpha)==1:
        if alpha == -1:
            C = 1/log(xmax/xmin)
        else:
            C = (1+alpha)/(xmax**(1+alpha)-xmin**(1+alpha))
    else:
        C = zeros(size(alpha))
        w = where(alpha==-1)
        if len(w[0]>0):
            C[w] = 1./log(xmax/xmin)*ones(len(w[0]))
            nw = where(alpha != -1)
            C[nw] = (1+alpha[nw])/(xmax**(1+alpha[nw])-xmin**(1+alpha[nw]))
        else:
            C = (1+alpha)/(xmax**(1+alpha)-xmin**(1+alpha))
    return C

def eq2gal(r,d):
    eq = FK5Coordinates(r,d)
    gal = eq.convert(GalacticCoordinates)
    return gal.l.degrees,gal.b.degrees
    #A = cos(d*pi/180)*cos((r-282.25)*pi/180)
    #B = sin(d*pi/180)*sin(62.6*pi/180) + cos(d*pi/180)*sin((r-282.25)*pi/180)*cos(62.6*pi/180)
    #C = sin(d*pi/180)*cos(62.6*pi/180) - cos(d*pi/180)*sin((r-282.25)*pi/180)*sin(62.6*pi/180)
    #b = arcsin(C)
    #l = arccos(A/cos(b))*180/pi + 33
    #b = b*180/pi
    #return l,b

def append_field(rec,name,arr,dt=None):
    arr = asarray(arr)
    if dt is None:
        dt = arr.dtype
    newdtype = dtype(rec.dtype.descr + [(name,dt)])
    newrec = empty(rec.shape,dtype=newdtype)
    for field in rec.dtype.fields:
        newrec[field] = rec[field]
    newrec[name] = arr
    return np.core.records.array(newrec)

def expfunc(p,x):
    return p[2] + p[0]*exp(-x/p[1])

def fitexp(x,y,p0=[1,10,0.03]):
    errfunc = lambda p,x,y: expfunc(p,x)-y
    p1,success = leastsq(errfunc,p0[:],args=(x,y))
    return p1

def save(obj,filename):
    f = open(filename,'wb')
    pickle.dump(obj,f)
    f.close()

def load(filename):
    f = open(filename,'rb')
    obj = pickle.load(f)
    f.close()
    return obj

def dict2arr(indict):
    keys = indict.keys()
    keysarr = array(keys)
    maxind = keysarr.max()
    arr = zeros(maxind+1)
    for key in keysarr:
        arr[key] = indict[key]
    return arr


def repeats(arr,return_index=False,return_counts=False):
    #add "return_counts" something....i.e. saying how many there are of each
    already = dict()
    ininds=dict()
    n=0
    inds=[]
    i=0
    for el in arr:
        if el in already:
            already[el]+=1
            if not el in ininds:
                inds.append(i)
                n+=1
                ininds[el]=1
            else:
                ininds[el]+=1
        else:
            already[el] = 1
        i+=1
    if return_index:
        return n,inds
    if return_counts:
        nreps = dict2arr(already)
        return n,inds,nreps
    else:
        return n


def confreg(x,Lin,conf=0.68,tol=0.005):
    L = Lin/trapz(Lin,x)  #normalize likelihood
    imax = argmax(L)
    if imax==0:
        imax=1
    if imax==len(L)-1:
        imax = len(L)-2

    Lmax = L[imax]

    xlo = x[0:imax]
    xhi = x[imax:]
    Llo = L[0:imax]
    Lhi = L[imax:]


    prob = 0
    level=Lmax
    dL = Lmax/1000.
    while prob < conf:
        level -= dL
        i1 = argmin(abs(Llo-level))
        i2 = argmin(abs(Lhi-level))+imax
        prob = trapz(L[i1:i2],x[i1:i2])
        if level < 0:
            print 'error in calculating confidence interval: only reached %.2f\% of probability' % prob
            return nan,nan

    return x[i1],x[i2]


def pctile(x,q):
    q /= 100.
    s = sort(x)
    n = size(x)
    i = s[int(n*q)]
    return x[i]

def qstd(x,quant=0.05,top=False,bottom=False):
    """returns std, ignoring outer 'quant' pctiles
    """
    s = sort(x)
    n = size(x)
    lo = s[int(n*quant)]
    hi = s[int(n*(1-quant))]
    if top:
        w = where(x>=lo)
    elif bottom:
        w = where(x<=hi)
    else:
        w = where((x>=lo)&(x<=hi))
    return std(x[w])

def meshgrid3d(x,y,z):
    gridx = x + 0*y[:,newaxis] + 0*z[:,newaxis,newaxis]
    gridy = 0*x + y[:,newaxis] + 0*z[:,newaxis,newaxis]
    gridz = 0*x + 0*y[:,newaxis] + z[:,newaxis,newaxis]

    return gridx,gridy,gridz


### classes defining statitistical distributions

class Distribution(object):
    def __init__(self,pdf,cdf=None,name='',minval=-np.inf,maxval=np.inf,norm=None,
                 no_cdf=False,cdf_pts=100):
        self.name = name
        self.pdf = pdf
        self.cdf = cdf
        self.minval = minval
        self.maxval = maxval

        if not hasattr(self,'Ndists'):
            self.Ndists = 1

        if norm is None:
            self.norm = quad(pdf,minval,maxval,full_output=1)[0]
        else:
            self.norm = norm

        if cdf is None and not no_cdf and minval != -np.inf and maxval != np.inf:
            pts = np.linspace(minval,maxval,cdf_pts)
            pdfgrid = self(pts)
            cdfgrid = pdfgrid.cumsum()/pdfgrid.cumsum().max()
            cdf_fn = interpolate(pts,cdfgrid,s=0)
            def cdf(x):
                x = np.atleast_1d(x)
                y = np.atleast_1d(cdf_fn(x))
                y[np.where(x < self.minval)] = 0
                y[np.where(x > self.maxval)] = 1
                return y
            self.cdf = cdf

    def pctile(self,pct,res=1000):
        grid = np.arange(self.minval,self.maxval,(self.maxval-self.minval)/float(res))
        return grid[np.argmin(np.absolute(pct-self.cdf(grid)))]

    def __add__(self,other):
        return Combined_Distribution((self,other))

    def __radd__(self,other):
        return self.__add__(other)

    def __call__(self,x):
        y = self.pdf(x)
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        w = np.where((x < self.minval) | (x > self.maxval))
        y[w] = 0
        return y/self.norm


    def plot(self,minval=None,maxval=None,fig=None,log=False,npts=500,**kwargs):
        if minval is None:
            minval = self.minval
        if maxval is None:
            maxval = self.maxval
        if maxval==np.inf or minval==-np.inf:
            raise ValueError('must have finite upper and lower bounds to plot. (set minval, maxval kws)')

        if log:
            xs = np.logspace(np.log10(minval),np.log10(maxval),npts)
        else:
            xs = np.linspace(minval,maxval,npts)

        plu.setfig(fig)
        plt.plot(xs,self(xs),**kwargs)
        plt.xlabel(self.name)
        plt.ylim(ymin=0)

    def resample(self,N,minval=None,maxval=None,log=False,res=1e4):
        if minval is None:
            minval = self.minval
        if maxval is None:
            maxval = self.maxval
        if maxval==np.inf or minval==-np.inf:
            raise ValueError('must have finite upper and lower bounds to resample. (set minval, maxval kws)')

        u = rand.random(size=N)
        if log:
            vals = np.logspace(log10(minval),log10(maxval),res)
        else:
            vals = np.linspace(minval,maxval,res)
            
        ys = self.cdf(vals)
        inds = np.digitize(u,ys)
        return vals[inds]

class DoubleGauss_Distribution(Distribution):
    def __init__(self,mu,siglo,sighi,**kwargs):
        self.mu = mu
        self.siglo = siglo
        self.sighi = sighi
        def pdf(x):
            x = np.atleast_1d(x)
            A = 1./(np.sqrt(2*np.pi)*(siglo+sighi)/2.)
            ylo = A*np.exp(-(x-mu)**2/(2*siglo**2))
            yhi = A*np.exp(-(x-mu)**2/(2*sighi**2))
            y = x*0
            wlo = np.where(x < mu)
            whi = np.where(x >= mu)
            y[wlo] = ylo[wlo]
            y[whi] = yhi[whi]
            return y

        if 'minval' not in kwargs:
            kwargs['minval'] = mu - 5*siglo
        if 'maxval' not in kwargs:
            kwargs['maxval'] = mu + 5*sighi

        Distribution.__init__(self,pdf,**kwargs)

    def __str__(self):
        return '%s = %.1f +%.1f -%.1f' % (self.name,self.mu,self.sighi,self.siglo)

    def resample(self,N,**kwargs):
        lovals = self.mu - np.absolute(rand.normal(size=N)*self.siglo)
        hivals = self.mu + np.absolute(rand.normal(size=N)*self.sighi)

        u = rand.random(size=N)
        whi = np.where(u < float(self.sighi)/(self.sighi + self.siglo))
        wlo = np.where(u >= float(self.sighi)/(self.sighi + self.siglo))

        vals = np.zeros(N)
        vals[whi] = hivals[whi]
        vals[wlo] = lovals[wlo]
        return vals
        
        return rand.normal(size=N)*self.sig + self.mu


class Gaussian_Distribution(Distribution):
    def __init__(self,mu,sig,**kwargs):
        self.mu = mu
        self.sig = sig
        def pdf(x):
            return 1./np.sqrt(2*np.pi*sig**2)*np.exp(-(x-mu)**2/(2*sig**2))

        if 'minval' not in kwargs:
            kwargs['minval'] = mu - 5*sig
        if 'maxval' not in kwargs:
            kwargs['maxval'] = mu + 5*sig

        Distribution.__init__(self,pdf,**kwargs)

    def __str__(self):
        return '%s = %.1f +/- %.1f' % (self.name,self.mu,self.sig)
        
    def resample(self,N,**kwargs):
        return rand.normal(size=N)*self.sig + self.mu

class KDE_Distribution(Distribution):
    def __init__(self,samples,**kwargs):
        self.samples = samples
        self.kde = gaussian_kde(samples)

        Distribution.__init__(self,self.kde,**kwargs)

    def __str__(self):
        return '%s = %.1f +/- %.1f' % (self.name,self.samples.mean(),self.samples.std())

    def resample(N,**kwargs):
        return self.kde.resample(N)

class Hist_Distribution(Distribution):
    def __init__(self,samples,bins=10,smooth=0,**kwargs):
        self.samples = samples
        hist,bins = np.histogram(samples,bins=bins,normed=True)
        self.bins = bins
        self.hist = hist #debug
        bins = (bins[1:] + bins[:-1])/2.
        pdf = interpolate(bins,hist,s=smooth)
        cdf = interpolate(bins,hist.cumsum()/hist.cumsum().max(),s=smooth)

        if 'maxval' not in kwargs:
            kwargs['maxval'] = samples.max()
        if 'minval' not in kwargs:
            kwargs['minval'] = samples.min()

        Distribution.__init__(self,pdf,cdf,**kwargs)

    def __str__(self):
        return '%s = %.1f +/- %.1f' % (self.name,self.samples.mean(),self.samples.std())

    def plothist(self,fig=None,**kwargs):
        plu.setfig(fig)
        plt.hist(self.samples,bins=self.bins,**kwargs)

    def resample(self,N):
        inds = rand.randint(len(self.samples),size=N)
        return self.samples[inds]

class Box_Distribution(Distribution):
    def __init__(self,lo,hi,**kwargs):
        self.lo = lo
        self.hi = hi
        def pdf(x):
            return 1./(hi-lo) + 0*x
        def cdf(x):
            x = np.atleast_1d(x)
            y = (x - lo) / (hi - lo)
            y[np.where(x < lo)] = 0
            y[np.where(x > hi)] = 1
            return y

        Distribution.__init__(self,pdf,cdf,minval=lo,maxval=hi,**kwargs)

    def __str__(self):
        return '%.1f < %s < %.1f' % (self.lo,self.name,self.hi)

    def resample(self,N):
        return rand.random(size=N)*(self.maxval - self.minval) + self.minval

class Combined_Distribution(Distribution):
    def __init__(self,dist_list,minval=-np.inf,maxval=np.inf,**kwargs):
        self.dist_list = list(dist_list)
        #self.Ndists = len(dist_list)
        N = 0
        for dist in dist_list:
            N += dist.Ndists
            
        self.Ndists = N
        self.minval = minval
        self.maxval = maxval

        def pdf(x):
            y = x*0
            for dist in dist_list:
                y += dist(x)
            return y/N

        Distribution.__init__(self,pdf,minval=minval,maxval=maxval,**kwargs)

    def __getitem__(self,ind):
        return self.dist_list[ind]

    #def __add__(self,other):
        
    #    def pdf(x):
    #        return (self(x) + other(x))/(self.Ndists + other.Ndists)
    #    self.dist_list.append(other)
    #    maxval = max(self.maxval,other.maxval)
    #    minval = min(self.minval,other.minval)

    #    Distribution.__init__(self,pdf,maxval=maxval,minval=minval)

    #def __radd__(self,other):
    #    return self.__add__(other)
