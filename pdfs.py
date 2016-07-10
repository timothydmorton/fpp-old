from numpy import *
from scipy.stats import gaussian_kde
import pylab as p
from consts import *

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
        C = self.plnorm
        return ((u*(a+1))/C + self.xmin**(a+1))**(1./(a+1))

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
