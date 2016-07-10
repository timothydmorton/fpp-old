from numpy import *
from consts import *
import numpy.random as rand
import utils
import plotutils as pu
import pylab as p
from scipy.integrate import quad

def mutualRH(m1,m2,a1,a2,ms):
    m1 *= MEARTH
    m2 *= MEARTH
    a1 *= AU
    a2 *= AU
    ms *= MSUN
    return ((m1+m2)/(3*ms))**(1./3)*(a1 + a2)/2.


def simpleM(R):
    return R**2.06*MEARTH

def simplestable(m1,m2,a1,a2,ms,return_bool=False):
    RH = mutualRH(m1,m2,a1,a2,ms)
    Delta = (a2-a1)*AU/RH
    if return_bool:
        return Delta < 3.46
    else:
        return Delta

def dilutedradius(rp,dm,dr=0):
    df = 10**(-0.4*dm)
    newr = rp*(1+df)**(0.5)
    return newr


def rp_dilutedpdf(pop,r=1,dr=0.1,fb=0.4,n=1e4,fig=None,plot=True,allpdfs=False): #add contrast curves here
    simr = rand.normal(size=n)*dr + r
    mainpdf = utils.gaussian(r,dr,norm=1-fb)

    inds = rand.randint(size(pop.dkepmag),size=n)

    diluted1 = utils.kde(dilutedradius(simr,pop.dkepmag[inds]),norm=fb/2,adaptive=False)
    diluted2 = utils.kde(dilutedradius(simr,-pop.dkepmag[inds]),norm=fb/2,adaptive=False)
    diluted2.renorm((fb/2)**2/quad(diluted2,0,20)[0])

    totpdf = mainpdf + diluted1 + diluted2

    if plot:
        pu.setfig(fig)
        rs = arange(0,2*r,0.01)
        p.plot(rs,totpdf(rs))
        if allpdfs:
            p.plot(rs,mainpdf(rs))
            p.plot(rs,diluted1(rs)) 
            p.plot(rs,diluted2(rs))

    if allpdfs:
        return totpdf,mainpdf,diluted1,diluted2
    else:
        return totpdf

