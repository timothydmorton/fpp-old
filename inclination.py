from numpy import *
#from utils import *
from consts import *
import scipy
from scipy.interpolate import interp1d
import math
import scipy.stats as stats
import numpy.random as rand
import sys,re,os

#DATAFOLDER = os.environ['ASTROUTIL_DATADIR'] #'/Users/tdm/Dropbox/astroutil/data'

#kozpsi = loadtxt('%s/inclinations/psidist.txt' % DATAFOLDER)
#kozkde = stats.gaussian_kde(concatenate((kozpsi,-kozpsi)))

def rayleigh(n,sigma):
    r = scipy.random.random(n)
    return sqrt(-2*sigma**2*log(1-r))

def px(x):
    return x/sqrt(1-x**2)

def rand_sini(n,xmin=0,xmax=1):
    rmax = 1-sqrt(1-xmax**2)
    rmin = 1-sqrt(1-xmin**2)
    r = scipy.random.random(n)*(rmax-rmin)+rmin
    return sqrt(1-(1-r)**2)


def build_dist(Y,ymin=1,ymax=500,npts=100,n=1e3,lg=False,pdf=False):
    if size(Y)==1:
        Y = array([Y])

    if lg:
        Z = logspace(log10(ymin),log10(ymax),npts)
    else:
        Z = linspace(ymin,ymax,npts)
    PZ = zeros(npts)
    for y in Y:
        sini = rand_sini(n)
        z = y/sini
        h = histogram(z,bins=Z)
        PZ[:-1] += h[0]
    Z = Z[:-1]
    PZ = PZ[:-1]
    if pdf:
        norm = trapz(PZ,Z)
    else:
        norm = sum(PZ)
    return Z,PZ/norm

def pickfromdist(n,z,pz,interp=True):
    cumprob = cumsum(pz)
    rand = scipy.random.random(n)
    if interp:
        f = interp1d(cumprob,z,kind='cubic')
        return f(rand)
    else:
        inds = digitize(rand,cumprob)
        return z[inds]

def rand_inc(n,mininc=0,maxinc=pi/2):
    return arcsin(rand_sini(n,xmin=sin(mininc),xmax=sin(maxinc)))
    
def rand_spherepos(n,mininc=0,maxinc=pi/2,randfn=None,fnarg=None):
    if n==0:
        return None
    if randfn==None:
        theta = (rand_inc(n,mininc=mininc,maxinc=maxinc)-pi/2)*sign(scipy.random.random(n)-0.5)+pi/2
    else:
        if fnarg==None:
            theta = randfn(n)
        else:
            theta = randfn(n,fnarg)
    phi = scipy.random.random(n)*2*pi-pi
    return spherepos((theta,phi))


def old_rand_spherepos(n,mininc=0,maxinc=pi/2,half=False,fishk=None,kozai=False):
    if n==0:
        return None
    if not half and not fishk and not kozai:
        theta = (rand_inc(n,mininc=mininc,maxinc=maxinc)-pi/2)*sign(scipy.random.random(n)-0.5)+pi/2
    elif not fishk and not kozai:
        theta = rand_inc(n,mininc=mininc,maxinc=maxinc)
    elif fishk:
        theta = fisher(n,k=fishk)
    elif kozai:
        theta = rand_kozai(n)
    phi = scipy.random.random(n)*2*pi-pi
    return spherepos((theta,phi))

def dot(v1,v2):
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]

class spherepos:
    def __init__(self,coords,normed=True):
        if len(coords)==2:
            self.theta = coords[0]
            self.phi = coords[1]
            self.x = sin(self.theta)*cos(self.phi)
            self.y = sin(self.theta)*sin(self.phi)
            self.z = cos(self.theta)
        elif len(coords)==3:
            self.x = coords[0]
            self.y = coords[1]
            self.z = coords[2]
            self.norm = sqrt(self.x**2 + self.y**2 + self.z**2)
            self.normed = normed
            if normed:
                self.x = self.x/self.norm
                self.y = self.y/self.norm
                self.z = self.z/self.norm
            self.theta = arccos(self.z/sqrt(self.x**2+self.y**2+self.z**2))
            self.phi = arctan2(self.y,self.x)

    def __getitem__(self,inds):
        return spherepos((self.theta[inds],self.phi[inds]))

    def __len__(self):
        return len(self.theta)

    def __call__(self):
        return self

    def __add__(self,pos2):
        if pos2==None and self != None:
            return self
        if self==None and pos2 != None:
            return pos2
        if self==None and pos2==None:
            return None
        self.theta = concatenate((self.theta,pos2.theta))
        self.phi = concatenate((self.phi,pos2.phi))
        self.x = concatenate((self.x,pos2.x))
        self.y = concatenate((self.y,pos2.y))
        self.z = concatenate((self.z,pos2.z))
        return self

    def transform(self,th,ph):
        if self==None:
            return None
        th = inc; ph = phi
        newaxis = spherepos((th,ph))
        (x,y,z) = newaxis.cart()
        theta = arccos(z)
        phi = arctan2(x,y)  #should it be arctan2(y,x)?
        (x,y,z) = self.cart()
        x2 = cos(phi)*x + cos(theta)*sin(phi)*y + sin(theta)*sin(phi)*z
        y2 = -sin(phi)*x + cos(theta)*cos(phi)*y + sin(theta)*cos(phi)*z
        z2 = -sin(theta)*y + cos(theta)*z
        return spherepos((x2,y2,z2),normed=self.normed)
    
    def cart(self):
        return (self.x,self.y,self.z)

    def sph(self):
        return (self.theta,self.phi)

    def angsep(self,pos2):
        return arccos(dot(self.cart(),pos2.cart()))

def rand_kozai(n):
    psis = kozkde.resample(int(n))
    return abs(psis).ravel()*pi/180

def kozai_psi(n):  #completely obselete?
    data = loadtxt('/home/tdm/kepler/Fstars/psidist.txt')
    num,bins = histogram(data,bins=14)
    norm = sum(num)
    prob = num/float(norm)
    #bins = (bins[1:]+bins[:-1])/2.
    #bins = insert(bins,0,0)
    prob = insert(prob,0,0)
    psi = pickfromdist(n,bins,prob)
    return psi*pi/180
    
def fisher(n,k=0.01,res=1e4):
    if n==0:
        return 0
    psi = linspace(0,pi,res)
    pdf = k/(2*sinh(k))*exp(k*cos(psi))*sin(psi)
    norm = sum(pdf)
    pmf = pdf/norm
    return pickfromdist(n,psi,pmf,interp=False)


def isotropize2d(pos):
    n = size(pos.x)
    psi = rand.random(n)*2*pi
    phi = rand.random(n)*2*pi
    theta = rand.random(n)*2*pi
    
    (cpsi,spsi,cphi,sphi,ctheta,stheta) = (cos(psi),sin(psi),cos(phi),sin(phi),
                                           cos(theta),sin(theta))
    (x,y,z) = pos.cart()
    newx = (cpsi*cphi - ctheta*sphi*spsi)*x + (cpsi*sphi + ctheta*cphi*spsi)*y + (spsi*stheta)*z
    newy = (-spsi*cphi-ctheta*sphi*cpsi)*x + (-spsi*sphi + ctheta*cphi*cpsi)*y + (cpsi*stheta)*z
    newz = (stheta*sphi)*x + (-stheta*cphi)*y + (ctheta)*z

    return spherepos((newx,newy,newz))

    #the Euler rotation matrix
    #A = array([[cpsi*cphi - ctheta*sphi*spsi, cpsi*sphi + ctheta*cphi*spsi, spsi*stheta],
    #           [-spsi*cphi-ctheta*sphi*cpsi, -spsi*sphi + ctheta*cphi*cpsi, cpsi*stheta],
    #           [stheta*sphi,                 -stheta*cphi,                  ctheta]])
    
              
