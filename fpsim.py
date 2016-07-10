from numpy import *
import numpy as np
import starutils_old as su
import os,sys,re,os.path
import pylab as p
import transit_basic as tr
import atpy
from astropy.table import Table,Column
import numpy.random as rand
from consts import *
import scipy.stats as stats
import utils
from extinction import getAV
import subprocess as sp
import shutil
import orbitutils as ou

from scipy.optimize import fsolve

MODELS = 'padova'
MAFN = tr.MAInterpolationFunction(nzs=200,nps=400,pmin=0.007,pmax=1/0.007)
TRILEGALDIR = os.environ['TRILEGALDIR']

BANDS = ['g','r','i','z','J','H','Ks','Kepler']


def rand_dRV(n,filename='%s/kepler_absrv.npy' % os.environ['ASTROUTIL_DATADIR']):
    rvs = load(filename)
    ind1 = rand.randint(len(rvs),size=n)
    ind2 = rand.randint(len(rvs),size=n)
    return rvs[ind1]-rvs[ind2]

def randpos_in_circle(n,rad=1,return_rad=False):
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

def loadfield(name,dir=TRILEGALDIR,binaries=False,n=None):
    if binaries:
        starfile = '%s/%s_binaries.fits' % (dir,name)
    else:
        starfile = '%s/%s.fits' % (dir,name)        

    try:
        stardata = atpy.Table(starfile,verbose=False)
    except:  #HACK-- this will break if binaries=True and binaries aren't simulated for field.
        trilegal2fits(name,dir=dir)
        stardata = atpy.Table(starfile,verbose=False)
    if n is not None:
        inds = rand.randint(len(stardata),size=n)
        stardata = stardata.rows(inds)
    return stardata
    
def PDMF(name,N,dir=TRILEGALDIR):
    stardata = loadfield(name,dir)
    

def modelpopulation(ms,ages,fehs,bands=BANDS,models=MODELS):
    t = atpy.Table()
    t.add_column('m_ini',ms)
    t.add_column('age',ages)
    t.add_column('feh',fehs)
    t.add_column('M',su.model_M(ms,ages,fehs,models=models))
    t.add_column('logL',su.model_logL(ms,ages,fehs,models=models))
    t.add_column('logTe',su.model_logTe(ms,ages,fehs,models=models))
    t.add_column('logg',su.model_logg(ms,ages,fehs,models=models))
    for band in bands:
        t.add_column(band,su.model_mag(band,ms,ages,fehs,models=models))
    return t

def get_trilegal(name,ra,dec,dir=TRILEGALDIR,filterset='kepler_2mass',area=1,maglim=27):
    l,b = utils.eq2gal(ra,dec)
    outfile = '%s.dat' % (name)
    AV = getAV(l,b,'gal')
    cmd = 'get_trilegal 1.6beta %f %f %f %.3f %s 1 %.1f %s' % (l,b,area,AV,filterset,maglim,outfile)
    print cmd
    sp.Popen(cmd,shell=True).wait()
    shutil.move(outfile,'%s/%s' % (dir,outfile))

def calcBGflux(name,dir=TRILEGALDIR,band='i',mags=[12,13,14],binwidth=None,return_density=False):
    filename = '%s/%s.fits' % (dir,name)
    try:
        cat = atpy.Table(filename,verbose=False)
    except:
        trilegal2fits(name,dir=dir)
        cat = atpy.Table(filename,verbose=False)
    unit = 10**(-0.4*12)
    bgflux = []
    density = []
    for mag in mags:
        if binwidth is None:
            w = where(cat[band] > mag)
        else:
            w = where((cat[band] > mag-binwidth/2.) & (cat[band] < mag+binwidth/2.))
        totflux = (10**(-0.4*cat[band][w])).sum()
        bgflux.append(totflux/unit)
        density.append(size(w))
    if return_density:
        return bgflux,density
    else:
        return bgflux


def trilegal2fits(name,dir=TRILEGALDIR):
    filename = '%s/%s.dat' % (dir,name)
    data = recfromtxt(filename,names=True)
    t = atpy.Table()
    for prop in ['logAge','MH','m_ini','logL','logTe','logg','mM0','Av','m2m1','mbol','Kepler','g','r','i','z','J','H','Ks','Mact']:
        t.add_column(prop,data[prop])
    outfile = '%s/%s.fits' % (dir,name)
    t.write(outfile,overwrite=True,verbose=False)
    print 'TRILEGAL FITS table written to %s.' % outfile
    
def makebinaries(name,dir=TRILEGALDIR,bands=BANDS,fixnans=False,overwrite=True,models=MODELS):
    outfile = '%s/%s_binaries.fits' % (dir,name)
    if not overwrite and os.path.exists(outfile):
        print '%s exists; not overwriting.' % outfile
        return
    
    #tries to load TRILEGAL FITS table; makes if does not exist.
    filename = '%s/%s.fits' % (dir,name)
    try:
        pridata = atpy.Table(filename,verbose=False)
    except:
        trilegal2fits(name,dir=dir)
        pridata = atpy.Table(filename,verbose=False)

    n = len(pridata)

    minm = 0.05
    minq = minm/pridata.m_ini
    minq[where(minq < 0.1)] = 0.1
    qs = rand.random(n)*(1-minq) + minq
    m2s = pridata.m_ini * qs
    wbackward = where(m2s > pridata.m_ini)
    m2s[wbackward] = pridata.m_ini[wbackward]

    if fixnans:
        bad = (m2s > su.PADOVAMAXMASS(pridata.logAge))
        wbad = where(bad)
        nbad = bad.sum()
        while nbad > 0:
            qs[wbad] = rand.random(nbad)*(1-minq[wbad]) + minq[wbad]
            m2s[wbad] = pridata.m_ini[wbad] * qs[wbad]
            bad = (m2s > su.PADOVAMAXMASS(pridata.logAge))
            wbad = where(bad)
            nbad = bad.sum()
        

    t = modelpopulation(m2s,pridata.logAge,pridata.MH,bands=bands,models=models)
    t.add_column('Mact',t['M'])
    wlo = where(pridata.Mact < su.LOWMASSLIMIT['dartmouth'])
    for band in bands:
        t[band] += pridata.mM0 + su.EXTINCTION[band]*pridata.Av
        m1 = pridata.m_ini[wlo]
        m2 = t.Mact[wlo]
        dm = m2-m1; dmsq = m2**2-m1**2
        mag1 = pridata[band][wlo]
        t[band][wlo] = mag1 + 2755.22878829*dmsq - 636.68960397*dm #based on quadratic fit to low-mass (<0.12) Kep band
    t.add_column('mM0',pridata.mM0)
    t.add_column('distance',su.dfromdm(pridata.mM0))
    t.add_column('Av',pridata.Av)

    t.add_keyword('MODELS',models)

    t.write(outfile,overwrite=True,verbose=False)
    print 'binaries fits table written to %s' % outfile

    return t

def eclipse_population(M1s,M2s,R1s,R2s,mag1s,mag2s,u11s=None,u21s=None,u12s=None,u22s=None,Ps=None,incs=None,eccs=None,band='i',logperkde=su.LOGPERKDE,return_probability=False,return_indices=False,minP=0.5,mininc=None,period=None,maxecc=0.97,verbose=True,debug=False):

    n = size(M1s)
    simPs = False
    if period:
        Ps = ones(n)*period
    else:
        if Ps is None:
            Ps = 10**(logperkde.draw(n))
            simPs = True
    simeccs = False
    if eccs is None:
        if not simPs and period is not None:
            eccs = su.draw_eccs(n,period,maxecc=maxecc)
        else:
            eccs = su.draw_eccs(n,Ps,maxecc=maxecc)
        simeccs = True

    if u11s is None or u21s is None:
        u11s = ones(n)*0.394
        u21s = ones(n)*0.261
    if u12s is None or u22s is None:
        u12s = ones(n)*0.394
        u22s = ones(n)*0.261

    semimajors = su.semimajor(Ps,M1s+M2s)*AU

    #wtooclose = where(semimajors*(1-eccs) < 2*(R1s+R2s)*RSUN)
    tooclose = su.withinroche(semimajors*(1-eccs)/AU,M1s,R1s,M2s,R2s)
    ntooclose = tooclose.sum()
    tries = 0
    maxtries=5
    if simPs:
        while ntooclose > 0:
            lastntooclose=ntooclose
            Ps[tooclose] = 10**(logperkde.draw(ntooclose))
            if simeccs:
                eccs[wtooclose] = su.draw_eccs(ntooclose,Ps[tooclose])
            semimajors[wtooclose] = su.semimajor(Ps[tooclose],M1s[tooclose]+M2s[tooclose])*AU
            tooclose = su.withinroche(semimajors*(1-eccs)/AU,M1s,R1s,M2s,R2s)
            ntooclose = tooclose.sum()
            if ntooclose==lastntooclose:   #prevent infinite loop
                tries += 1
                if tries > maxtries:
                    if verbose:
                        print '%i binaries are "too close"; gave up trying to fix.' % ntooclose
                    break                       
    else:
        while ntooclose > 0:
            lastntooclose=ntooclose
            if simeccs:
                eccs[tooclose] = su.draw_eccs(ntooclose,Ps[tooclose])
            semimajors[tooclose] = su.semimajor(Ps[tooclose],M1s[tooclose]+M2s[tooclose])*AU
            #wtooclose = where(semimajors*(1-eccs) < 2*(R1s+R2s)*RSUN)
            tooclose = su.withinroche(semimajors*(1-eccs)/AU,M1s,R1s,M2s,R2s)
            ntooclose = tooclose.sum()
            if ntooclose==lastntooclose:   #prevent infinite loop
                tries += 1
                if tries > maxtries:
                    if verbose:
                        print '%i binaries are "too close"; gave up trying to fix.' % ntooclose
                    break                       

    if incs is None:
        if mininc is None:
            incs = arccos(rand.random(n)) #random inclinations in radians
        else:
            incs = arccos(rand.random(n)*cos(mininc*pi/180))
    if mininc:
        prob = cos(mininc*pi/180)
    else:
        prob = 1

    if debug:
        pass
        #p.figure()
        #p.hist(eccs)
        #p.xlabel('ecc')

    ws = rand.random(n)*2*pi        

    switched = (R2s > R1s)
    R_large = switched*R2s + ~switched*R1s
    R_small = switched*R1s + ~switched*R2s

    if debug:
        #p.figure()
        #p.hist(M2s,histtype='step',lw=3,label='all',normed=True)
        #p.hist(M2s[wtooclose],histtype='step',lw=3,label='too close',normed=True)
        #p.xlabel('M2')
        #p.legend()

        #p.figure()
        #w1 = where(M2s > 0.5)
        #w2 = where(M2s < 0.5)
        #p.hist(log10(semimajors[w1]/AU),histtype='step',lw=3,label='big',normed=True)
        #p.hist(log10(semimajors[w2]/AU),histtype='step',lw=3,label='small',normed=True)
        #p.xlabel('log(a)')
        #p.legend()
        pass
        


    b_tras = semimajors*cos(incs)/(R_large*RSUN) * (1-eccs**2)/(1 + eccs*sin(ws))
    b_occs = semimajors*cos(incs)/(R_large*RSUN) * (1-eccs**2)/(1 - eccs*sin(ws))

    b_tras[tooclose] = inf
    b_occs[tooclose] = inf

    ks = R_small/R_large
    Rtots = (R_small + R_large)/R_large
    tra = (b_tras < Rtots)
    occ = (b_occs < Rtots)
    nany = (tra | occ).sum()
    peb = nany/float(n)
    if return_probability:
        return prob*peb,prob*sqrt(nany)/n


    i = (tra | occ)
    wany = where(i)
    P,M1,M2,R1,R2,mag1,mag2,inc,ecc,w = Ps[i],M1s[i],M2s[i],R1s[i],R2s[i],\
        mag1s[i],mag2s[i],incs[i]*180/pi,eccs[i],ws[i]*180/pi
    a = semimajors[i]  #in cm already
    b_tra = b_tras[i]
    b_occ = b_occs[i]
    u11 = u11s[i]
    u21 = u21s[i]
    u12 = u12s[i]
    u22 = u22s[i]

    if debug:
        #debug plots
        pass
        #p.figure()
        #p.hist(M2s/M1s,histtype='step',lw=3,label='all',normed=True)
        #p.hist(M2/M1,histtype='step',lw=3,label='eclipsing',normed=True)
        #p.legend()

        #p.figure()
        #p.hist(R1s,histtype='step',lw=3,label='all',normed=True)
        #p.hist(R1,histtype='step',lw=3,label='eclipsing',normed=True)
        #p.title('R1')
        #p.legend()

        #p.figure()
        #p.hist(M1s,histtype='step',lw=3,label='all',normed=True)
        #p.hist(M1,histtype='step',lw=3,label='eclipsing',normed=True)
        #p.title('M1')
        #p.legend()

        #p.figure()
        #p.hist(R2s,histtype='step',lw=3,label='all',normed=True)
        #p.hist(R2,histtype='step',lw=3,label='eclipsing',normed=True)
        #p.title('R2')
        #p.legend()

        #p.figure()
        #p.hist(M2s,histtype='step',lw=3,label='all',normed=True)
        #p.hist(M2,histtype='step',lw=3,label='eclipsing',normed=True)
        #p.title('M2')
        #p.legend()

    #####


    #p.figure()
    #p.hist(ecc)

    switched = (R2 > R1)
    R_large = switched*R2 + ~switched*R1
    R_small = switched*R1 + ~switched*R2
    k = R_small/R_large
    
    #calculate durations
    T14_tra = P/pi*arcsin(R_large*RSUN/a * sqrt((1+k)**2 - b_tra**2)/sin(inc*pi/180)) *\
        sqrt(1-ecc**2)/(1+ecc*sin(w*pi/180)) #*24*60
    T23_tra = P/pi*arcsin(R_large*RSUN/a * sqrt((1-k)**2 - b_tra**2)/sin(inc*pi/180)) *\
        sqrt(1-ecc**2)/(1+ecc*sin(w*pi/180)) #*24*60
    T14_occ = P/pi*arcsin(R_large*RSUN/a * sqrt((1+k)**2 - b_occ**2)/sin(inc*pi/180)) *\
        sqrt(1-ecc**2)/(1-ecc*sin(w*pi/180)) #*24*60
    T23_occ = P/pi*arcsin(R_large*RSUN/a * sqrt((1-k)**2 - b_occ**2)/sin(inc*pi/180)) *\
        sqrt(1-ecc**2)/(1-ecc*sin(w*pi/180)) #*24*60

    bad = (isnan(T14_tra) & isnan(T14_occ))
    if bad.sum() > 0:
        print 'Something snuck through with no eclipses [details below]'
        print 'k:',k[wbad]
        print 'b_tra:',b_tra[wbad]
        print 'b_occ:',b_occ[wbad]
        print 'T14_tra:',T14_tra[wbad]
        print 'T14_occ:',T14_occ[wbad]
        print 'under sqrt (tra):',(1+k[wbad])**2 - b_tra[wbad]**2
        print 'under sqrt (occ):',(1+k[wbad])**2 - b_occ[wbad]**2
        print 'ecc:',ecc[wbad]**2
        print 'a in Rsun:',a[wbad]/RSUN
        print 'R_large:',R_large[wbad]
        print 'R_small:',R_small[wbad]
        print 'P:',P[wbad]
        print 'total M:',M1[w]+M2[wbad]

    T14_tra[(isnan(T14_tra))] = 0
    T23_tra[(isnan(T23_tra))] = 0
    T14_occ[(isnan(T14_occ))] = 0
    T23_occ[(isnan(T23_occ))] = 0

    #implement specific limb darkening? (i think it is; just need to call eclipse_population with parameters)

    ftra = MAFN(k,b_tra,u11,u21)
    focc = MAFN(1/k,b_occ/k,u12,u22)

    #fix those with k or 1/k out of range of MAFN....or do it in MAfn eventually?
    wtrabad = where((k < MAFN.pmin) | (k > MAFN.pmax))
    woccbad = where((1/k < MAFN.pmin) | (1/k > MAFN.pmax))
    #print '{} eclipses have k < {} or k > {} (out of range of MAFN), min={:.2g}, max={:.2g}'.format(len(wtrabad[0]),MAFN.pmin,MAFN.pmax,k.min(),k.max())
    #print '{} occultations have k < {} or k > {} (out of range of MAFN), min={:.2g}, max={:.2g}'.format(len(woccbad[0]),MAFN.pmin,MAFN.pmax,(1/k).min(),(1/k).max())
    for ind in wtrabad[0]:
        ftra[ind] = tr.occultquad(b_tra[ind],u11[ind],u21[ind],k[ind])
    for ind in woccbad[0]:
        focc[ind] = tr.occultquad(b_occ[ind]/k[ind],u12[ind],u22[ind],1/k[ind])

    F1 = 10**(-0.4*mag1) + switched*10**(-0.4*mag2)
    F2 = 10**(-0.4*mag2) + switched*10**(-0.4*mag1)

    dtra = 1-(F2 + F1*ftra)/(F1+F2)
    docc = 1-(F1 + F2*focc)/(F1+F2)

    totmag = -2.5*log10(F1+F2)

    #wswitched = where(switched)
    dtra[switched],docc[switched] = (docc[switched],dtra[switched])
    T14_tra[switched],T14_occ[switched] = (T14_occ[switched],T14_tra[switched])
    T23_tra[switched],T23_occ[switched] = (T23_occ[switched],T23_tra[switched])
    b_tra[switched],b_occ[switched] = (b_occ[switched],b_tra[switched])
    #mag1[wswitched],mag2[wswitched] = (mag2[wswitched],mag1[wswitched])
    F1[switched],F2[switched] = (F2[switched],F1[switched])
    u11[switched],u12[switched] = (u12[switched],u11[switched])
    u21[switched],u22[switched] = (u22[switched],u21[switched])

    dtra[(isnan(dtra))] = 0
    docc[(isnan(docc))] = 0

    t = atpy.Table()
    t.add_column('%s_mag_tot' % band,totmag)
    t.add_column('P',P)
    t.add_column('ecc',ecc)
    t.add_column('inc',inc)
    t.add_column('w',w)
    t.add_column('dpri',dtra)
    t.add_column('dsec',docc)
    t.add_column('T14_pri',T14_tra)
    t.add_column('T23_pri',T23_tra)
    t.add_column('T14_sec',T14_occ)
    t.add_column('T23_sec',T23_occ)
    t.add_column('b_pri',b_tra)
    t.add_column('b_sec',b_occ)
    t.add_column('%s_mag1' % band,mag1)
    t.add_column('%s_mag2' % band,mag2)
    t.add_column('fluxfrac1',F1/(F1+F2))
    t.add_column('fluxfrac2',F2/(F1+F2))
    t.add_column('switched',switched)
    t.add_column('u11',u11)
    t.add_column('u21',u21)
    t.add_column('u12',u12)
    t.add_column('u22',u22)

    t.add_keyword('prob',prob*peb)
    t.add_keyword('dprob',prob*sqrt(nany)/n)

    if return_indices:
        return wany,t
    else:
        return t

def simbgplanets(name,dir=TRILEGALDIR,fpl=1,band='Kepler',n=1e5,mininc=None,P=None,alpha=-2,alpha1=-1,alpha2=-3,rbreak=2,mag2=35,rbin=None,return_probability=False,verbose=True,bands = ['g','r','i','z','J','H','Ks','Kepler']):
    starfile = '%s/%s.fits' % (dir,name)
    try:
        stardata = atpy.Table(starfile,verbose=False)
    except:
        trilegal2fits(name,dir=dir)
        stardata = atpy.Table(starfile,verbose=False)


    simkeywords = {}
    if P is not None:
        simkeywords['P'] = P
    if alpha1 is not None:
        simkeywords['ALPHA1'] = alpha1
        simkeywords['ALPHA2'] = alpha2
        simkeywords['RBREAK'] = rbreak
    else:
        simkeywords['ALPHA'] = alpha

    nall = len(stardata)
    inds = rand.randint(nall,size=n)
    stardata = stardata.rows(inds)

    nstars = len(stardata)
    density,ddensity = (nstars/(3600.**2),sqrt(nstars)/(3600.**2))  #per sq. arcsec -- assumes field is 1 degsq!

    M1s = stardata.Mact
    R1s = sqrt(G*M1s*MSUN/(10**stardata.logg))/RSUN
    mag1s = stardata[band]
    logg1s = stardata.logg
    Teff1s = 10**stardata.logTe
    u1s,u2s = tr.ldcoeffs(Teff1s,logg1s)

    if rbin is None:
        if alpha1 is not None:
            plaw = utils.broken_powerlaw(alpha1,alpha2,rbreak)
            R2s = plaw.draw(n)
        else:
            plaw = utils.powerlaw(alpha,0.4,20)
        R2s = plaw.draw(n)
    else:
        r,dr = rbin
        R2s = r + (rand.random(size=n)*2*dr - dr)
    M2s = ((R2s**2.06) * (1 + rand.normal(size=n)*0.1))*MEARTH/MSUN  #hokey M-R relationship
    R2s *= REARTH/RSUN
    mag2s = ones(n)*mag2

    if P is not None:
        if mininc is None:
            mininc = tr.minimum_inclination(P,M1s,M2s,R1s,R2s)

    inds,transits = eclipse_population(M1s,M2s,R1s,R2s,mag1s,mag2s,band=band,return_indices=True,mininc=mininc,period=P,u11s=u1s,u21s=u2s,verbose=verbose)

    prob,dprob = (transits.keywords['prob'],transits.keywords['dprob'])
    transits.keywords['prob'] = prob*density
    transits.keywords['dprob'] = transits.keywords['prob']*sqrt((ddensity/density)**2 + (dprob/prob)**2)
    if return_probability:
        return transits.keywords['prob'],transit.keywords['dprob']

    transits.add_column('M1',M1s[inds])
    transits.add_column('M2',M2s[inds])
    transits.add_column('R1',R1s[inds])
    transits.add_column('R2',R2s[inds])
    transits.add_column('rad',randpos_in_circle(len(transits),return_rad=True))
    transits.add_column('RV',rand_dRV(len(transits)))


    for b in bands:
        transits.add_column(b,stardata[b][inds])
    
    for kw,val in simkeywords.iteritems():
        transits.add_keyword(kw,val)

    return transits

def simplanets(P=None,M=1,R=1,dM=0.1,dR=0.1,u1=0.394,u2=0.296,alpha=-2,n=3e4,mag2=35,band='Kepler',mininc=None,rbin=None,return_probability=False,verbose=True,exactstar=True,stellarmodels=MODELS,age=None,Mdist=None,Rdist=None,alpha1=-1,alpha2=-3,rbreak=2):

    if exactstar:
        dM = 0
        dR = 0
    if Mdist is not None and not exactstar:
        M1s = Mdist.resample(n)
    else:
        M1s = M + rand.normal(size=n)*dM
    M1s[where(M1s < 0.072)] = 0.072
    
    if Rdist is not None and not exactstar:
        R1s = Rdist.resample(n)
        R = Rdist.mu
    else:
        R1s = R + rand.normal(size=n)*dR
    R1s[where(R1s < RJUP/RSUN)] = RJUP/RSUN

    if age is None:
        age = log10(0.5*10**(su.PADOVAMAXAGE(M1s)))

    mag1s = su.model_mag(band,M1s,models=stellarmodels,age=age)

    simkeywords = {}
    if P is not None:
        simkeywords['P'] = P
        
    if Mdist is not None:
        simkeywords['M'] = Mdist.mu
        try:
            simkeywords['DM_P'] = Mdist.sighi
            simkeywords['DM_N'] = Mdist.siglo
        except AttributeError:
            simkeywords['DM_P'] = Mdist.sig
            simkeywords['DM_N'] = Mdist.sig
    else:
        simkeywords['M'] = M
        if not exactstar:
            simkeywords['DM_P'] = dM
            simkeywords['DM_N'] = dM
    if Rdist is not None:
        simkeywords['R'] = Rdist.mu
        try:
            simkeywords['DR_P'] = Rdist.sighi
            simkeywords['DR_N'] = Rdist.siglo
        except AttributeError:
            simkeywords['DR_P'] = Rdist.sig
            simkeywords['DR_N'] = Rdist.sig
    else:
        simkeywords['R'] = R
        simkeywords['DR_P'] = dR
        simkeywords['DR_N'] = dR

    simkeywords['U1'] = u1
    simkeywords['U2'] = u2
    if rbin is not None:
        simkeywords['RBINCEN'] = rbin[0]
        simkeywords['RBINWID'] = rbin[1]
    elif alpha1 is not None:
        simkeywords['ALPHA1'] = alpha1
        simkeywords['ALPHA2'] = alpha2
        simkeywords['RBREAK'] = rbreak
    else:
        simkeywords['ALPHA'] = alpha

    if rbin is None:
        if alpha1 is not None:
            plaw = utils.broken_powerlaw(alpha1,alpha2,rbreak)
            R2s = plaw.draw(n)
        else:
            plaw = utils.powerlaw(alpha,0.4,20)
            R2s = plaw.draw(n)
    else:
        r,dr = rbin
        #ror,dror = (r*REARTH/(R*RSUN),dr*REARTH/(R*RSUN))
        #R2s = r + (rand.random(size=n)*2*dr - dr)
        R2s = r + (rand.random(size=n)*2*dr - dr)
        R2s *= R1s/R  #ensure that R/R* is the actual rbin
    M2s = ((R2s**2.06) * (1 + rand.normal(size=n)*0.1))*MEARTH/MSUN  #hokey M-R relationship
    R2s *= REARTH/RSUN
    mag2s = ones(n)*mag2

    if P is not None:
        if mininc is None:
            try:
                mininc = tr.minimum_inclination(P,M1s,M2s,R1s,R2s)
            except tr.AllWithinRocheError:
                raise tr.AllWithinRocheError('Planet simulation failed (all simulated planets inside star).  Parameters used %s, stellar radii: %s' % (simkeywords,R1s))

    inds,transits = eclipse_population(M1s,M2s,R1s,R2s,mag1s,mag2s,band=band,return_indices=True,mininc=mininc,period=P,u11s=u1*ones(n),u21s=u2*ones(n),verbose=verbose)

    if return_probability:
        return (transits.keywords['prob'],transits.keywords['dprob'])

    transits.add_column('M1',M1s[inds])
    transits.add_column('M2',M2s[inds])
    transits.add_column('R1',R1s[inds])
    transits.add_column('R2',R2s[inds])

    for kw,val in simkeywords.iteritems():
        transits.add_keyword(kw,val)
    
    return transits

def solve_mags(magtot,dmag):
    def fn(mag):
        return addmags(mag + dmag,mag) - magtot
    mag1 = fsolve(fn,magtot,xtol=0.01)
    mag2 = mag1 + dmag
    return mag1,mag2

def addmags(*mags):
    tot=0
    for mag in mags:
        tot += 10**(-0.4*mag)
    return -2.5*log10(tot)

def simHEBs_specific(M,dmags,dist=None,PA=None,obsmags=None,P=None,dm=None,band='i',
                     stellarmodels=MODELS,bands=BANDS,verbose=True,n=1e5,colors=['gr','JK'],
                     blended=True,starfieldname=None,colortol=0.1,**kwargs):
    if obsmags is None:
        raise ValueError('obsmags must be passed for simHEBs_specific')

    simkeywords = {'DMAG_{}'.format(b):dmags[b] for b in dmags.keys()} 
    simkeywords['DIST'] = dist
    simkeywords['PA'] = PA

    mult = sim_multiples(M,P=P,dm=dm,fB1=1,fB2=1,obsmags=obsmags,colors=colors,
                         n=n,band=band,stellarmodels=stellarmodels,bands=bands,
                         n_resample_colormatch=n*10,
                         verbose=verbose,starfieldname=starfieldname,
                         colortol=colortol,only_heb=True,
                         **kwargs)
    trips = mult[3]

    CwA = (trips.which_eclipse=='A')
    CwB = (trips.which_eclipse=='B')

    cond = np.ones(len(trips)).astype(bool)
    for band in dmags.keys():        
        magA = trips['%s_A' % band]
        magB = trips['%s_B' % band]
        magC = trips['%s_C' % band]
        
        mag1 = magA*CwB + addmags(magA,magC)*CwA
        mag2 = addmags(magB,magC)*CwB + magB*CwA

        dmag = mag2-mag1

        cond &= np.absolute(dmag - dmags[band]) < colortol

    hebs = trips.where(cond)
    
    for kw,val in simkeywords.iteritems():
        hebs.add_keyword(kw,val)

    return hebs


def simBEBs_specific(name,mags,dist=None,PA=None,n=1e5,
                     dmag=0.15,dir=TRILEGALDIR,band='i',mininc=None,return_density=False,
                     P=None,verbose=True,bands=BANDS,stellarmodels=MODELS,**kwargs):
    simkeywords = {}
    simkeywords['DIST'] = dist
    simkeywords['PA'] = PA

    if P is not None:
        simkeywords['P'] = P
    
    pris = loadfield(name,dir)
    secs = loadfield(name,dir,binaries=True)

    mask = np.ones(len(pris)).astype(bool)
    for b in mags.keys():
        simkeywords['{}MAG'.format(b)] = mags[b]
        summags = addmags(pris[b],secs[b])
        mask &= (summags < (mags[b] + dmag)) & (summags > (mags[b] - dmag))

    pridata = pris.where(mask)
    secdata = secs.where(mask)

    print len(pridata)

    inds = rand.randint(len(pridata),size=n)
    pridata = pridata.rows(inds)
    secdata = secdata.rows(inds)

    M1s,M2s = (pridata['Mact'],secdata['Mact'])
    R1s = sqrt(G*M1s*MSUN/(10**pridata['logg']))/RSUN
    R2s = sqrt(G*M2s*MSUN/(10**secdata['logg']))/RSUN
    mag1s,mag2s = (pridata[band],secdata[band])
    Teff1s = 10**pridata['logTe']
    Teff2s = 10**secdata['logTe']
    logg1s = pridata['logg']
    u11s,u21s = tr.ldcoeffs(Teff1s,logg1s)
    logg2s = secdata.logg
    u12s,u22s = tr.ldcoeffs(Teff1s,logg1s)
    
    if P is not None:
        if mininc is None:
            mininc = tr.minimum_inclination(P,M1s,M2s,R1s,R2s)

    inds,ebs = eclipse_population(M1s,M2s,R1s,R2s,mag1s,mag2s,band=band,return_indices=True,verbose=verbose,
                                  mininc=mininc,period=P,u11s=u11s,u21s=u21s,u12s=u12s,u22s=u22s)

    #prob = ebs.keywords['prob']
    #dprob = ebs.keywords['dprob']
    #ebs.keywords['prob'] = prob*density
    #ebs.keywords['dprob'] = ebs.keywords['prob']*sqrt((ddensity/density)**2 + (dprob/prob)**2)
    
    #if return_density:
    #    return ebs.keywords['prob'],ebs.keywords['dprob']


    #ebs.add_column('tile',ones(size(inds))*itile)
    ebs.add_column('M1',M1s[inds])
    ebs.add_column('M2',M2s[inds])
    ebs.add_column('R1',R1s[inds])
    ebs.add_column('R2',R2s[inds])
    ebs.add_column('Teff1',Teff1s[inds])
    ebs.add_column('Teff2',Teff2s[inds])
    ebs.add_column('distance',secdata.distance[inds])
    ebs.add_column('age',secdata.age[inds])
    ebs.add_column('mM0',pridata.mM0[inds])
    ebs.add_column('Av',pridata.Av[inds])
    ebs.add_column('MH',pridata.MH[inds])
    ebs.add_column('rad',randpos_in_circle(len(ebs),return_rad=True))
    ebs.add_column('RV',rand_dRV(len(ebs)))

    for b in bands:  #make this work, in general.  
        ebs.add_column('%s_1' % b,pridata[b][inds])
        ebs.add_column('%s_2' % b,secdata[b][inds])


    for kw,val in simkeywords.iteritems():
        ebs.add_keyword(kw,val)

    return ebs


def simBEBs(name,dir=TRILEGALDIR,fB=0.4,band='i',plot=False,fig=None,return_density=False,mininc=None,P=None,verbose=True,bands=BANDS,stellarmodels=MODELS,resim=False):
    models = stellarmodels
    prifile = '%s/%s.fits' % (dir,name)
    try:
        pridata = atpy.Table(prifile,verbose=False)
    except:
        trilegal2fits(name,dir=dir)
        pridata = atpy.Table(prifile,verbose=False)

    secfile = '%s/%s_binaries.fits' % (dir,name)
    try:
        secdata = atpy.Table(secfile,verbose=False)
    except:
        makebinaries(name,dir=dir,models=models)
        secdata = atpy.Table(secfile,verbose=False)

    #resimulate if desired
    if resim:
        makebinaries(name,dir=dir,models=models)
        secdata = atpy.Table(secfile,verbose=False)

    #record parameters used for simulation
    simkeywords = {}
    #simkeywords['fB'] = fB  #not recording this; not important.
    if P is not None:
        simkeywords['P'] = P

    n = len(pridata)

    wbinary = where(rand.random(n) < fB)[0]
    pridata = pridata.rows(wbinary)
    secdata = secdata.rows(wbinary)
    nbins = len(pridata)
    density,ddensity = (nbins/(3600.**2),sqrt(nbins)/(3600.**2))  #per sq. arcsec -- assumes field is 1 degsq!

    M1s,M2s = (pridata.Mact,secdata.Mact)
    R1s = sqrt(G*M1s*MSUN/(10**pridata.logg))/RSUN
    R2s = sqrt(G*M2s*MSUN/(10**secdata.logg))/RSUN
    mag1s,mag2s = (pridata[band],secdata[band])
    Teff1s = 10**pridata.logTe
    Teff2s = 10**secdata.logTe
    logg1s = pridata.logg
    u11s,u21s = tr.ldcoeffs(Teff1s,logg1s)
    logg2s = secdata.logg
    u12s,u22s = tr.ldcoeffs(Teff1s,logg1s)

    
    if P is not None:
        if mininc is None:
            mininc = tr.minimum_inclination(P,M1s,M2s,R1s,R2s)

    #prob,dprob = eclipse_population(M1s,M2s,R1s,R2s,mag1s,mag2s,band=band,return_probability=True,verbose=verbose,
    #                                mininc=mininc,period=P,u11s=u11s,u21s=u21s,u12s=u12s,u22s=u22s)
    
    inds,ebs = eclipse_population(M1s,M2s,R1s,R2s,mag1s,mag2s,band=band,return_indices=True,verbose=verbose,
                                  mininc=mininc,period=P,u11s=u11s,u21s=u21s,u12s=u12s,u22s=u22s)

    prob = ebs.keywords['prob']
    dprob = ebs.keywords['dprob']
    ebs.keywords['prob'] = prob*density
    ebs.keywords['dprob'] = ebs.keywords['prob']*sqrt((ddensity/density)**2 + (dprob/prob)**2)
    
    if return_density:
        return ebs.keywords['prob'],ebs.keywords['dprob']


    #ebs.add_column('tile',ones(size(inds))*itile)
    ebs.add_column('M1',M1s[inds])
    ebs.add_column('M2',M2s[inds])
    ebs.add_column('R1',R1s[inds])
    ebs.add_column('R2',R2s[inds])
    ebs.add_column('Teff1',Teff1s[inds])
    ebs.add_column('Teff2',Teff2s[inds])
    ebs.add_column('distance',secdata.distance[inds])
    ebs.add_column('age',secdata.age[inds])
    ebs.add_column('mM0',pridata.mM0[inds])
    ebs.add_column('Av',pridata.Av[inds])
    ebs.add_column('MH',pridata.MH[inds])
    ebs.add_column('rad',randpos_in_circle(len(ebs),return_rad=True))
    ebs.add_column('RV',rand_dRV(len(ebs)))

    for b in bands:  #make this work, in general.  
        ebs.add_column('%s_1' % b,pridata[b][inds])
        ebs.add_column('%s_2' % b,secdata[b][inds])

    for kw,val in simkeywords.iteritems():
        ebs.add_keyword(kw,val)

    return ebs

def mult_masses(MAs,hasB,hasC,CwA,CwB,minm=0.05,minq=0.1):
    n = len(MAs)
    q1s = rand.random(n)*(1-minq) + minq
    q2s = rand.random(n)*(1-minq) + minq
    
    MBs = (CwA*MAs*q1s*(1-q2s) + CwB*MAs*q1s)*hasC + q1s*MAs*~hasC
    MCs = CwA*MAs*q2s + CwB*MBs*q2s    

    MBs *= hasB
    MCs *= hasC

    return MBs,MCs
    

def mult_masses_simple(MAs,hasB,hasC,CwA,CwB,minm=0.05,minq=0.1):
    n = len(MAs)
    q1s = rand.random(n)*(1-minq) + minq
    q2s = rand.random(n)*(1-minq) + minq
    
    #MBs = q1s*MAs*~hasC + (CwA*q1s*(1+q2s)*MAs + CwB*q1s/(1+q2s)*MAs)*hasC
    MBs = q1s*MAs
    MCs = CwA*MAs*q2s + CwB*MBs*q2s

    MBs *= hasB
    MCs *= hasC

    return MBs,MCs

def multiple_population(MAs,fB1=0.5,fB2=0.25,n=1e5,minm=0.05,P=None,multmassfn='default',
                        minq=0.1,return_keys=False):
    simkeywords = {}
    simkeywords['MINM'] = minm
    simkeywords['MINQ'] = minq
    simkeywords['MULTMASSFN'] = multmassfn
    
    if multmassfn == 'simple':
        multmassfn = mult_masses_simple
    elif multmassfn == 'default':
        multmassfn = mult_masses

    if size(MAs) == 1:
        MAs = ones(n)*MAs
    n = len(MAs)
    
    hasB = (rand.random(n) < fB1)
    hasC = (rand.random(n) < fB2) & hasB

    if P is None:
        P1s = 10**(su.LOGPERKDE.draw(n))      #drawn from binary period distribution 
        P2s = 10**(su.TRIPLOGPERKDE.draw(n))  #drawn from multiple-system period distribution

        #also do orbits here so that i can ensure no "inside-star" orbits at this stage?
    
        CwA = (P1s < P2s)   # C orbits A if P1 is shorter
        CwB = ~CwA          # C orbits B if P2 is shorter

        Pshorter = P1s*CwA + P2s*CwB
        Plonger = P1s*CwB + P2s*CwA

        P1 = hasB*P1s + hasC*Plonger
        P2 = hasC*Pshorter
    else:
        CwA = (rand.random(n) < 0.5)
        CwB = ~CwA

        P1 = hasB*P + hasC*P
        P2 = hasC*P

    #q1s = rand.random(n)*0.9 + 0.1  #maybe can restrict q1 to be more like max of 0.9?
    #q2s = rand.random(n)*0.9 + 0.1
    
    #MBs = (CwA*MAs*q1s*(1-q2s) + CwB*MAs*q1s)*hasC + q1s*MAs*~hasC
    #MCs = CwA*MAs*q2s + CwB*MBs*q2s    

    #MBs *= hasB
    #MCs *= hasC

    MBs,MCs = multmassfn(MAs,hasB,hasC,CwA,CwB,minm=minm,minq=minq)

    lo = ((MBs < minm) & (MBs != 0)) | ((MCs < minm) & (MCs !=0))
    nlo = lo.sum()
    niter = 0
    while nlo > 0:
        wlo = where(lo)
        MBs[wlo],MCs[wlo] = multmassfn(MAs[wlo],hasB[wlo],hasC[wlo],CwA[wlo],CwB[wlo],minm=minm,minq=minq)
        lo = ((MBs < minm) & (MBs != 0)) | ((MCs < minm) & (MCs !=0))
        nlo = lo.sum()
        niter += 1
        if niter == 100:
            break
    wlo = where(lo)
    MBs[wlo] = 0
    MCs[wlo] = 0
    hasB[wlo] = False
    hasC[wlo] = False
            

    N = 1 + hasB + hasC

    #MB,MC = (MBs*CwA + MCs*CwB, MCs*CwA + MBs*CwB)
    which_eclipse = array([' ']*n)
    which_eclipse[where(CwA & hasC)] = 'A'
    which_eclipse[where(CwB & hasC)] = 'B'
    
    mult = rec.array([N,MAs,MBs,MCs,P1,P2,which_eclipse],names=('N','MA','MB','MC','P1','P2','which_eclipse'))
    if return_keys:
        return mult,simkeywords
    else:
        return mult


def sim_binaries(MAs=None,dM=None,n=1e5,obsmags=None,colors=['gr','JK'],
                 bands=['g','r','i','z','J','H','Ks','Kepler'],
                 n_resample_colormatch=None,
                 fehs=0,ages=None,minm=0.075,stellarmodels=MODELS,minq=0.1):
    #MAs = M + rand.normal(size=n)*dM
    models=stellarmodels

    if MAs is None:
        M = None
        dM = None
    elif size(MAs) == 1:
        M = MAs
        if dM is not None:
            MAs = MAs + rand.normal(size=n)*dM
        else:
            MAs = ones(n)*MAs
    else:
        M=None
        dM=None

    if obsmags is None:
        if MAs is None:
            raise ValueError('If you do not give obsmags, you have to specify MAs and vice versa')
        mult = multiple_population(MAs,fB1=1,fB2=0,minq=minq,minm=minm)
        simkeywords = {}
    else:
        mult,ages,fehs,simkeywords = colormatch_multiples(obsmags,colors,n=n,tol=colortol,M=M,dM=dM,N=2,return_keys=True,
                                              return_all=True,minm=minm,starfieldname=starfieldname,stellarmodels=models)
        
        if n_resample_colormatch is not None:
        #resampling to speed things up
            inds = rand.randint(len(mult),size=n_resample_colormatch)
            mult = mult[inds]
            ages = ages[inds]
            fehs = fehs[inds]
            MAs = mult.MA[inds]
        else:
            MAs = mult.MA

    if ages is None:
        maxages = 10**su.PADOVAMAXAGE(MAs)
        ages = rand.random(size(MAs))*(maxages*0.95) #max age is 95% of total lifetime
        ages = log10(ages)
    elif size(ages)==1:
        ages = ones(n)*ages

    t = atpy.Table()
    t.add_column('MA',mult.MA)
    t.add_column('MB',mult.MB)
    t.add_column('P',mult.P1)
    eccs = su.draw_eccs(len(t.MA),t.P,maxecc=0.97)
    t.add_column('ecc',eccs)
    for b in bands:
        magA = su.model_mag(b,mult.MA,ages,fehs,models=models)
        magB = su.model_mag(b,mult.MB,ages,fehs,models=models)
        FA = 10**(-0.4*magA)
        FB = 10**(-0.4*magB)
        t.add_column('%s_A' % b,magA)
        t.add_column('%s_B' % b,magB)
        t.add_column('%s_tot' % b,-2.5*log10(FA+FB))
    #rad,rv = su.sim_binary_orbits(mult.MA,mult.MB,mult.P1)
    t.add_column('age',ages)
    t.add_column('feh',fehs)
    orbpop = ou.OrbitPopulation(t.MA,t.MB,t.P,t.ecc)
    t.add_column('rsky',orbpop.Rsky)
    t.add_column('RV',orbpop.RVs)
    t.add_column('Manomaly',orbpop.Ms)
    t.add_column('obsx',orbpop.obspos.x)
    t.add_column('obsy',orbpop.obspos.y)
    t.add_column('obsz',orbpop.obspos.z)
        
    for kw,val in simkeywords.iteritems():
        t.add_keyword(kw,val)

    return t


def sim_multiples(MAs=None,RAs=None,mags=None,Teffs=None,loggs=None,age=None,feh=0,dfeh=0.2,band='i',
                  Mdist=None,fehdist=None,distances=None,n_resample_colormatch=None,
                  fB1=0.5,fB2=0.25,minq=0.1,return_indices=False,verbose=True,allmags=None,magcorr=None,
                  return_probs=False,bands = ['g','r','i','z','J','H','Ks','Kepler'],n=1e5,P=None,dm=None,
                  stellarmodels=MODELS,
                  debug=False,obsmags=None,colors=['gr','JK'],starfieldname=None,colortol=0.1,
                  minm=0.07,multmassfn='default',only_heb=False):
    """fB here is like a 'bifurcation fraction'; magAs an absolute magnitude

    if 'distances' is set, then distancemodulus added to magcorr
    """
    models = stellarmodels
    if allmags is None:
        allmags = {}

    if MAs is None:
        M = None
        dM = None
    elif size(MAs) == 1:
        M = MAs
        dM = dm
        if dm is not None:
            MAs = MAs + rand.normal(size=n)*dm
        else:
            MAs = ones(n)*MAs
    else:
        M=None
        dM=None

    if distances is not None:
        if magcorr is None:
            magcorr = {b:su.distancemodulus(distances) for b in bands}
        else:
            magcorr = {b:magcorr[b]+su.distancemodulus(distances) for b in bands}

    #override any other MAs if Mdist is provided
    if Mdist is not None:
        MAs = Mdist.resample(n)


    #if feh is None:
    #    feh = -0.1


    if obsmags is None:
        if MAs is None:
            raise ValueError('If you do not give obsmags, you have to specify MAs and vice versa')
        mult,simkeywords = multiple_population(MAs,fB1=fB1,fB2=fB2,P=P,multmassfn=multmassfn,return_keys=True,minq=minq,minm=minm)
        if feh is None:
            feh = rand.normal(size=n)*0.2 - 0.1
            print 'using default metallicity distribution (-0.1 +/- 0.2)'
        #what if age is None?
        if len(feh) > 1:
            fehs = feh
        if len(age) > 1:
            ages = age
    else:
        mult,ages,fehs,simkeywords = colormatch_multiples(obsmags,colors,n=n,tol=colortol,M=M,dM=dM,feh=feh,dfeh=dfeh,return_keys=True,
                                                          return_all=True,minm=minm,starfieldname=starfieldname,multmassfn=multmassfn,
                                                          stellarmodels=models,Mdist=Mdist,fehdist=fehdist)

        MAs = mult.MA

    if P is not None:
        simkeywords['P'] = P


    if ages is None:
        maxages = 10**su.PADOVAMAXAGE(MAs)
        ages = rand.random(size(MAs))*(maxages*0.95) #max age is 95% of total lifetime
        ages = log10(ages)
    elif size(ages)==1:
        ages = ones(n)*ages


    if RAs is None:
        RAs = su.model_R(MAs,ages,fehs,models=models)
    RBs = su.model_R(mult.MB,ages,fehs,models=models)
    RCs = su.model_R(mult.MC,ages,fehs,models=models)
    if Teffs is None:
        TeffAs = su.model_Teff(MAs,ages,fehs,models=models)
    else:
        TeffAs = Teffs
    TeffBs = su.model_Teff(mult.MB,ages,fehs,models=models)
    TeffCs = su.model_Teff(mult.MC,ages,fehs,models=models)
    if loggs is None:
        loggAs = log10(G*MAs*MSUN/(RAs*RSUN)**2)
    else:
        loggAs = loggs
    loggBs = log10(G*mult.MB*MSUN/(RBs*RSUN)**2)
    loggCs = log10(G*mult.MC*MSUN/(RCs*RSUN)**2)
    u1As,u2As = tr.ldcoeffs(TeffAs,loggAs)
    u1Bs,u2Bs = tr.ldcoeffs(TeffBs,loggBs)
    u1Cs,u2Cs = tr.ldcoeffs(TeffCs,loggCs)

    magAs,magBs,magCs = ({},{},{})
    FAs,FBs,FCs = ({},{},{})
    
    for b in bands:
        if b in allmags:
            magAs[b] = allmags[b]
        else:
            magAs[b] = su.model_mag(b,mult.MA,ages,fehs,models=models)
        magBs[b] = su.model_mag(b,mult.MB,ages,fehs,models=models)
        magCs[b] = su.model_mag(b,mult.MC,ages,fehs,models=models)
        if magcorr is not None:  #correct magnitudes for distance, extinction, etc.
            magAs[b] += magcorr[b]
            magBs[b] += magcorr[b]
            magCs[b] += magcorr[b]
        FAs[b] = 10**(-0.4*magAs[b])
        FBs[b] = 10**(-0.4*magBs[b])
        FCs[b] = 10**(-0.4*magCs[b])
 
    #if magAs is None:
    #    magAs = su.model_mag(band,mult.MA,ages,fehs,models=MODELS)
    #magBs = su.model_mag(band,mult.MB,ages,fehs,models=MODELS)
    #magCs = su.model_mag(band,mult.MC,ages,fehs,models=MODELS)

    if len(mult) < n:
        if n_resample_colormatch is not None:
            n = n_resample_colormatch
        inds = rand.randint(len(mult),size=n)
        mult = mult[inds]
        ages = ages[inds]
        fehs = fehs[inds]
        RAs = RAs[inds]
        RBs = RBs[inds]
        RCs = RCs[inds]
        TeffAs = TeffAs[inds]
        TeffBs = TeffBs[inds]
        TeffCs = TeffCs[inds]
        loggAs = loggAs[inds]
        loggBs = loggBs[inds]
        loggCs = loggCs[inds]
        u1As = u1As[inds]
        u2As = u2As[inds]
        u1Bs = u1Bs[inds]
        u2Bs = u2Bs[inds]
        u1Cs = u1Cs[inds]
        u2Cs = u2Cs[inds]
        for b in bands:
            magAs[b] = magAs[b][inds]
            magBs[b] = magBs[b][inds]
            magCs[b] = magCs[b][inds]
            FAs[b] = FAs[b][inds]
            FBs[b] = FBs[b][inds]
            FCs[b] = FCs[b][inds]
        print 'resampled to test {} systems.'.format(n)

    n = len(mult)

    single = (mult.N==1)
    wsingle = where(single)

    binary = (mult.N==2)
    wbin = where(binary)

    #print binary.sum(),'binaries'

    if not only_heb:

        if P is not None:
            bin_mininc = tr.minimum_inclination(P,mult.MA[wbin],mult.MB[wbin],RAs[wbin],RBs[wbin])
        else:
            bin_mininc = None

        #debug mass ratios...
        if debug:
            p.figure()
            p.hist(mult.MB[wbin]/mult.MA[wbin],histtype='step',lw=3,label='all')
            p.xlabel('MB/MA (before)')

            a_s = su.semimajor(mult.P1[wbin],mult.MA[wbin]+mult.MB[wbin])
            R_s = RAs[wbin]+RBs[wbin]
            ec_probs = R_s*RSUN/(a_s*AU)
            w1 = where(mult.MB[wbin]/mult.MA[wbin] < 0.5)
            w2 = where(mult.MB[wbin]/mult.MA[wbin] > 0.5)
            p.figure()
            p.hist(log10(ec_probs[w1]),histtype='step',lw=3,normed=True,label='q<0.5',bins=linspace(-3,1,20))
            p.hist(log10(ec_probs[w2]),histtype='step',lw=3,normed=True,label='q>0.5',bins=linspace(-3,1,20))
            p.legend()
            p.xlabel('log(p_ecl) [EBs]')

            wclose = where(mult.P1[wbin] < 100)
            p.figure()
            p.hist(mult.MB[wbin][wclose]/mult.MA[wbin][wclose],histtype='step',lw=3,normed=True)
            p.xlabel('MB/MA for P<100 day')

        ###

        bininds,EBs = eclipse_population(mult.MA[wbin],mult.MB[wbin],RAs[wbin],RBs[wbin],
                                         magAs[band][wbin],magBs[band][wbin],Ps=mult.P1[wbin],
                                         return_indices=True,period=P,mininc=bin_mininc,band=band,
                                         u11s=u1As[wbin],u21s=u2As[wbin],u12s=u1Bs[wbin],u22s=u2Bs[wbin],
                                         verbose=verbose,debug=debug)

        #more debugging
        if debug:
            pass
            #print EBs.names
            #p.figure()
            #wok = where((EBs.dsec > 0) & (EBs.dpri > 0))
            #p.hist(log10(EBs.dsec[wok]/EBs.dpri[wok]),histtype='step',lw=3,bins=linspace(-3,2,20))
            #p.xlabel('log(sec/pri)')
        ###

        binprob = EBs.keywords['prob']
        dbinprob = EBs.keywords['dprob']

    triple = (mult.N==3)
    #print triple.sum(),'triples'
    wtrip = where(triple)
    CwA = (triple & (mult.which_eclipse=='A'))
    CwB = (triple & (mult.which_eclipse=='B'))

    tripM1s = (mult.MA*CwA + mult.MB*CwB)[wtrip]
    tripM2s = mult.MC[wtrip]
    tripR1s = (RAs*CwA + RBs*CwB)[wtrip]
    tripR2s = RCs[wtrip]
    tripPs = mult.P2[wtrip]
    tripmag1s = (magAs[band]*CwA + magBs[band]*CwB)[wtrip]
    tripmag2s = magCs[band][wtrip]
    tripu11s = (u1As*CwA + u1Bs*CwB)[wtrip]
    tripu21s = (u2As*CwA + u2Bs*CwB)[wtrip]
    tripu12s,tripu22s = (u1Cs[wtrip],u2Cs[wtrip])

    if P is not None:
        trip_mininc = tr.minimum_inclination(P,tripM1s,tripM2s,tripR1s,tripR2s)
    else:
        trip_mininc = None


    if debug:
        a_s = su.semimajor(tripPs,tripM1s+tripM2s)
        R_s = tripR1s+tripR2s
        ec_probs = R_s*RSUN/(a_s*AU)

        w1 = where(tripM2s/tripM1s < 0.5)
        w2 = where(tripM2s/tripM1s > 0.5)

        p.figure()
        p.hist(log10(ec_probs[w1]),histtype='step',lw=3,normed=True,label='q < 0.5')
        p.hist(log10(ec_probs[w2]),histtype='step',lw=3,normed=True,label='q > 0.5')
        p.xlabel('log(p_ecl) [HEBs]')
        p.legend()

        

    tripinds,HEBs = eclipse_population(tripM1s,tripM2s,tripR1s,tripR2s,tripmag1s,tripmag2s,Ps=tripPs,return_indices=True,period=P,mininc=trip_mininc,band=band,u11s=tripu11s,u12s=tripu12s,u21s=tripu21s,u22s=tripu22s,verbose=verbose)

    tripprob = HEBs.keywords['prob']
    dtripprob = HEBs.keywords['dprob']

    if only_heb:
        binprob,dbinprob = (1,1)
        ebprob,debprob = (1,1)

    prob_binary,dprob_binary = (binary.sum()/float(n),sqrt(binary.sum())/n)
    prob_triple,dprob_triple = (triple.sum()/float(n),sqrt(triple.sum())/n)
    ebprob = binprob*prob_binary
    debprob = ebprob*sqrt((dbinprob/binprob)**2 + (dprob_binary/prob_binary)**2)
    hebprob = tripprob*prob_triple
    dhebprob = hebprob*sqrt((dtripprob/tripprob)**2 + (dprob_triple/prob_triple)**2)

    nonecl_binaryprob = prob_binary - ebprob
    dnonecl_binaryprob = sqrt(dprob_binary**2 + debprob**2)
    nonecl_tripleprob = prob_triple - hebprob
    dnonecl_tripleprob = sqrt(dprob_triple**2 + dhebprob**2)

    #EBs.keywords['prob'] = ebprob
    #EBs.keywords['dprob'] = debprob
    #HEBs.keywords['prob'] = hebprob
    #HEBs.keywords['dprob'] = dhebprob

    if return_probs:
        return (nonecl_binaryprob,dnonecl_binaryprob),(nonecl_tripleprob,dnonecl_tripleprob),(ebprob,debprob),(hebprob,dhebprob)

    #make binary tables

    if not only_heb:
        EBs.add_column('age',ages[wbin][bininds])
        EBs.add_column('MA',mult.MA[wbin][bininds])
        EBs.add_column('MB',mult.MB[wbin][bininds])
        EBs.add_column('RA',RAs[wbin][bininds])
        EBs.add_column('RB',RBs[wbin][bininds])
        EBs.add_column('logg',loggAs[wbin][bininds])
        EBs.add_column('Teff',TeffAs[wbin][bininds])
        EBs.add_column('feh',fehs[wbin][bininds])
        for b in bands:
            EBs.add_column('%s_A' % b,magAs[b][wbin][bininds])
            EBs.add_column('%s_B' % b,magBs[b][wbin][bininds])
            EBs.add_column('%s_tot' % b,-2.5*log10(FAs[b][wbin][bininds] + FBs[b][wbin][bininds]))

        tot = FAs[band][wbin][bininds]+FBs[band][wbin][bininds]
        EBs.add_column('%s_fluxfrac_A' % band,FAs[band][wbin][bininds]/tot)
        EBs.add_column('%s_fluxfrac_B' % band,FBs[band][wbin][bininds]/tot)
        #EBs.add_column('magA',magAs[wbin][bininds])
        #EBs.add_column('magB',magBs[wbin][bininds])

        #simulate positions,RVs of binaries
        orbpop = ou.OrbitPopulation(EBs.MA,EBs.MB,EBs.P,EBs.ecc)
        EBs.add_column('rad',orbpop.Rsky)
        EBs.add_column('RV',orbpop.RVs)
        EBs.add_column('Manomaly',orbpop.Ms)
        EBs.add_column('obsx',orbpop.obspos.x)
        EBs.add_column('obsy',orbpop.obspos.y)
        EBs.add_column('obsz',orbpop.obspos.z)

    #rad,rv = su.sim_binary_orbits(EBs.MA,EBs.MB,EBs.P,EBs.ecc)
    #EBs.add_column('rad',rad)
    #EBs.add_column('RV',rv)

    #make triple tables


    FA = 10**(-0.4*magAs[band][wtrip][tripinds])
    FB = 10**(-0.4*magBs[band][wtrip][tripinds])
    FC = 10**(-0.4*magCs[band][wtrip][tripinds])

    F_pri = FA*CwB[wtrip][tripinds] + FB*CwA[wtrip][tripinds]
    F_EB = (FB+FC)*CwB[wtrip][tripinds] + (FA+FC)*CwA[wtrip][tripinds]
    dilution_factor = (F_EB/(F_pri+F_EB))
    HEBs['dpri'] *= dilution_factor
    HEBs['dsec'] *= dilution_factor
    HEBs.add_column('dilution_factor',dilution_factor)
    HEBs.add_column('MA',mult.MA[wtrip][tripinds])
    HEBs.add_column('MB',mult.MB[wtrip][tripinds])
    HEBs.add_column('MC',mult.MC[wtrip][tripinds])
    HEBs.add_column('RA',RAs[wtrip][tripinds])
    HEBs.add_column('RB',RBs[wtrip][tripinds])
    HEBs.add_column('RC',RCs[wtrip][tripinds])
    HEBs.add_column('logg',loggAs[wtrip][tripinds])
    HEBs.add_column('Teff',TeffAs[wtrip][tripinds])
    HEBs.add_column('feh',fehs[wtrip][tripinds])
    for b in bands:
        FA = 10**(-0.4*magAs[b][wtrip][tripinds])
        FB = 10**(-0.4*magBs[b][wtrip][tripinds])
        FC = 10**(-0.4*magCs[b][wtrip][tripinds])
        F_pri = FA*CwB[wtrip][tripinds] + FB*CwA[wtrip][tripinds]
        F_EB = (FB+FC)*CwB[wtrip][tripinds] + (FA+FC)*CwA[wtrip][tripinds]
        dilution_factor = (F_EB/(F_pri+F_EB))
        HEBs.add_column('%s_dilution_factor' % b,dilution_factor)
        HEBs.add_column('%s_A' % b,magAs[b][wtrip][tripinds])
        HEBs.add_column('%s_B' % b,magBs[b][wtrip][tripinds])
        HEBs.add_column('%s_C' % b,magCs[b][wtrip][tripinds])
        HEBs.add_column('%s_tot' % b,-2.5*log10(FAs[b][wtrip][tripinds] + FBs[b][wtrip][tripinds] + 
                                                FAs[b][wtrip][tripinds]))
    tot = FAs[b][wtrip][tripinds]+FBs[b][wtrip][tripinds]+FCs[b][wtrip][tripinds]
    HEBs.add_column('%s_fluxfrac_A' % band,FAs[band][wtrip][tripinds]/tot)
    HEBs.add_column('%s_fluxfrac_B' % band,FBs[band][wtrip][tripinds]/tot)
    HEBs.add_column('%s_fluxfrac_C' % band,FCs[band][wtrip][tripinds]/tot)
    HEBs.add_column('age',ages[wtrip][tripinds])
        
    #HEBs.add_column('magA',magAs[wtrip][tripinds])
    #HEBs.add_column('magB',magBs[wtrip][tripinds])
    #HEBs.add_column('magC',magCs[wtrip][tripinds])
    HEBs.add_column('which_eclipse',mult.which_eclipse[wtrip][tripinds])

    #simulate positions, RVs
    if P is None:
        #rad,rv = su.sim_binary_orbits(HEBs.MA,tripM1s[tripinds],mult.P1[wtrip][tripinds]) #no ecc specified
        #HEBs.add_column('Plong',mult.P1[wtrip][tripinds])

        ecclong = su.draw_eccs(len(HEBs.MA),HEBs.Plong,maxecc=0.97)
        ##orbpop = ou.OrbitPopulation(HEBs.MA,tripM1s[tripinds],mult.P1[wtrip][tripinds],ecclong)
        #M1s = tripM1s[tripinds] + tripM2s[tripinds]
        #M2s = (mult.MA*CwB + mult.MB*CwA)[wtrip][tripinds]
        #orbpop = ou.OrbitPopulation(M1s,M2s,mult.P1[wtrip][tripinds],ecclong)
        #rad,rv_long = (orbpop.Rsky,orbpop.RVs)
    else:
        #simulate outper period here if inner period is fixed
        Plong = 10**su.LOGPERKDE.draw(len(HEBs.MA))
        ecclong = su.draw_eccs(len(Plong),Plong,maxecc=0.97)
        bad = Plong < P
        nbad = bad.sum()
        while nbad > 0:
            Plong[where(bad)] = 10**su.LOGPERKDE.draw(nbad)
            bad = Plong < P
            nbad = bad.sum()
        #rad,rv = su.sim_binary_orbits(HEBs.MA,tripM1s[tripinds],Plong) #no ecc specified
        #HEBs.add_column('Plong',Plong)

        ##orbpop = ou.OrbitPopulation(HEBs.MA,tripM1s[tripinds],mult.P1[wtrip][tripinds],ecclong)
        #M1s = tripM1s[tripinds] + tripM2s[tripinds]
        #M2s = (mult.MA*CwB + mult.MB*CwA)[wtrip][tripinds]
        #orbpop = ou.OrbitPopulation(M1s,M2s,mult.P1[wtrip][tripinds],ecclong)

        #rad,rv_long = (orbpop.Rsky,orbpop.RVs)

    M1s = (mult.MA*CwB + mult.MB*CwA)[wtrip][tripinds]
    M2s = (mult.MB*CwB + mult.MA*CwA)[wtrip][tripinds]
    M3s = mult.MC[wtrip][tripinds]
    
    trip_orbpop = ou.TripleOrbitPopulation(M1s,M2s,M3s,Plong,HEBs.P,
                                           ecclong=ecclong,eccshort=HEBs.ecc)

    # set Radial velocities relative to COM 
    RV_A = trip_orbpop.RV_1*CwB[wtrip][tripinds] + trip_orbpop.RV_2*CwA[wtrip][tripinds]
    RV_B = trip_orbpop.RV_1*CwA[wtrip][tripinds] + trip_orbpop.RV_2*CwB[wtrip][tripinds]
    RV_C = trip_orbpop.RV_3

    HEBs.add_column('Plong',Plong)
    HEBs.add_column('ecclong',ecclong)
    HEBs.add_column('rad',trip_orbpop.orbpop_long.Rsky)
    HEBs.add_column('RV',RV_A - RV_B)
    
    HEBs.add_column('Manomaly_long',trip_orbpop.orbpop_long.Ms)
    HEBs.add_column('obsx_long',trip_orbpop.orbpop_long.obspos.x)
    HEBs.add_column('obsy_long',trip_orbpop.orbpop_long.obspos.y)
    HEBs.add_column('obsz_long',trip_orbpop.orbpop_long.obspos.z)
    
    HEBs.add_column('Manomaly_short',trip_orbpop.orbpop_short.Ms)
    HEBs.add_column('obsx_short',trip_orbpop.orbpop_short.obspos.x)
    HEBs.add_column('obsy_short',trip_orbpop.orbpop_short.obspos.y)
    HEBs.add_column('obsz_short',trip_orbpop.orbpop_short.obspos.z)
    
                                             
    #HEBs.add_column('orb_M1',M1s)
    #HEBs.add_column('orb_M2',M2s)
    #HEBs.add_column('ecclong',ecclong)
    #HEBs.add_column('rad',rad)
    #HEBs.add_column('RV',rv_long)  #call this RV_relative?
    #HEBs.add_column('Manomaly',orbpop.Ms)
    #HEBs.add_column('obsx',orbpop.obspos.x)
    #HEBs.add_column('obsy',orbpop.obspos.y)
    #HEBs.add_column('obsz',orbpop.obspos.z)

    if not only_heb:
        is_noneclipsing_binary = ones(n)
        is_noneclipsing_binary[wsingle] = 0
        is_noneclipsing_binary[wbin[0][bininds]] = 0
        is_noneclipsing_binary[wtrip] = 0
        w_nonecl_binary = where(is_noneclipsing_binary)

    if not only_heb:
        nonecl_binaries = atpy.Table()
        nonecl_binaries.add_column('MA',mult.MA[w_nonecl_binary])
        nonecl_binaries.add_column('MB',mult.MB[w_nonecl_binary])
        nonecl_binaries.add_column('RA',RAs[w_nonecl_binary])
        nonecl_binaries.add_column('RB',RBs[w_nonecl_binary])
        nonecl_binaries.add_column('logg',loggAs[w_nonecl_binary])
        nonecl_binaries.add_column('Teff',TeffAs[w_nonecl_binary])
        nonecl_binaries.add_column('feh',fehs[w_nonecl_binary])
        for b in bands:
            nonecl_binaries.add_column('%s_A' % b,magAs[b][w_nonecl_binary])
            nonecl_binaries.add_column('%s_B' % b,magBs[b][w_nonecl_binary])
            nonecl_binaries.add_column('%s_tot' % b,-2.5*log10(FAs[b][w_nonecl_binary] + FBs[b][w_nonecl_binary]))
        tot =  FAs[band][w_nonecl_binary]+FBs[band][w_nonecl_binary]
        nonecl_binaries.add_column('%s_fluxfrac_A' % band,FAs[band][w_nonecl_binary]/tot)
        nonecl_binaries.add_column('%s_fluxfrac_B' % band,FBs[band][w_nonecl_binary]/tot)
        nonecl_binaries.add_column('age',ages[w_nonecl_binary])

        nonecl_binaries.add_keyword('prob',nonecl_binaryprob)
        nonecl_binaries.add_keyword('dprob',dnonecl_binaryprob)
        #nonecl_binaries.add_column('magA',magAs[w_nonecl_binary])
        #nonecl_binaries.add_column('magB',magBs[w_nonecl_binary])

        is_HT = ones(n)
        is_HT[wsingle] = 0
        is_HT[wbin] = 0
        is_HT[wtrip[0][tripinds]] = 0
        w_HT = where(is_HT)

        HTs = atpy.Table()
        HTs.add_column('MA',mult.MA[w_HT])
        HTs.add_column('MB',mult.MB[w_HT])
        HTs.add_column('MC',mult.MC[w_HT])
        HTs.add_column('RA',RAs[w_HT])
        HTs.add_column('RB',RBs[w_HT])
        HTs.add_column('RC',RCs[w_HT])
        HTs.add_column('logg',loggAs[w_HT])
        HTs.add_column('Teff',TeffAs[w_HT])
        HTs.add_column('feh',fehs[w_HT])
        for b in bands:
            HTs.add_column('%s_A' % b,magAs[b][w_HT])
            HTs.add_column('%s_B' % b,magBs[b][w_HT])
            HTs.add_column('%s_C' % b,magCs[b][w_HT])
            HTs.add_column('%s_tot' % b,-2.5*log10(FAs[b][w_HT] + FBs[b][w_HT] + FCs[b][w_HT]))
        tot =  FAs[band][w_HT]+FBs[band][w_HT]+FCs[band][w_HT]
        HTs.add_column('%s_fluxfrac_A' % band, FAs[band][w_HT]/tot)
        HTs.add_column('%s_fluxfrac_B' % band, FBs[band][w_HT]/tot)
        HTs.add_column('%s_fluxfrac_C' % band, FCs[band][w_HT]/tot)
        HTs.add_column('age',ages[w_HT])

        HTs.add_keyword('prob',nonecl_tripleprob)
        HTs.add_keyword('dprob',dnonecl_tripleprob)

    if not only_heb:
        pops = [EBs,HEBs,nonecl_binaries,HTs]
    else:
        pops = [HEBs]

    for pop in pops:
        for kw,val in simkeywords.iteritems():
            pop.add_keyword(kw,val)
    


    if return_indices:
        if not only_heb:
            return (wsingle,w_nonecl_binary,w_HT,wbin[0][bininds],wtrip[0][tripinds]),(nonecl_binaries,HTs,EBs,HEBs)
        else:
            return (None,None,None,None,wtrip[0]),(None,None,None,HEBs)
    else:
        if not only_heb:
            return nonecl_binaries,HTs,EBs,HEBs
        else:
            return None,None,None,HEBs

def colormatch_multiples(mags,colors=['gr','JK'],n=1e5,tol=0.1,feh=None,dfeh=None,age=None,plot=False,
                         Mdist=None,fehdist=None,
                         N=None,minm=0.07,starfieldname=None,stellarmodels=MODELS,minq=0.1,
                         M=None,dM=None,return_all=False,maxage=0.95,return_keys=False,
                         return_fractions=False,Teff=None,logg=None,multmassfn='default',
                         minmatches=10,dmags=None):
    models = stellarmodels
    if starfieldname is not None:
        starfile = '%s/%s.fits' % (TRILEGALDIR,starfieldname)
        try:
            starfield = atpy.Table(starfile,verbose=False)
        except:
            trilegal2fits(name,dir=TRILEGALDIR)
            starfield = atpy.Table(starfile,verbose=False)
    else:
        starfield = None

    if dfeh is None:
        dfeh = 0.1

    using_starfield = False
    if M is None and Mdist is None:
        if starfield is None:
            MAs = su.IMFdraw(n,minm=minm,single=True)
            if feh is None:
                feh = -0.05
            print 'drawing primary masses from IMF, feh=%.2f +/- %.2f' % (feh,dfeh)
            fehs = rand.normal(size=n)*dfeh + feh
        else:
            using_starfield = True
            ds = su.dfromdm(starfield.mM0)
            w = where(ds < 1500)
            inds = rand.randint(size(w),size=n)
            #print starfield.names
            MAs = starfield.Mact[w][inds]
            print 'primary masses chosen according to PDMF of TRILEGAL starfield'
            ages = starfield.logAge[w][inds]
            if feh is None:
                fehs = starfield.MH[w][inds]
                print '[Fe/H] chosen according to given starfield.'
            else:
                print 'metallicities [Fe/H] = %.2f +/- %.2f' % (feh,dfeh)
                fehs = rand.normal(size=n)*dfeh + feh
    else:
        if Mdist is not None:
            print 'drawing primary masses according to provided mass distribution: %s' % Mdist
            MAs = Mdist.resample(n)
        else:
            MAs = M + rand.normal(size=n)*dM
            print 'drawing primary masses according to %.2f +/- %.2f' % (M,dM)
        if fehdist is not None:
            fehs = fehdist.resample(n)
            print 'metallicities [Fe/H] according to provided distribution: %s' % fehdist
        else:
            if feh is None:
                feh = -0.05
            if isnan(feh):
                print 'given [Fe/H] is nan: using default [Fe/H]'
                feh = -0.05
            print 'metallicities [Fe/H] = %.2f +/- %.2f' % (feh,dfeh)
            fehs = rand.normal(size=n)*dfeh + feh


    #set keyword values important for simulation
    simkeywords = {}
    if using_starfield:
        simkeywords['STARFIELD'] = starfieldname
    else:
        if Mdist is not None:
            simkeywords['M'] = Mdist.mu
            try:
                simkeywords['DM_P'] = Mdist.sighi
                simkeywords['DM_N'] = Mdist.siglo
            except AttributeError:
                simkeywords['DM_P'] = Mdist.sig
                simkeywords['DM_N'] = Mdist.sig                
        else:
            if M is not None:
                simkeywords['M'] = M
                simkeywords['DM_P'] = dM
                simkeywords['DM_N'] = dM

        if fehdist is not None:
            simkeywords['FEH'] = fehdist.mu
            try:
                simkeywords['DFEH_P'] = fehdist.sighi
                simkeywords['DFEH_N'] = fehdist.siglo
            except AttributeError:
                simkeywords['DFEH_P'] = fehdist.sig
                simkeywords['DFEH_N'] = fehdist.sig
        else:
            simkeywords['FEH'] = feh
            simkeywords['DFEH_P'] = dfeh
            simkeywords['DFEH_N'] = dfeh

    if colors is None:
        colors = []
    else:
        simkeywords['COLORTOL'] = 0.1
            
    for c in colors:
        m = re.search('^(\w)(\w)$',c)
        if m:
            b1 = m.group(1)
            b2 = m.group(2)
            simkeywords['%s-%s' % (b1,b2)] = mags[b1]-mags[b2]
    simkeywords['MINM'] = minm
    simkeywords['MULTMASS'] = multmassfn

    if N is None:
        mult = multiple_population(MAs,multmassfn=multmassfn,minq=minq)
    if N==2:
        print 'simulating a binary population'
        mult = multiple_population(MAs,fB1=1.,fB2=0.,multmassfn=multmassfn,minq=minq)
    if N==3:
        print 'simulating a triple population'
        mult = multiple_population(MAs,fB1=1.,fB2=1.,multmassfn=multmassfn,minq=minq)        

    if age is None:
        maxages = 10**su.PADOVAMAXAGE(mult.MA)
        age = rand.random(size(mult.MA))*(maxages*maxage) #max age is e.g. 90% of total lifetime
        ages = log10(age)
    elif size(age)==1:
        ages = ones(len(mult.MA))*age
    else:
        ages = age



    cond = ones(n).astype(bool)
    for c in colors:
        m = re.search('^(\w)(\w)$',c)
        if m:
            b1 = m.group(1)
            b2 = m.group(2)

            if isnan(mags[b1]) or isnan(mags[b2]):
                print 'color %s not used (%s and/or %s band is/are nan)' % (c,b1,b2)
                continue

            FA1 = 10**(-0.4*(su.model_mag(b1,mult.MA,age=ages,feh=fehs,models=models)))
            FB1 = 10**(-0.4*(su.model_mag(b1,mult.MB,age=ages,feh=fehs,models=models)))
            FB1[where(mult.MB==0)] = 0
            if N==3 or N is None:
                FC1 = 10**(-0.4*(su.model_mag(b1,mult.MC,age=ages,feh=fehs,models=models)))
                FC1[where(mult.MC==0)] = 0
                mag1 = -2.5*log10(FA1+FB1+FC1)
            else:
                mag1 = -2.5*log10(FA1+FB1)

            FA2 = 10**(-0.4*(su.model_mag(b2,mult.MA,age=ages,feh=fehs,models=models)))
            FB2 = 10**(-0.4*(su.model_mag(b2,mult.MB,age=ages,feh=fehs,models=models)))
            FB2[where(mult.MB==0)] = 0
            if N==3 or N is None:
                FC2 = 10**(-0.4*(su.model_mag(b2,mult.MC,age=ages,feh=fehs,models=models)))
                FC2[where(mult.MC==0)] = 0
                mag2 = -2.5*log10(FA2+FB2+FC2)
            else:
                mag2 = -2.5*log10(FA2+FB2)


            modcolor = mag1-mag2
            obscolor = mags[b1]-mags[b2]

            wnan = where(isnan(modcolor))
            #print 'MB',mult.MB[wnan]
            #print 'FB1',FB1[wnan]
            #print 'MC',mult.MC[wnan]
            #print 'FC1',FC1[wnan]
                        

            wok = where(~isnan(modcolor))

            cmatch = absolute(modcolor - obscolor) < tol
            matchbin = cmatch & (mult.N==2)
            matchtrip = cmatch & (mult.N==3)
            print 'matching color: %s ~ %.2f (%i matches--%i binary, %i triple)' % (c,obscolor,cmatch.sum(),
                                                                                    matchbin.sum(),matchtrip.sum())  

            cond &= cmatch
            totbin = cond & (mult.N==2)
            tottrip = cond & (mult.N==3)

            if c == colors[-1]:
                print '%i total matches (%i binary, %i triple)' % (cond.sum(),totbin.sum(),tottrip.sum())

        else:
            raise ValueError('unrecognized color: %s' % color)

    if Teff is not None:
        T,dT = Teff
        Teffs = su.model_Teff(mult.MA,ages,feh=fehs,models=models)
        cond &= absolute(Teffs-T) < (3*dT)

    if logg is not None:
        g,dg = logg
        loggs = su.model_logg(mult.MA,ages,feh=fehs,models=models)
        cond &= absolute(loggs-g) < (3*dg)
        
    wmatch = where(cond)

    if cond.sum() < minmatches:
        raise BadColorsError('Colors (%s, within %.2f) allow only %i matches. Maybe drop some or increase colortol?' %\
                                 (colors,tol,cond.sum()))

    if plot:
        p.figure()
        p.hist(mult.MA[wmatch],lw=3,histtype='step',bins=linspace(0,3,30))
        p.hist(mult.MB[wmatch],lw=3,histtype='step',bins=linspace(0,3,30))
        p.hist(mult.MC[wmatch],lw=3,histtype='step',bins=linspace(0,3,30))


    if return_fractions:
        nbin = (cond & (mult.N==2)).sum()
        ntrip = (cond & (mult.N==3)).sum()
        ntot = cond.sum()
        binfrac,dbinfrac = (float(nbin)/ntot,sqrt(nbin)/ntot)
        tripfrac,dtripfrac = (float(ntrip)/ntot,sqrt(ntrip)/ntot)
        return (binfrac,dbinfrac),(tripfrac,dtripfrac)
    if return_all:
        if return_keys:
            return mult[wmatch],ages[wmatch],fehs[wmatch],simkeywords
        else:
            return mult[wmatch],ages[wmatch],fehs[wmatch]
    else:
        if return_keys:
            return mult[wmatch],simkeywords
        else:
            return mult[wmatch]


#custom exceptions

class BadColorsError(Exception):
    pass

class EmptyPopulationError(Exception):
    pass
