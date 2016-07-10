"""Dependencies:  astropysics,progressbar,atpy,emcee
"""

import fpsim
import os,sys,re,os.path,shutil,fnmatch
import time
from numpy import *
import numpy as np
import transit_basic as tr
from progressbar import Percentage,Bar,RotatingMarker,ETA,ProgressBar
import atpy
import pylab as p
from astropysics.coords import FK5Coordinates
import koiutils as ku
import numpy.random as rand
from consts import *
from matplotlib import mlab
from numpy import ma

import plotutils as pu
import pickle

KEPLERDIR = os.environ['KEPLERDIR']

FIELDSDIR = os.environ['TRILEGALDIR']
FPMODELSDIR = '%s/FPP/models' % KEPLERDIR 

BANDS = ['g','r','i','z','J','H','Ks','Kepler']


def write_fitebs_script(kois,outfile='%s/master_script' % FPMODELSDIR,
                        fitallebs='/disk/arden1/tdm/Dropbox/FPP_morton/bin//fitallebs'):
    file = open(outfile,'w')
    file.write('#!/bin/tcsh\n')
    file.write('setenv FPMODELSDIR %s\n' % FPMODELSDIR)
    file.write('setenv FITALLEBS %s\n' % fitallebs)
    for koi in kois:
        file.write('cd $FPMODELSDIR/KOI%.2f\n' % (koi))
        file.write('$FITALLEBS .\n')
    file.close()
        

def makemodels_list(kois,verbose=True,overwrite=False,raise_exceptions=False,**kwargs):
    for koi in kois:
        print 'KOI %.2f:' % koi
        try:
            makemodels(koi,verbose=verbose,overwrite=overwrite,**kwargs)
        except KeyboardInterrupt:
            raise
        except:
            print 'error with KOI %.2f' % (koi)
            if raise_exceptions:
                raise

def koimakemodels(koi,P=None,M=None,R=None,dM=None,dR=None,age=None,feh=None,dfeh=0.15,
               alpha=-2,rbin=None,n=2e4,u1=None,u2=None,resimbebs=False,minq=0.1,
               overwrite=False,verbose=False,colors=['gr','JK'],colortol=0.1,
               spectrum=False,exactstar=False,starfield=True,multmassfn='default',
               models=['eb','heb','beb','bgpl','pl'],tag=None,stellarmodels='padova',obsmags=None,
               **kwargs):
    """Builds all the models for a given KOI.
    """

    if P is None:
        P = ku.period(koi)
    if M is None:
        try:
            M,dM = ku.Mstar(koi,error=True)
        except:
            M = ku.Mstar(koi)
            dM = M*0.2
    if R is None:
        try:
            R,dR = ku.Rstar(koi,error=True)
        except:
            R = ku.Rstar(koi)
            dR = R*0.2
            
    if exactstar:
        dM = 0
        dR = 0

    if rbin is None:   #fix this maybe?  make sure it's calculating radius ratio bin?
        rp = ku.Rplanet(koi)
        rbin = (rp,rp/3.)

    if u1 is None or u2 is None:
        Teff = ku.Teff(koi)
        logg = ku.logg(koi)
        u1,u2 = tr.ldcoeffs(Teff,logg)
        u1 = u1[0] #hack, b/c single-element array
        u2 = u2[0] #hack, b/c single-element array

    if feh is None:
        if spectrum:
            try:
                feh,dfeh = ku.feh(koi,error=True)
            except:
                feh = ku.feh(koi)
                dfeh = 0.2

    if starfield:
        ra,dec = koiradec(koi)    
        name = radec2name(ra,dec)
        starfieldname = name
    else:
        starfieldname = None        

    if obsmags is None:
        obsmags = ku.KICmags(koi)

    if tag is None:
        folder = '%s/KOI%.2f' % (FPMODELSDIR,koi)
    else:
        folder = '%s/KOI%.2f_%s' % (FPMODELSDIR,koi,tag)
    if verbose:
        print folder
    if not os.path.exists(folder):
        os.mkdir(folder)

    if 'heb' not in models and 'eb' not in models:
        print 'not making EB/HEB models.'
    elif not os.path.exists('%s/hebs.fits' % folder) or not os.path.exists('%s/ebs.fits' % folder) or overwrite:
        if verbose:
            print 'HEBs & EBs...'
        hebs,ebs = HEBs(P,M,n,dm=dM,age=age,verbose=verbose,obsmags=obsmags,colors=colors,colortol=colortol,spectrum=spectrum,
                        feh=feh,dfeh=dfeh,starfieldname=starfieldname,stellarmodels=stellarmodels,
                        multmassfn=multmassfn,minq=minq)
        hebs = firstnrows(hebs,n)
        ebs = firstnrows(ebs,n)
        hebs.write('%s/hebs.fits' % folder,overwrite=True,verbose=False)
        parfile = '%s/hebs_params.fits' % folder

        ebs.write('%s/ebs.fits' % folder,overwrite=True,verbose=False)
        parfile = '%s/ebs_params.fits' % folder
        if os.path.exists(parfile):
            os.remove(parfile)

        if os.path.exists(parfile):
            os.remove(parfile)
    else:
        if verbose:
            print '%s and %s exists; not overwriting.' % ('%s/hebs.fits' % folder,'%s/ebs.fits' % folder)
        
    if 'beb' not in models:
        print 'not making BEB models'
    elif not os.path.exists('%s/bebs.fits' % folder) or overwrite:
        if verbose:
            print 'BEBs...'
        bebs = BEBs(starfieldname,P,n,verbose=verbose,resim=resimbebs,stellarmodels=stellarmodels)
        bebs = firstnrows(bebs,n)
        bebs.write('%s/bebs.fits' % folder,overwrite=True,verbose=False)
        parfile = '%s/bebs_params.fits' % folder
        if os.path.exists(parfile):
            os.remove(parfile)
    else:
        if verbose:
            print '%s exists; not overwriting.' % ('%s/bebs.fits' % folder)
        
    if 'bgpl' not in models:
        print 'not making BGPL models'
    elif not os.path.exists('%s/bgpls.fits' % folder) or overwrite:
        if verbose:
            print 'BG planets...'
        bgpls = BGplanets(starfieldname,P,n,alpha,verbose=verbose)
        bgpls = firstnrows(bgpls,n)
        bgpls.write('%s/bgpls.fits' % folder,overwrite=True,verbose=False)
        parfile = '%s/bgpls_params.fits' % folder
        if os.path.exists(parfile):
            os.remove(parfile)
    else:
        if verbose:
            print '%s exists; not overwriting.' % ('%s/bgpls.fits' % folder)
            
    if 'pl' not in models:
        print 'not making PL models.'
    elif not os.path.exists('%s/pls.fits' % folder) or overwrite:        
        if verbose:
            print 'planets (Mstar=%.2f +/- %.2f, Rstar=%.2f +/- %.2f)...' % (M,dM,R,dR)
        #not including age variation for now...there are reasons...
        pls = planets(P,M=M,R=R,dM=dM,dR=dR,u1=u1,u2=u2,n=n,alpha=alpha,rbin=rbin,
                      verbose=verbose,stellarmodels=stellarmodels) #exactstar is always True..
        pls = firstnrows(pls,n)
        pls.write('%s/pls.fits' % folder,overwrite=True,verbose=False)
        parfile = '%s/pls_params.fits' % folder
        if os.path.exists(parfile):
            os.remove(parfile)
    else:
        if verbose:
            print '%s exists; not overwriting.' % ('%s/pls.fits' % folder)


def makemodels(name,ra=None,dec=None,P=None,rp=None,M=1,R=1,dM=None,dR=None,age=None,
               Mdist=None,Rdist=None,fehdist=None,
               feh=None,dfeh=0.15,Teff=None,logg=None,dTeff=None,dlogg=None,
               alpha=-2,n=2e4,u1=None,u2=None,resimbebs=False,minq=0.1,
               overwrite=False,verbose=False,colors=['gr','JK'],colortol=0.1,
               spectrum=False,exactstar=False,multmassfn='default',band='Kepler',
               models=['eb','heb','beb','bgpl','pl'],tag=None,stellarmodels='dartmouth',obsmags=None):
    """Builds all the models for a candidate of given name.
    """

    if ra is None or dec is None:
        ra,dec = koiradec(name)

    if dM is None:
        dM = M*0.2
    if dR is None:
        dR = R*0.2
            
    if exactstar:  #should this be removed?
        dM = 0
        dR = 0

    rbin = (rp,rp/3.)  #Earth radii

    starfieldname = radec2name(ra,dec)

    Teff,logg = ku.get_property(name,'teff','logg')

    if u1 is None or u2 is None:
        #if Teff is None or isnan(Teff):
        #    Teff = fpsim.su.model_Teff(M)
        #if logg is None or isnan(logg):
        #    logg = log10(G*M*MSUN/(R*RSUN)**2)
        u1,u2 = tr.ldcoeffs(Teff,logg)
        u1 = u1[0] #hack, b/c single-element array
        u2 = u2[0] #hack, b/c single-element array
        if verbose:
            print 'LD coeffs: %.2f, %.2f' % (u1,u2)

    if tag is None:
        folder = '%s/%s' % (FPMODELSDIR,name)
    else:
        folder = '%s/%s_%s' % (FPMODELSDIR,name,tag)
    if verbose:
        print folder
    if not os.path.exists(folder):
        os.mkdir(folder)

    if 'heb' not in models and 'eb' not in models:
        print 'not making EB/HEB models.'
    elif not os.path.exists('%s/hebs.fits' % folder) or not os.path.exists('%s/ebs.fits' % folder) or overwrite:
        if verbose:
            print 'HEBs & EBs...'
        hebs,ebs = HEBs(P,M,n,dm=dM,age=age,verbose=verbose,obsmags=obsmags,colors=colors,
                        colortol=colortol,spectrum=spectrum,Mdist=Mdist,fehdist=fehdist,
                        feh=feh,dfeh=dfeh,starfieldname=starfieldname,stellarmodels=stellarmodels,band=band,
                        multmassfn=multmassfn,minq=minq)
        hebs = firstnrows(hebs,n)
        ebs = firstnrows(ebs,n)
        fname = '%s/hebs.fits' % folder
        hebs.write(fname,overwrite=True,verbose=False)
        parfile = '%s/hebs_params.fits' % folder

        ebs.write('%s/ebs.fits' % folder,overwrite=True,verbose=False)
        parfile = '%s/ebs_params.fits' % folder
        if os.path.exists(parfile):
            os.remove(parfile)

        if os.path.exists(parfile):
            os.remove(parfile)
    else:
        if verbose:
            print '%s and %s exists; not overwriting.' % ('%s/hebs.fits' % folder,'%s/ebs.fits' % folder)
        
    if 'beb' not in models:
        print 'not making BEB models'
    elif not os.path.exists('%s/bebs.fits' % folder) or overwrite:
        if verbose:
            print 'BEBs...'
        bebs = BEBs(starfieldname,P,n,verbose=verbose,resim=resimbebs,stellarmodels=stellarmodels,band=band)
        bebs = firstnrows(bebs,n)
        bebs.write('%s/bebs.fits' % folder,overwrite=True,verbose=False)
        parfile = '%s/bebs_params.fits' % folder
        if os.path.exists(parfile):
            os.remove(parfile)
    else:
        if verbose:
            print '%s exists; not overwriting.' % ('%s/bebs.fits' % folder)
        
    if 'bgpl' not in models:
        print 'not making BGPL models'
    elif not os.path.exists('%s/bgpls.fits' % folder) or overwrite:
        if verbose:
            print 'BG planets...'
        bgpls = BGplanets(starfieldname,P,n,alpha=alpha,verbose=verbose,band=band)
        bgpls = firstnrows(bgpls,n)
        bgpls.write('%s/bgpls.fits' % folder,overwrite=True,verbose=False)
        parfile = '%s/bgpls_params.fits' % folder
        if os.path.exists(parfile):
            os.remove(parfile)
    else:
        if verbose:
            print '%s exists; not overwriting.' % ('%s/bgpls.fits' % folder)
            
    if 'pl' not in models:
        print 'not making PL models.'
    elif not os.path.exists('%s/pls.fits' % folder) or overwrite:        
        if verbose:
            #print 'planets (Mstar=%.2f +/- %.2f, Rstar=%.2f +/- %.2f)...' % (M,dM,R,dR)
            print 'planets (%s, %s)...' % (Mdist,Rdist)
        #not including age variation for now...there are reasons...
        pls = planets(P,M=M,R=R,dM=dM,dR=dR,u1=u1,u2=u2,n=n,alpha=alpha,rbin=rbin,Mdist=Mdist,Rdist=Rdist,
                      verbose=verbose,stellarmodels=stellarmodels,band=band,exactstar=exactstar) #exactstar is always True..
        pls = firstnrows(pls,n)
        print pls.keywords
        pls.write('%s/pls.fits' % folder,overwrite=True,verbose=False)
        parfile = '%s/pls_params.fits' % folder
        if os.path.exists(parfile):
            os.remove(parfile)
    else:
        if verbose:
            print '%s exists; not overwriting.' % ('%s/pls.fits' % folder)



def koiradec(koi):
    return ku.radec(koi)
    if koi==463:
        return '20:00:49.5','+45:01:05'
    if koi==1474:
        return '19:41:40.3','+51:11:05'

def radec2name(ra,dec):
    """ra,dec in decimal degrees
    """
    c = FK5Coordinates(ra,dec)
    chips,ras,decs = loadtxt('%s/chiplocs.txt' % KEPLERDIR,unpack=True)
    ds = ((c.ra.d-ras)**2 + (c.dec.d-decs)**2)
    chip = chips[argmin(ds)]
    return 'kepfield%i' % chip

def firstnrows(table,n):
    inds = arange(0,n)
    return table.rows(inds)

def BGplanets(starfieldname,P,n=2e4,verbose=True,**kwargs): #change from 'koi' to ra/dec? or (starfieldname)
    """
    kwargs: alpha (-2), stellarmodels ('padova') 
    """
    #if P is None:
    #    P = ku.period(koi)    
    #ra,dec = koiradec(koi)    
    #name = radec2name(ra,dec)

    name = starfieldname
    pls = fpsim.simbgplanets(name,P=P,verbose=verbose,**kwargs)
    npls = len(pls)
    while npls < n:
        newpls = fpsim.simbgplanets(name,P=P,verbose=verbose,**kwargs)
        pls.append(newpls)
        prob,dprob = (pls.keywords['prob'],pls.keywords['dprob'])
        newprob,newdprob = (newpls.keywords['prob'],newpls.keywords['dprob'])
        norm = (1./dprob**2 + 1./newdprob**2) 
        pls.keywords['prob'] = (prob/dprob**2 + newprob/newdprob**2)/norm
        pls.keywords['dprob'] = 1/sqrt(norm)        
        npls = len(pls)
        if verbose:
            print 'total of %i planets' % npls

    return pls

def planets(P,rbin=None,n=2e4,verbose=True,**kwargs):
    """
    P, rbin, n 
    kwargs: M,R,dM(0.1),dR(0.1),u1(0.394),u2(0.296),alpha(-2),verbose(True),exactstar(False),stellarmodels('padova')
    """
    pls = fpsim.simplanets(P,rbin=rbin,verbose=verbose,**kwargs)
    npls = len(pls)
    while npls < n:
        newpls = fpsim.simplanets(P,rbin=rbin,verbose=verbose,**kwargs)
        pls.append(newpls)
        prob,dprob = (pls.keywords['prob'],pls.keywords['dprob'])
        newprob,newdprob = (newpls.keywords['prob'],newpls.keywords['dprob'])
        norm = (1./dprob**2 + 1./newdprob**2) 
        pls.keywords['prob'] = (prob/dprob**2 + newprob/newdprob**2)/norm
        pls.keywords['dprob'] = 1/sqrt(norm)
        npls = len(pls)        
        if verbose:
            print 'total of %i planets' % npls

    return pls

def BEBs(starfieldname,P,n=2e4,resim=False,verbose=True,**kwargs): #change to ra/dec rather than koi->name? (or 'starfieldname'?)
    """
    P,n,resim
    kwargs: verbose(True),stellarmodels('padova'),band('Kepler')
    """
    kwargs['fB'] = 1

    #if P is None:
    #    P = ku.period(koi)
    #ra,dec = koiradec(koi)    
    #name = radec2name(ra,dec)

    name = starfieldname
    bebs = fpsim.simBEBs(name,P=P,resim=resim,verbose=verbose,**kwargs)
    nbebs = len(bebs)
    while nbebs < n:
        newbebs = fpsim.simBEBs(name,P=P,verbose=verbose,**kwargs)
        bebs.append(newbebs)
        prob,dprob = (bebs.keywords['prob'],bebs.keywords['dprob'])
        newprob,newdprob = (newbebs.keywords['prob'],newbebs.keywords['dprob'])
        norm = (1./dprob**2 + 1./newdprob**2) 
        bebs.keywords['prob'] = (prob/dprob**2 + newprob/newdprob**2)/norm
        bebs.keywords['dprob'] = 1/sqrt(norm)        
        nbebs = len(bebs)
        if verbose:
            print 'total of %i bebs' % nbebs

    #bebs.add_keyword('P',P)
    return bebs

def HEBs(P,m=1,n=2e4,spectrum=False,verbose=True,**kwargs):
    """and EBs too.
    m,P,n,spectrum
    kwargs: dm,age,verbose,obsmags,colors,feh,dfeh,colortol,multmassfn,starfieldname,stellarmodels,minq,band

    """    
    if not spectrum:
        m = None

    mult = fpsim.sim_multiples(m,P=P,**kwargs)
    hebs = mult[3]
    ebs = mult[2]

    nhebs = len(hebs)
    nebs = len(ebs)
    if verbose:
        print 'total of %i hebs' % nhebs
        print 'total of %i ebs' % nebs
    while nhebs < n or nebs < n:
        newmult = fpsim.sim_multiples(m,P=P,**kwargs)
        newhebs = newmult[3]
        newebs = newmult[2]
        hebs.append(newhebs)
        ebs.append(newebs)

        hebprob,hebdprob = (hebs.keywords['prob'],hebs.keywords['dprob'])
        hebnewprob,hebnewdprob = (newhebs.keywords['prob'],newhebs.keywords['dprob'])
        hebnorm = (1./hebdprob**2 + 1./hebnewdprob**2) 
        hebs.keywords['prob'] = (hebprob/hebdprob**2 + hebnewprob/hebnewdprob**2)/hebnorm
        hebs.keywords['dprob'] = 1/sqrt(hebnorm)        
        nhebs = len(hebs)

        ebprob,ebdprob = (ebs.keywords['prob'],ebs.keywords['dprob'])
        ebnewprob,ebnewdprob = (newebs.keywords['prob'],newebs.keywords['dprob'])
        ebnorm = (1./ebdprob**2 + 1./ebnewdprob**2) 
        ebs.keywords['prob'] = (ebprob/ebdprob**2 + ebnewprob/ebnewdprob**2)/ebnorm
        ebs.keywords['dprob'] = 1/sqrt(ebnorm)        
        nebs = len(ebs)


        if verbose:
            print 'total of %i hebs' % nhebs
            print 'total of %i ebs' % nebs

    A = (hebs.which_eclipse=='A')
    B = (hebs.which_eclipse=='B')
    M1 = A*hebs.MA + B*hebs.MB  #Do all this earlier, in fpsim?
    M2 = hebs.MC
    R1 = A*hebs.RA + B*hebs.RB
    R2 = hebs.RC
    hebs.add_column('M1',M1)
    hebs.add_column('M2',M2)
    hebs.add_column('R1',R1)
    hebs.add_column('R2',R2)


    ebs.add_column('M1',ebs.MA)
    ebs.add_column('M2',ebs.MB)
    ebs.add_column('R1',ebs.RA)
    ebs.add_column('R2',ebs.RB)


    return hebs,ebs

#new FP scenarios for when a faint close companion is detected

def makemodels_specific(koi,mags,dist=None,PA=None,dmag=0.15,tag=None,verbose=True,n=2e4,
                        number=1,overwrite=False,band='Kepler',
                        M=1,R=1,dM=None,dR=None,age=None,stellarmodels='dartmouth',
                        Mdist=None,Rdist=None,fehdist=None,colors=['gr','JK'],
                        feh=None,dfeh=0.15,Teff=None,logg=None,
                        dTeff=None,dlogg=None,minq=0.1,colortol=0.1,
                        spectrum=False,exactstar=False,multmassfn='default',
                        **kwargs):
    """mags is dictionary of companion magnitudes
    """

    koimags = ku.KICmags(koi)  #takes place of obsmags keyword
    obsmags = koimags
    dmags = {}
    for b in mags.keys():
        if b in koimags:
            dmags[b] = mags[b] - koimags[b]
    #dmags = {b:mags[b]-koimags[b] for b in mags.keys()}

    ra,dec = ku.radec(koi)
    starfieldname = radec2name(ra,dec)

    if tag is None:
        folder = '%s/%s' % (FPMODELSDIR,ku.koiname(koi))    
    else:
        folder = '%s/%s_%s' % (FPMODELSDIR,ku.koiname(koi),tag)
    
    if verbose:
        print folder
    if not os.path.exists(folder):
        os.mkdir(folder)

    if not os.path.exists('%s/bebs_specific%i.fits' % (folder,number)) or overwrite:
        bebs = BEBs_specific(koi,mags,dist=dist,PA=PA,
                             dmag=dmag,n=n,verbose=verbose,band=band,
                             stellarmodels=stellarmodels,**kwargs)
        bebs = firstnrows(bebs,n)
        fname = '%s/bebs_specific%i.fits' % (folder,number)
        bebs.write(fname,overwrite=True,verbose=False)
        parfile = '%s/bebs_specific%i_params.fits' % (folder,number)
        if os.path.exists(parfile):
            os.remove(parfile)
    else:
        if verbose:
            print '%s exists; not overwriting.' % ('%s/bebs_specific%i.fits' % (folder,number))

    if not os.path.exists('%s/hebs_specific%i.fits' % (folder,number)) or overwrite:
        hebs = HEBs_specific(koi,dmags,dist=dist,PA=PA,
                             n=n,dm=dM,age=age,verbose=verbose,obsmags=obsmags,colors=colors,
                             colortol=colortol,spectrum=spectrum,Mdist=Mdist,fehdist=fehdist,
                        feh=feh,dfeh=dfeh,starfieldname=starfieldname,stellarmodels=stellarmodels,band=band,
                        multmassfn=multmassfn,minq=minq)
        hebs = firstnrows(hebs,n)
        fname = '%s/hebs_specific%i.fits' % (folder,number)
        hebs.write(fname,overwrite=True,verbose=False)
        parfile = '%s/hebs_specific%i_params.fits' % (folder,number)
        if os.path.exists(parfile):
            os.remove(parfile)
    else:
        if verbose:
            print '%s exists; not overwriting.' % ('%s/hebs_specific%i.fits' % (folder,number))

def BEBs_specific(koi,mags,dmag=0.15,n=2e4,verbose=True,overwrite=False,**kwargs):
    """
    koi,mags,dmag,n  ; mag is dictionary of mags of detected companion; dmag is allowed bin width
    kwargs: stellarmodels('padova'),band('Kepler') , P
    """
    ra,dec = koiradec(koi)    #different strategy than other simulations that pass name; resolve sometime...
    name = radec2name(ra,dec)

    P = ku.DATAFRAME.ix[ku.koiname(koi),'koi_period']

    bebs = fpsim.simBEBs_specific(name,mags,dmag=dmag,verbose=verbose,**kwargs)
    nbebs = len(bebs)
    while nbebs < n:
        newbebs = fpsim.simBEBs_specific(name,mags,dmag=dmag,verbose=verbose,**kwargs)
        bebs.append(newbebs)
        prob,dprob = (bebs.keywords['prob'],bebs.keywords['dprob'])
        newprob,newdprob = (newbebs.keywords['prob'],newbebs.keywords['dprob'])
        norm = (1./dprob**2 + 1./newdprob**2) 
        bebs.keywords['prob'] = (prob/dprob**2 + newprob/newdprob**2)/norm
        bebs.keywords['dprob'] = 1/sqrt(norm)        
        nbebs = len(bebs)
        if verbose:
            print 'total of %i SPECIFIC bebs (%s)' % (nbebs,mags)

    #bebs.add_keyword('P',P)

    #bebs.add_column('M1',bebs.MA)
    #bebs.add_column('M2',bebs.MB)
    #bebs.add_column('R1',bebs.RA)
    #bebs.add_column('R2',bebs.RB)
    
    return bebs
    

def HEBs_specific(koi,dmags,m=1,n=2e4,spectrum=False,verbose=True,**kwargs):
    """
    koi,contrasts,dmag,n  ; contrasts is dictionary of dmags of detected companion; 
                            dmag is unc (bin width)
    koi,dmags,m,n,spectrum
    kwargs: dm,age,verbose,obsmags,colors,feh,dfeh,colortol,multmassfn,starfieldname,stellarmodels,minq,band
    """

    P = ku.DATAFRAME.ix[ku.koiname(koi),'koi_period']

    mags = ku.KICmags(koi)

    if not spectrum:
        m=None

    hebs = fpsim.simHEBs_specific(m,dmags,P=P,**kwargs)
    nhebs = len(hebs)
    if verbose:
        print 'total of %i specific hebs' % nhebs
    while nhebs < n:
        newhebs = fpsim.simHEBs_specific(m,dmags,P=P,**kwargs)
        hebs.append(newhebs)

        hebprob,hebdprob = (hebs.keywords['prob'],hebs.keywords['dprob'])
        hebnewprob,hebnewdprob = (newhebs.keywords['prob'],newhebs.keywords['dprob'])
        hebnorm = (1./hebdprob**2 + 1./hebnewdprob**2) 
        hebs.keywords['prob'] = (hebprob/hebdprob**2 + hebnewprob/hebnewdprob**2)/hebnorm
        hebs.keywords['dprob'] = 1/sqrt(hebnorm)        
        nhebs = len(hebs)
        if verbose:
            print 'total of %i specific hebs' % nhebs


    A = (hebs.which_eclipse=='A')
    B = (hebs.which_eclipse=='B')
    M1 = A*hebs.MA + B*hebs.MB  #Do all this earlier, in fpsim?
    M2 = hebs.MC
    R1 = A*hebs.RA + B*hebs.RB
    R2 = hebs.RC
    hebs.add_column('M1',M1)
    hebs.add_column('M2',M2)
    hebs.add_column('R1',R1)
    hebs.add_column('R2',R2)

    return hebs


########################

def loadfield(koi,binaries=False):
    name = radec2name(*ku.radec(koi))
    return fpsim.loadfield(name,FIELDSDIR,binaries=binaries)

def addmags(*mags):
    tot=0
    for mag in mags:
        tot += 10**(-0.4*mag)
    return -2.5*log10(tot)

def bgstars(koi,bands=fpsim.BANDS,fb=0.4):
    pris = loadfield(koi)
    secs = loadfield(koi,binaries=True)
    ubin = rand.random(size=len(pris))
    isbin = ubin < fb
    wbin = where(isbin & ~isnan(secs.Mact))
    Mact2 = zeros(len(pris))
    Mact2[wbin] = secs.Mact[wbin]
    for b in BANDS:
        pris[b][wbin] = addmags(pris[b][wbin],secs[b][wbin])
    pris.add_column('Mact2',Mact2)
    pris.add_column('is_binary',isbin)
    return pris
