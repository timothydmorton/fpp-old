#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import keplerfpp as kfpp
#import keplerfpsim as kfp
import numpy as np
import numpy.random as rand
import sys,os,re,time,os.path,glob
#import transit_basic as tr
#import fitebs
from consts import *
import atpy
#import parsetable
import argparse
import traceback
import logging

#DATAFILE = '%s/FPP/KOI_forFPP.tbl' % os.environ['DROPBOX']
DATAFILE = os.environ['KOIDATAFILE']
FPMODELSDIR = '%s/FPP/models' % os.environ['KEPLERDIR']

#DATA = parsetable.parseKOItable(DATAFILE)

#DATA = parsetable.csv2dict(DATAFILE,'kepoi_name',names=True)
#parsetable.correct_KICmags(DATA)

import koiutils as ku
DATA = ku.DATA
import kicutils as kicu
import distributions as dists

def imaging_fcomplete(koi,cc,M=None,dM=None,N=1e4,rmax=None):
    pop = binary_starpop(koi,M=M,dM=dM,N=N)
    pop.apply_cc(cc)

    return 1-pop.selectfrac

def binary_starpop(koi,M=None,dM=None,N=1e4,**kwargs):
    import fpsim 
    import transitFPP as fpp
    
    props = DATA[koi]
    if M is None:
        smass = props['koi_smass']
        if smass==False or np.isnan(smass):
            smass = 10**props['koi_slogg']*(props['koi_srad']*RSUN)**2 / G / MSUN
            if np.isnan(smass):
                smass = props['koi_srad']
        M = smass
    if dM is None:
        dM = M*0.1

    Ms = rand.normal(size=N)*dM + M

    mags = ku.KICmags(koi)

    return fpp.BinaryPopulation(fpsim.sim_binaries(Ms),mags=mags)
    return 


def bg_starpop(koi,**kwargs):
    import keplerfpsim as kfp
    import transitFPP as fpp

    mags = ku.KICmags(koi)
    
    return fpp.BGStarPopulation(kfp.bgstars(koi),mags=mags)

def prob_bound(koi,rsky,dmag,band='Kepler',dmag2=None,band2=None,fb=0.4,N=1e5,M=None,dM=None):
    import transitFPP as fpp

    binpop = binary_starpop(koi,M=M,dM=dM,N=N)
    bgpop = bg_starpop(koi)

    mags = ku.KICmags(koi)

    mag = dmag + mags[band]

    if dmag2 is None:
        mag2 = None
    else:
        mag2 = dmag2 + mags[band2]
    

    return fpp.prob_bound(rsky,mag,binpop,bgpop,band=band,fb=fb,mag2=mag2,band2=band2)


def mstar(R,logg,dR=None,dlogg=None):
    if dR is None or dlogg is None:
        return(R*RSUN)**2*10**logg/G / MSUN
    else:
        M = (R*RSUN)**2 * 10**logg/G
        dM = ((dR*RSUN)**2 * (10**logg/G * 2*(R*RSUN))**2 + dlogg**2 * ((R*RSUN)**2/G * np.log(10)*10**logg)**2)**(1./2)
        return M/MSUN, dM/MSUN

def makemodel_kwargs(name,verbose=False,use_JRowe_fit=True,**kwargs):
    import keplerfpp as kfpp

    try:
        props = DATA[name].copy()
    except KeyError:
        allok = True
        needs = []
        for key in ['ra','dec','P','rp','M','R','obsmags']:
            if key not in kwargs:
                allok = False
                needs.append(key)
        if not allok:
            print 'candidate %s not in KOI table; need to provide %s' % (name,needs)
            return
        else:
            props = None

    #norp = True
    #if 'rp' in kwargs:
    #    if kwargs['rp'] is not None:
    #        norp = False
    #
    #if 'R' in kwargs and norp:
    #    newrp = props['koi_prad'] * kwargs['R']/props['koi_srad']
    #    print 'using %.2f as rp (instead of %.2f), since given Rstar is %.2f (instead of %.2f)' %\
    #        (newrp,props['koi_prad'],kwargs['R'],props['koi_srad'])
    #    props['koi_prad'] = newrp
    

    if props is not None:

        mags = {'g':props['koi_gmag'],'r':props['koi_rmag'],
                'i':props['koi_imag'],'z':props['koi_zmag'],
                'J':props['koi_jmag'],'H':props['koi_hmag'],'K':props['koi_kmag']}

        if np.isnan(props['koi_smass']) or props['koi_smass']==False:
            if not np.isnan(props['koi_slogg']):
                props['koi_smass'] = 10**(props['koi_slogg'])*(props['koi_srad']*RSUN)**2/G / MSUN
                print 'no mass in table for %s, calculated M=%.2f from R, logg' % (name,props['koi_smass'])
            else:
                props['koi_smass'] = props['koi_srad']
                print 'no mass or logg in table for %s, setting M=R=%.2f' % (name,props['koi_smass'])

        if 'M' in kwargs:
            props['koi_smass'] = kwargs['M']  #obselete?
            kwargs['spectrum'] = True #obselete?
            if 'dM' not in kwargs:
                dM = 0.1*kwargs['M']
            else:
                dM = kwargs['dM']
            kwargs['Mdist'] = dists.Gaussian_Distribution(kwargs['M'],dM,name='M')

        else:
            kwargs['M'] = props['koi_smass']  #obselete?
            kwargs['Mdist'] = ku.smass_distribution(name)
            kwargs['M'] = kwargs['Mdist'].mu

        if 'R' in kwargs:
            props['koi_srad'] = kwargs['R'] #obselete?
            if 'dR' not in kwargs:
                dR = 0.1*kwargs['R']
            else:
                dR = kwargs['dR']
            props['Rdist'] = dists.Gaussian_Distribution(kwargs['R'],dR,name='R')

        else:
            kwargs['R'] = props['koi_srad'] #obslete?
            kwargs['Rdist'] = ku.srad_distribution(name)
            kwargs['R'] = kwargs['Rdist'].mu

        if use_JRowe_fit:
            rowefit = kfpp.get_rowefit(name)
            ror = rowefit.ix['RD1','val'] #rp/rs from Jason Rowe's fit.
        else:
            if 'ror' in kwargs:
                ror = kwargs['ror']
            else:
                ror = props['koi_ror'] #this is pipeline-returned value
        kwargs['rp'] = ror*kwargs['R']*RSUN/REARTH 
        
        del kwargs['ror'] #b/c kfp.makemodels does not expect this.


        #make sure uncertainties in M, radius are assigned [this all obselete now with dists being default?]
        if 'dR' in kwargs:
            props['koi_srad_err1'] = kwargs['dR']
        else:
            if props['koi_srad_err1'] == False or np.isnan(props['koi_srad_err1']):
                props['koi_srad_err1'] = kwargs['R'] * 0.2
            kwargs['dR'] = props['koi_srad_err1']
        if 'dM' in kwargs:
            props['koi_smass_err1'] = kwargs['dM']
        else:
            if props['koi_smass_err1'] == False or np.isnan(props['koi_smass_err1']):
                props['koi_smass_err1'] = props['koi_srad_err1']
            kwargs['dM'] = props['koi_smass_err1']

            
        if 'Teff' in kwargs:
            props['koi_steff'] = kwargs['Teff']
        else:
            kwargs['Teff'] = props['koi_steff']
        if 'logg' in kwargs:
            props['koi_slogg'] = kwargs['logg']
        else:
            kwargs['logg'] = props['koi_slogg']
        if 'feh' in kwargs:
            if 'dfeh' in kwargs:
                dfeh = kwargs['dfeh']
            else:
                dfeh = 0.2
            fehdist = dists.Gaussian_Distribution(kwargs['feh'],dfeh)
            #props['koi_smet'] = kwargs['feh']

        else:
            fehdist = ku.feh_distribution(name)
            #kwargs['feh'] = props['koi_smet']
            #if kwargs['feh'] == False:
            #    kwargs['feh'] = None
        if 'obsmags' not in kwargs:
            kwargs['obsmags'] = mags

        kwargs['fehdist'] = fehdist

        print 'input kwargs: %s' % kwargs
        #print 'properties dictionary (from catalog): %s' % props

        if 'spectrum' in kwargs:
            if kwargs['spectrum']==True:
                props['koi_sparprov'] = 'custom'


        if props['koi_sparprov'] in ['SPC','SME','Astero']:
            kwargs['spectrum'] = False

        kwargs['spectrum'] = props['koi_sparprov'] in ['SPC','SME','Astero','custom']

        if 'P' not in kwargs or kwargs['P'] is None:
            kwargs['P'] = props['koi_period']
        if 'rp' not in kwargs or kwargs['rp'] is None:
            kwargs['rp'] = props['koi_prad']

        if verbose:
            print 'KOI stellar parameters from %s, "spectrum" kw set to %s.' %\
                (props['koi_sparprov'],kwargs['spectrum'])

        kwargs['spectrum'] = False  #testing to see if this just is obselete and the dists thing works

        if np.isnan(kwargs['M']):
            raise NoStellarPropsError('Stellar mass estimate is nan for %s.  Aborting calculation.' % name)

    return kwargs

def makemodels(name,verbose=False,return_kwargs=False,
               use_JRowe_fit=True,comps=None,**kwargs):
    """Simulates transit/eclipse scenarios for a KOI

    Here's how stellar parameters should be handled:
       -If M,R are explicitly provided, those values are used
       -If not, if 'koi_sparprov' field is SME, SPC or Astero, use catalog parameters
       -Else, use no parameters, require colors

    ***UPDATED Stellar Parameters Flowchart***
       -If M,R are explicitly provided, those values are used
       -If not, then Mdist and Rdist will be according to KIC 2.0 
         (this should eliminate "spectrum" keyword--always true--and colormatching)


    """
    import keplerfpsim as kfp

    kwargs = makemodel_kwargs(name,verbose=verbose,use_JRowe_fit=use_JRowe_fit,
                              **kwargs)

    if comps is None or ignore_specific:
        n_specific = 0
    else:
        n_specific = len(comps)

    try:
        if return_kwargs:
            return kwargs
        kfp.makemodels(name,verbose=verbose,**kwargs)
    except kfp.fpsim.BadColorsError:
        kwargs['colors'] = []
        logging.warning('Problem generating binary/triple populations with given colors; relaxing color constraints for %s and retrying.' % name)
        kfp.makemodels(name,verbose=verbose,**kwargs)

    #for each companion, make a specific simulation
    for i in range(n_specific):
        mags = {}
        for col in comps.columns:
            if col in ['dist','PA','survey'] or re.match('e_',col):
                continue
            else:
                mags[col] = comps[col][i]
        kfp.makemodels_specific(koi,mags,number=i+1,
                                dist=comps['dist'][i],
                                PA=comps['PA'][i],**kwargs)
        


    #This used to be the 'else' to 'if props is not None'...not used anymore?
    #else:
    #    try:
    #        if return_kwargs:
    #            return kwargs
    #        kfp.makemodels(name,verbose=verbose,**kwargs)
    #    except kfp.fpsim.BadColorsError:
    #        kwargs['colors'] = []
    #        logging.warning('Problem generating binary/triple populations with given colors; relaxing color constraints for %s and retrying' % name)
    #        kfp.makemodels(name,props['ra'],props['dec'],verbose=verbose,**kwargs)
            

def foldername(name,tag=None):
    if tag is None:
        folder = '%s/%s' % (FPMODELSDIR,name)
    else:
        folder = '%s/%s_%s' % (FPMODELSDIR,name,tag)
        
    return folder
    

def fitallebs(name,tag=None,conv=True,use_pbar=True,n_specific=0):
    import transit_basic as tr 
    import fitebs

    folder = foldername(ku.koiname(name),tag)
    if not os.path.exists(folder):
        print '%s does not exist!' % folder
        return

    MAfn = tr.MAInterpolationFunction()
    models = ['hebs','ebs','bebs','bgpls','pls']
    for i in range(n_specific):
        models.append('hebs_specific{}'.format(i+1))
        models.append('bebs_specific{}'.format(i+1))

    for model in models:
        ebfile = '%s/%s.fits' % (folder,model)
        ebs = atpy.Table(ebfile,verbose=False)
        logfile = '%s/%s.log' % (folder,model)
        outfile = '%s/%s_params.fits' % (folder,model)
        if not os.path.exists(outfile) or os.stat(outfile).st_mtime < os.stat(ebfile).st_mtime:
            t = fitebs.fitebs(ebs,MAfn=MAfn,conv=conv,use_pbar=use_pbar,logfile=logfile,msg='%s: ' % model)
            t.write(outfile,verbose=False,overwrite=True)

def fit_specific(name,tag=None,conv=True,use_pbar=True):
    import transit_basic as tr
    import fitebs
    
    folder = foldername(ku.koiname(name),tag)
    if not os.path.exists(folder):
        print '%s does not exist!' % folder
        return
    
    MAfn = tr.MAInterpolationFunction()
    
    files = glob.glob('%s/*_specific?.fits' % folder)
    for f in files:
        ebs = atpy.Table(f,verbose=False)
        m = re.search('%s/(.+_specific[0-9])\.fits' % folder,f)
        logfile = '%s/%s.log' % (folder,m.group(1))
        outfile = '%s/%s_params.fits' % (folder,m.group(1))
        if not os.path.exists(outfile) or os.stat(outfile).st_mtime < os.stat(f).st_mtime:
            t = fitebs.fitebs(ebs,MAfn=MAfn,conv=conv,use_pbar=use_pbar,logfile=logfile,
                              msg='%s: ' % m.group(1))
            t.write(outfile,verbose=False,overwrite=True)        


def modelfits_done(name,tag=None,n_specific=0):
    folder = foldername(ku.koiname(name),tag)
    done = True
    models = ['hebs','ebs','bebs','bgpls','pls']
    for i in range(n_specific):
        models.append('hebs_specific{}'.format(i+1))
        models.append('bebs_specific{}'.format(i+1))

    for model in models:
        ebfile = '%s/%s.fits' % (folder,model)
        logfile = '%s/%s.log' % (folder,model)
        outfile = '%s/%s_params.fits' % (folder,model)
        if os.path.exists(outfile) and os.path.exists(ebfile):
            done &= os.stat(outfile).st_mtime > os.stat(ebfile).st_mtime
        else:
            done = False
    return done
            
def models_exist(name,tag=None,n_specific=0):
    models = ['hebs','ebs','bebs','bgpls','pls']
    for i in range(n_specific):
        models.append('hebs_specific{}'.format(i+1))
        models.append('bebs_specific{}'.format(i+1))

    folder = foldername(name,tag)
    all_files_exist = True
    for model in models:
        all_files_exist = all_files_exist and os.path.exists('{}/{}.fits'.format(folder,model))

def specific_models_exist(name,number=1,tag=None):
    folder = foldername(name,tag)
    return os.path.exists('{}/bebs_specific{}.fits'.format(folder,number)) and\
        os.path.exists('{}/hebs_specific{}.fits'.format(folder,number))

def doall(koi,verbose=True,n=2e4,overwrite=False,resimbebs=False,minq=0.1,colors=['JH','HK'],
          colortol=0.1,stellarmodels='dartmouth',tag=None,redo_MCMC=False,
          noplot=False,stellarprops=None,figformat='png',cc=None,ccfile=None,
          use_cc=True,ccband=None,secthresh=None,plotfolder=None,vcc=None,maxrad=None,
          spectrum=False,rp=None,ror=None,Tdur=None,P=None,epoch=None,return_object=False,
          trend_limit=None,
          priorfactors=None,fp_specific=None,fp_bg=0.75,include_specific=True,
          use_JRowe_fit=True,photfile=None,photcols=(0,1)):
    import keplerfpp as kfpp

    koi = ku.koiname(koi)

    if koi not in ku.DATAFRAME.index:
        raise NoKOIError('%s not in cumulative table.' % koi)

    if figformat == 'png':
        plt.switch_backend('Agg')
    elif figformat == 'pdf':
        plt.switch_backend('PDF')
    try:
        if stellarprops is None or stellarprops=={}:
            stellarprops = {}
        else:
            print 'stellar properties provided:',stellarprops
            spectrum = True

        #get image data to find close companions
        imd = kfpp.kim.all_imagedata(koi)

        #set maxrad here, to be able to check for companions 
        if maxrad is None:
            maxrad = kfpp.koi_rexclusion(koi)

        #see if there are companions
        if include_specific:
            comps = imd.within_radius(maxrad)
            n_specific = len(comps)
        else:
            comps = None
            n_specific = 0

        if not models_exist(koi,tag,n_specific) or overwrite:
            makemodels(koi,comps=comps,verbose=verbose,n=n,resimbebs=resimbebs,minq=minq,colors=colors,
                       spectrum=spectrum,colortol=colortol,stellarmodels=stellarmodels,
                       overwrite=overwrite,tag=tag,rp=rp,ror=ror,use_JRowe_fit=use_JRowe_fit,
                       **stellarprops) #includes "specific" simulations


        if not modelfits_done(koi,tag,n_specific=n_specific):
            fitallebs(koi,tag=tag,n_specific=n_specific)

        f = kfpp.KOIFPP(koi,use_cc=use_cc,secthresh=secthresh,maxrad=maxrad,tag=tag,
                        redo_MCMC=redo_MCMC,P=P,epoch=epoch,ror=ror,Tdur=Tdur,
                        include_specific=include_specific,photfile=photfile,photcols=photcols)
        if not f.trsig.fit_converged:
            raise MCMCNotConvergedError('Trapezoidal MCMC fit did not converge for %s.  Acors: (T=%.2g, d=%.2g, slope=%.2g, tc=%.2g)' % (koi,f.trsig.Ts_acor,f.trsig.ds_acor,
                                                                                                                                         f.trsig.slopes_acor,f.trsig.tcs_acor))

        if priorfactors is None:
            priorfactors = {}
        if fp_specific is None:
            fp_specific = kfpp.kepler_fp_toy(f['pl'].stars.keywords['RBINCEN'])
        priorfactors['fp_specific'] = fp_specific
        if fp_bg is not None:
            priorfactors['fp'] = fp_bg


        f.change_prior(**priorfactors)
        if cc is not None:
            if type(cc) == type([1]):
                for c in cc:
                    f.apply_cc(c)
            else:
                f.apply_cc(cc)
        else:
            if ccfile is not None:
                if ccband is None:
                    raise ValueError('must provide band with cc file argument')
                cc = kfpp.fpp.ContrastCurveFromFile(ccfile,ccband)
                f.apply_cc(cc)

        if vcc is not None:
            f.apply_vcc(vcc)

        if trend_limit is not None:
            f.apply_trend_constraint(*trend_limit)
        
        #f.trsig.MCMC(refit=redo_MCMC)
        if not noplot:
            if plotfolder is not None:
                plotfolder = '%s/%s' % (plotfolder,koi)
                if not os.path.exists(plotfolder):
                    os.mkdir(plotfolder)
            try:
                f.FPPplots(format=figformat,folder=plotfolder)
            except:
                logging.warning('Error making plots for {}.'.format(koi))
        else:
            fpv = f.fpV()
            fpp = f.FPP()
            print 'fpV = %.3f; FPP = %.2g' % (fpv,fpp)
        results = {}
        results['priorfactors'] = priorfactors
        results['priors'] = {}
        results['lhoods'] = {}
        results['Ls'] = {}
        for model in ['eb','heb','bgeb','bgpl','pl']:
            results['priors'][model] = f.prior(model)
            results['lhoods'][model] = f.lhood(model)
            results['Ls'][model] = results['priors'][model]*results['lhoods'][model]
            
        #resultsfile = '%s/%s/results.log' % (FPMODELSDIR,koi)
        #if not os.path.exists(resultsfile):
        #    fout = open(resultsfile,'w')
        #    fout.write('time N ')

        if return_object:
            return f
        else:
            return results,f.fpV(),f.FPP()
    except:
        raise
        traceback.print_exception(*sys.exc_info())
        print 'error with FPP calculation for %s.  returning nans.' % koi
        results = {}
        results['priors'] = {}
        results['lhoods'] = {}
        results['Ls'] = {}
        for model in ['eb','heb','bgeb','bgpl','pl']:
            results['priors'][model] = np.nan
            results['lhoods'][model] = np.nan
            results['Ls'][model] = np.nan
        return results,np.nan


class NoStellarPropsError(Exception):
    pass

class MCMCNotConvergedError(Exception):
    pass

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Generate FP models for a list of KOIs')

    parser.add_argument('kois',metavar='KOI',nargs='*',help='a KOI (or list of kois)')
    parser.add_argument('-n','--n',type=int,default=2e4)
    parser.add_argument('-o','--overwrite',action='store_true')
    parser.add_argument('--noplot',action='store_true')
    parser.add_argument('--resimbebs',action='store_true')
    parser.add_argument('--minq',type=float,default=0.1)
    parser.add_argument('--colors',nargs='*',default=['JH','HK'])
    parser.add_argument('--colortol',type=float,default=0.1)
    parser.add_argument('--stellarmodels',type=str,default='dartmouth')
    parser.add_argument('--tag',default=None)
    parser.add_argument('--mstar',default=None)
    parser.add_argument('--nocolors',action='store_true')
    parser.add_argument('-f','--file',default=None)
    parser.add_argument('--pdf',action='store_true')
    parser.add_argument('-p','--stellarprops',default=None)
    parser.add_argument('--ccfile',default=None)
    parser.add_argument('--ccband',default=None)
    parser.add_argument('--nocc',action='store_true')
    parser.add_argument('--secthresh',default=None) #implement this through
    parser.add_argument('--refitmcmc',action='store_true')
    parser.add_argument('--plotfolder',default=None)
    parser.add_argument('--maxrad',type=float,default=None)
    parser.add_argument('-M','--Mstar',default=None,nargs='*',type=float)
    parser.add_argument('-R','--Rstar',default=None,nargs='*',type=float)
    parser.add_argument('--feh',default=None,nargs='*',type=float)
    parser.add_argument('-dM','--dMstar',default=None,nargs='*',type=float)
    parser.add_argument('-dR','--dRstar',default=None,nargs='*',type=float)
    parser.add_argument('--dfeh',default=None,nargs='*',type=float)
    parser.add_argument('--Teff',default=None,nargs='*',type=float)
    parser.add_argument('--dTeff',default=None,nargs='*',type=float)
    parser.add_argument('-rp','--rp',type=float,default=None)
    parser.add_argument('-fpp','--fpp',action='store_true')
    parser.add_argument('--ror',type=float,default=None)
    parser.add_argument('--Tdur',type=float,default=None)
    parser.add_argument('--noJRowefit',action='store_true')
    parser.add_argument('--photfile',default=None)
    parser.add_argument('--photcols',nargs=2,type=int,default=[0,1])
    parser.add_argument('--ignore_specific',action='store_true')

    args = parser.parse_args()
    use_JRowe_fit = not args.noJRowefit
    include_specific = not args.ignore_specific

    #if no KOI argument provided, default to 'kois.list' in working directory
    if len(args.kois)==0 and args.file is None:
        args.file = 'kois.list'  
        if args.plotfolder is None:
            args.plotfolder = '.'

    #if args.plotfolder is None:
    #    args.plotfolder = '.'

    if args.nocolors:
        args.colors = []

    if args.pdf:
        figformat='pdf'
    else:
        figformat='png'

    if args.stellarprops is not None:
        raise ValueError('fix parsing of stellar table...not using parsetable anymore.')
        #stellarprops = parsetable.parsetxt(args.stellarprops)
    else:
        stellarprops = {}
        if args.Mstar is not None:
            for k,M in zip(args.kois,args.Mstar):
                if k not in stellarprops:
                    stellarprops[k] = {}
                stellarprops[k]['M'] = M

        if args.dMstar is not None:
            for k,dM in zip(args.kois,args.dMstar):
                if k not in stellarprops:
                    stellarprops[k] = {}
                stellarprops[k]['dM'] = dM

        if args.Rstar is not None:
            for k,R in zip(args.kois,args.Rstar):
                if k not in stellarprops:
                    stellarprops[k] = {}
                stellarprops[k]['R'] = R

        if args.dRstar is not None:
            for k,dR in zip(args.kois,args.dRstar):
                if k not in stellarprops:
                    stellarprops[k] = {}
                stellarprops[k]['dR'] = dR

        if args.feh is not None:
            for k,f in zip(args.kois,args.feh):
                if k not in stellarprops:
                    stellarprops[k] = {}
                stellarprops[k]['feh'] = f

        if args.dfeh is not None:
            for k,df in zip(args.kois,args.dfeh):
                if k not in stellarprops:
                    stellarprops[k] = {}
                stellarprops[k]['dfeh'] = df

        if args.Teff is not None:
            for k,T in zip(args.kois,args.Teff):
                if k not in stellarprops:
                    stellarprops[k] = {}
                stellarprops[k]['Teff'] = T

        if args.dTeff is not None:
            for k,dT in zip(args.kois,args.dTeff):
                if k not in stellarprops:
                    stellarprops[k] = {}
                stellarprops[k]['dTeff'] = dT
                
                
    if args.secthresh is not None:
        args.secthresh = float(args.secthresh)

    photcols = (int(args.photcols[0]),args.photcols[1])

    outfile = None
    if args.file is not None:
        koilist = []
        for line in open(args.file):
            koilist.append(line.strip())
        outfile = '%s.results' % args.file
        logfile = '%s.errorlog' % args.file
        fout = open(outfile,'w')
        logout = open(logfile,'w')
    else:
        koilist = args.kois

    if outfile is not None:
        fout.write('# Name pr_eb pr_heb pr_bgeb pr_bgpl pr_pl fpV fp FPP exception\n')
        fout.close()
        logout.close()
    
        
    for i,koi in enumerate(koilist):
        start = time.time()
        print '%i of %i: %s' % (i+1,len(koilist),koi)

        #try:
        #    if 'koi_prad' not in DATA[koi]:
        #        print 'skipping %s (no planet radius in table).' % koi
        #        continue
        #except KeyError,e:
        #    print '%s not in data table (KeyError: %s)' % (koi,e)

        if outfile is not None:
            logout = open(logfile,'a')

        try:
            if koi in stellarprops:
                props = stellarprops[koi]
            else:
                props = {}
                
                
            results,fpV,fpp = doall(koi,verbose=True,n=args.n,resimbebs=args.resimbebs,minq=args.minq,
                                colors=args.colors,noplot=args.noplot,colortol=args.colortol,
                                stellarmodels=args.stellarmodels,overwrite=args.overwrite,maxrad=args.maxrad,
                                tag=args.tag,figformat=figformat,stellarprops=props,secthresh=args.secthresh,
                                use_cc=not args.nocc,ccfile=args.ccfile,ccband=args.ccband,rp=args.rp,ror=args.ror,Tdur=args.Tdur,
                                redo_MCMC=args.refitmcmc,plotfolder=args.plotfolder,use_JRowe_fit=use_JRowe_fit,
                                    photfile=args.photfile,photcols=photcols,include_specific=include_specific)
            fp = results['priorfactors']['fp_specific']
            Ls = results['Ls']
            Ltot = Ls['eb'] + Ls['heb'] + Ls['bgeb'] + Ls['bgpl'] + Ls['pl']
            line = '%s %.2e %.2e %.2e %.2e %.2e %.3f %.3f %.2g None\n' % (koi,Ls['eb']/Ltot,Ls['heb']/Ltot,
                                                                   Ls['bgeb']/Ltot,Ls['bgpl']/Ltot,
                                                                   Ls['pl']/Ltot,fpV,fp,fpp)
        except KeyboardInterrupt:
            raise
            
        except:
            if outfile is not None:
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info,file=logout)
                logout.write('-----------------^%s^-------------------\n' % koi)
                #print 'error with %s! (details written to log file)' % koi
                traceback.print_exception(*exc_info)
                m = re.search('([a-zA-Z]+Error)',str(exc_info[0]))
                if m:
                    exc_string = m.group(1)
                else:
                    exc_string = 'other?'
                line = '%s nan nan nan nan nan nan nan nan %s\n' % (koi,exc_string)
                fpV = np.nan
                fpp = np.nan
            else:
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                fpV = np.nan
                fpp = np.nan
                #raise

        if outfile is not None:
            fout = open(outfile,'a')
            fout.write(line)
            fout.close()
                        
        end = time.time()

        if outfile is not None:
            logout.close()
        
        print 'Done with %s.  end-to-end %.2f minutes.  fpV = %.3f, FPP = %.2g' % (koi,(end-start)/60.,fpV,fpp)


class NoKOIError(Exception):
    pass
