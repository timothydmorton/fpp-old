#!/Library/Frameworks/EPD64.framework/Versions/Current/bin/python
# #!/usr/bin/env python
from numpy import *
import transit_basic as tr
import sys,re,os
import atpy
from progressbar import Percentage,Bar,RotatingMarker,ETA,ProgressBar
import os,sys
from scipy.interpolate import LinearNDInterpolator as interpnd


def fitebs(ebs,MAfn=None,conv=True,use_pbar=True,startind=None,endind=None,logfile=None,trap=True,msg=''):
    if logfile is None:
        logf = sys.stdout
    else:
        logf = open(logfile,'w')

    nall = len(ebs)
    if startind is None:
        startind = 0
    if endind is None:
        endind = nall

    inds = arange(startind,endind)
    data = ebs.rows(inds)
    n = len(data)
    deps,durs,slopes = (zeros(n),zeros(n),zeros(n))
    secs = zeros(n).astype(bool)
    dsec = zeros(n)

    p0s,bs,aRs = tr.eclipse_pars(data.P,data.M1,data.M2,data.R1,data.R2,
                                 inc=data.inc,ecc=data.ecc,w=data.w)

    #u11s = data['u11']
    #u21s = data['u21']
    #u12s = data['u12']
    #u22s = data['u22']
    
    #u11s = 0.394*ones(n)
    #u21s = 0.296*ones(n)
    #u12s = 0.394*ones(n)
    #u22s = 0.296*ones(n)

    widgets = [msg+'fitting shape parameters for %i systems: ' % n,Percentage(),
               ' ',Bar(marker=RotatingMarker()),' ',ETA()]
    pbar = ProgressBar(widgets=widgets,maxval=n)

    if use_pbar:
        pbar.start()
    for i in arange(n):
        debug = i in []
        if debug:
            print i
        pri = (data.dpri[i] > data.dsec[i]) or isnan(data.dsec[i])
        sec = not pri
        secs[i] = sec
        p0,aR = (p0s[i],aRs[i])
        if sec:
            b = data.b_sec[i]
            frac = data.fluxfrac2[i]
            dsec[i] = data.dpri[i]
            u1 = data.u12[i]
            u2 = data.u22[i]
        else:
            b = data.b_pri[i]
            frac = data.fluxfrac1[i]
            dsec[i] = data.dsec[i]
            u1 = data.u11[i]
            u2 = data.u21[i]
        try:
            if MAfn is not None:
                if p0 > MAfn.pmax or p0 < MAfn.pmin:
                    if trap:
                        pp = tr.eclipse_tt(p0,b,aR,data.P[i],conv=conv,MAfn=None,debug=debug,
                                           frac=frac,ecc=data.ecc[i],w=data.w[i],sec=sec,u1=u1,u2=u2)
                    else:
                        pp = tr.eclipse_pp(p0,b,aR,data.P[i],conv=conv,MAfn=None,debug=debug,
                                           frac=frac,ecc=data.ecc[i],w=data.w[i],sec=sec,u1=u1,u2=u2)
                else:
                    if trap:
                        pp = tr.eclipse_tt(p0,b,aR,data.P[i],conv=conv,MAfn=MAfn,debug=debug,
                                           frac=frac,ecc=data.ecc[i],w=data.w[i],sec=sec,u1=u1,u2=u2)
                    else:
                        pp = tr.eclipse_pp(p0,b,aR,data.P[i],conv=conv,MAfn=MAfn,debug=debug,
                                           frac=frac,ecc=data.ecc[i],w=data.w[i],sec=sec,u1=u1,u2=u2)
            else:
                if trap:
                    pp = tr.eclipse_tt(p0,b,aR,data.P[i],conv=conv,MAfn=MAfn,debug=debug,
                                       frac=frac,ecc=data.ecc[i],w=data.w[i],sec=sec,u1=u1,u2=u2)
                else:
                    pp = tr.eclipse_pp(p0,b,aR,data.P[i],conv=conv,MAfn=MAfn,debug=debug,
                                       frac=frac,ecc=data.ecc[i],w=data.w[i],sec=sec,u1=u1,u2=u2)

        except tr.NoEclipseError:
            logf.write('binary %i did not register an eclipse (index %i)\n' % (inds[i],i))
            continue
        except tr.NoFitError:
            logf.write('fit for binary %i did not converge. (index %i)\n' % (inds[i],i))
            continue
        except:
            logf.write('unknown error for binary %i. (index %i)\n' % (inds[i],i))
            continue

        if debug:
            print pp
        
        durs[i],deps[i],slopes[i] = pp
        if use_pbar:
            pbar.update(i)

    if use_pbar:
        pbar.finish()


    t = atpy.Table()
    t.add_column('depth',deps)
    t.add_column('duration',durs)
    t.add_column('slope',slopes)
    t.add_column('secdepth',dsec)
    t.add_column('secondary',secs)
    return t



def main(*args): 
    MAfn = tr.MAInterpolationFunction()  #eventually implement this in C/Cython?
    conv = True
    trap = True

    if len(args) < 1:
        print 'usage: fitebs.py filename [startind endind]'

    if len(args) > 1:
        startind,endind = (args[1],args[2])
        use_pbar = False
    else:
        startind,endind = (None,None)
        use_pbar = True


    ebfile = args[0]  #BEBs table
    m = re.search('(\S+)\.fits$',ebfile)
    if not m:
        #print 'input file must be fits table.'
        #exit
        raise ValueError('input file must be fits table.')
    fileroot = m.group(1)
    logfile = '%s.log' % fileroot

    ebs = atpy.Table(ebfile,verbose=False)

    t = fitebs(ebs,MAfn=MAfn,conv=conv,use_pbar=use_pbar,startind=startind,endind=endind,logfile=logfile,trap=trap)

    #Write to file
    if startind is not None or endind is not None:
        outfile = '%s_params_%i_%i.fits' % (fileroot,startind,endind)
    else:
        outfile = '%s_params.fits' % (fileroot)


    t.write(outfile,verbose=False,overwrite=True)

if __name__=='__main__':
    main(*sys.argv[1:])
