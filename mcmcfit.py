#!/usr/bin/env python
import keplerfpp as kfpp
import numpy as np
import sys,os,re,time,os.path
import parsetable
import argparse

#DATAFILE = '%s/FPP/KOI_forFPP.tbl' % os.environ['DROPBOX']
DATAFILE = os.environ['KOIDATAFILE']
FPMODELSDIR = '%s/FPP/models' % os.environ['KEPLERDIR']
CHAINSDIR = '%s/data/chains' % os.environ['KEPLERDIR']


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Generate FP models for a list of KOIs')

    parser.add_argument('kois',metavar='KOI',nargs='*',help='a KOI (or list of kois)')
    parser.add_argument('-f','--file',default=None)
    parser.add_argument('--redo',action='store_true')

    

    args = parser.parse_args()

    if args.file is not None:
        koilist = []
        if not os.path.exists(args.file):
            args.file = '%s/lists/%s' % (os.environ['KEPLERDIR'],args.file)
        for line in open(args.file):
            koilist.append(line.strip())
    else:
        koilist = args.kois

    i = 0
    ntot = len(koilist)
    for koi in koilist:
        i += 1
        try:
            if os.path.exists('%s/%s/ts.npy' % (CHAINSDIR,koi)) and not args.redo:
                print '(%i of %i) %s done already (skipping).' % (i,ntot,koi)
            else:
                start = time.time()
                sig = kfpp.KeplerTransitsignal(koi)
                sig.MCMC(refit=True)
                end = time.time()
                print '(%i of %i) %s fit in %.1f sec (%i points)' % (i,ntot,koi,end-start,len(sig.ts))
        except KeyboardInterrupt:
            raise
        except:
            print sys.exc_info()[0]
