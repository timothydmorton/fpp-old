#!/usr/bin/env python

import os,re,sys,os.path,shutil,glob
import argparse
import numpy as np
import subprocess


FPPCALCDIR = '%s/FPP/calculations' % os.environ['KEPLERDIR']
CALCFPP_PATH = '%s/src/calcfpp.py' % os.environ['FPPDIR']

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Setup parallelized FPP calculations to run using Condor.')

    parser.add_argument('koilist',type=str)
    parser.add_argument('--ngroups',type=int, default=100)
    parser.add_argument('--rootfolder',default=FPPCALCDIR)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--fixnan',action='store_true')
    parser.add_argument('fppargs',nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.fixnan:
        args.resume = True

    print args

    basename = os.path.splitext(os.path.basename(args.koilist))[0]
    folder = '%s/%s' % (args.rootfolder,basename)
    
    if not args.resume:
        shutil.rmtree(folder,ignore_errors=True)
    if not os.path.exists(folder):
        os.mkdir(folder)

        
    try:
        allkois = np.loadtxt(args.koilist)
    except ValueError:
        allkois = np.loadtxt(args.koilist,dtype=str)

        

    #write the various subset group folders
    for i in range(args.ngroups):
        ifolder = '%s/group.%i' % (folder,i)
        if not args.resume:
            shutil.rmtree(ifolder,ignore_errors=True)
        if not os.path.exists(ifolder):
            os.mkdir(ifolder)
        subset = allkois[i::args.ngroups]
        np.savetxt('%s/kois.list' % ifolder,subset,fmt='%s')
            
    #write the condor script file
    fpparg_str = ''
    for arg in args.fppargs:
        fpparg_str += arg + ' '

    condorfile = '%s/%s.submit' % (folder,basename)
    fout = open(condorfile,'w')
    fout.write('universe = vanilla\n')
    fout.write('executable = %s\n' % CALCFPP_PATH)
    if fpparg_str != '':
        fout.write('arguments = %s\n' % fpparg_str)
    fout.write('getenv = True\n')
    fout.write('output = job_output\n')
    fout.write('error = job_error\n')
    fout.write('log = job_log\n')
    fout.write('initialdir = %s/group.$(Process)\n' % folder)
    fout.write('requirements = OpSysMajorVer == 6\n')
    fout.write('queue %i\n' % args.ngroups)
    fout.close()

    np.savetxt('%s/all.list' % folder,allkois,fmt='%s')

    #execute condor script file
    subprocess.call('condor_submit %s' % condorfile,shell=True)
