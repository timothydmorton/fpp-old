import pandas as pd
import numpy as np
import os
import distributions as dists

KICFILE = '%s/newkic.csv' % os.environ['KEPLERDIR']

DATA = pd.read_csv(KICFILE)
DATA.index = DATA.kepid

def get_property(kic,*args):
    #print args
    return DATA.ix[kic,list(args)]

def get_distribution(kic,prop):
    val = DATA.ix[kic,prop]
    u1 = DATA.ix[kic,prop+'_err1'] #upper error bar (positive)
    u2 = DATA.ix[kic,prop+'_err2'] #lower error bar (negative)
    return dists.fit_doublegauss(val,-u2,u1)
