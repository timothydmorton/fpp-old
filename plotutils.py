"""
A module with various plotting utilities

"""

import numpy as np
import matplotlib.pyplot as plt
import os,os.path,shutil

def setfig(fig,**kwargs):
    """
    Sets figure to 'fig' and clears; if fig is 0, does nothing (e.g. for overplotting)

    if fig is None (or anything else), creates new figure
    """
    if fig:
        plt.figure(fig,**kwargs)
        plt.clf()
    elif fig==0:
        pass
    else:
        plt.figure(**kwargs)

def makemovie(plotfn,args_fixed,args,name='movie'):
    """ makes a movie out of plotfn, called with series of args (a dictionary of lists)
    """
    moviefile = '%s.gif' % name
    if os.path.exists(moviefile):
        shutil.remove(moviefile)
    if os.path.exists(name):
        shutil.rmdir(name)

    os.mkdir(name)

    #convert dictionary of lists into list of dictionaries
    sizes = {}
    maxsize = 0
    for arg,val in args.iteritems():
        sizes[arg] = np.size(val)
        if sizes[arg] > maxsize:
            maxsize = sizes[arg]
    N = maxsize
                
    arglist = []
    for i in range(N):
        d = {}
        for k in args.keys():
            d[k] = args[k][i]
        for k in args_fixed.keys():
            d[k] = args_fixed[k]
            
        arglist.append(d)

    plt.ioff()
    for i in range(N):
        plt.figure()
        plotfn(**arglist[i])
        plt.savefig('%s/frame%i.png' % (name,i))
        plt.close()
        
def testfn(xs,y):
    plt.plot(xs,xs**y)

