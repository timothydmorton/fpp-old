"""
Various statistical and plotting utilities
"""

from numpy import *
from scipy.optimize import leastsq,fminbound
from scipy.special import erf
import distributions as dists
import numpy as np
import emcee
import numpy.random as rand

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotutils as plu
#from lmfit import minimize, Parameters, Parameter, report_fit

def fit_pdf(bins,h,distfn,p0,nwalkers=200,nburn=200,niter=1000,mcmc=True):

    #zero-pad the ends of the distribution to keep fits positive
    N = len(bins)
    dbin = (bins[1:]-bins[:-1]).mean()
    newbins = np.concatenate((np.linspace(bins.min() - N/10*dbin,bins.min(),N/10),
                             bins,
                             np.linspace(bins.max(),bins.max() + N/10*dbin,N/10)))
    newh = np.concatenate((np.zeros(N/10),h,np.zeros(N/10)))
    
    if mcmc:
        ndims = len(p0)
        model = Model_Distribution_Histogram(newbins,newh,distfn)
        sampler = emcee.EnsembleSampler(nwalkers,ndims,model)
        
        p0 = rand.random((nwalkers,ndims))
        pos, prob, state = sampler.run_mcmc(p0, nburn) #burn in
        
        #run for real
        sampler.reset()
        niter=1000
        foo = sampler.run_mcmc(pos, niter, rstate0=state)
        
        return sampler

    else:
        
        #mu0 = bins[np.argmax(h)]
        #sig0 = abs(mu0 - bins[np.argmin(np.absolute(h - 0.5*h.max()))])
        #p0 = (mu0,sig0,sig0,sig0,sig0,0.5,0.5)

        def resid(pars):
            return newh - distfn(newbins,pars)
        
        pfit,success = leastsq(resid,p0)
        return pfit,success
        
class Model_Distribution_Histogram(object):
    def __init__(self,bins,h,distfn):
        norm = np.trapz(h,bins)
        self.bins = bins
        self.h = h/norm #ensure integration to 1

        self.distfn = distfn

    def __call__(self,pars):
        """ returns log-likelihood 
        """
        hmodel = self.distfn(self.bins,pars)
        return (-0.5*(hmodel - self.h)**2/(self.h*0.01)**2).sum()

class Model_Distribution(object):
    def __init__(self,data,distfn):
        """distfn takes data as input; returns distribution evaluated at those data points
        """
        self.data = data
        self.distfn = distfn

    def __call__(self,pars):
        """ returns log-likelihood
        pars are parameters relevent for self.distfn
        """
        logl = np.log10(distfn(data,pars))
        return logl.sum()

def pctile(x,q):
    q /= 100.
    s = sort(x)
    n = size(x)
    i = s[int(n*q)]
    return x[i]

def normal(x,mu,sig):
    return 1./(sig*sqrt(2*pi))*exp(-(x-mu)**2/(2*sig**2))


def qstd(x,quant=0.05,top=False,bottom=False):
    """returns std, ignoring outer 'quant' pctiles
    """
    s = sort(x)
    n = size(x)
    lo = s[int(n*quant)]
    hi = s[int(n*(1-quant))]
    if top:
        w = where(x>=lo)
    elif bottom:
        w = where(x<=hi)
    else:
        w = where((x>=lo)&(x<=hi))
    return std(x[w])

def kdeconf(kde,conf=0.683,xmin=None,xmax=None,npts=500,shortest=True,conftol=0.001,return_max=False):
    if xmin is None:
        xmin = kde.dataset.min()
    if xmax is None:
        xmax = kde.dataset.max()
    x = linspace(xmin,xmax,npts)
    return conf_interval(x,kde(x),shortest=shortest,conf=conf,conftol=conftol,return_max=return_max)

def conf_interval(x,L,conf=0.683,shortest=True,conftol=0.001,return_max=False):
	#x,L = args
	cum = cumsum(L)
	cdf = cum/cum.max()
	if shortest:
	    maxind = L.argmax()
            if maxind==0:   #hack alert
                maxind = 1
            if maxind==len(L)-1:
                maxind = len(L)-2
            Lval = L[maxind]

            lox = x[0:maxind]
            loL = L[0:maxind]
            locdf = cdf[0:maxind]
            hix = x[maxind:]
            hiL = L[maxind:]
            hicdf = cdf[maxind:]

            dp = 0
            s = -1
            dL = Lval
            switch = False
            last = 0
            while absolute(dp-conf) > conftol:
                Lval += s*dL
                if maxind==0:
                    loind = 0
                else:
                    loind = (absolute(loL - Lval)).argmin()
                if maxind==len(L)-1:
                    hiind = -1
                else:
                    hiind = (absolute(hiL - Lval)).argmin()

                dp = hicdf[hiind]-locdf[loind]
                lo = lox[loind]
                hi = hix[hiind]
                if dp == last:
                    break
                last = dp
                cond = dp > conf
                if cond ^ switch:
                    dL /= 2.
                    s *= -1
                    switch = not switch
			
#		while dp < conf:
#			Lval -= dL
#			loind = argmin(abs(loL - Lval))
#			hiind = argmin(abs(hiL - Lval))
#			dp = hicdf[hiind]-locdf[loind]
#			lo = lox[loind]
#			hi = hix[hiind]

	else:
            alpha = (1-conf)/2.
            lo = x[absolute(cdf-alpha).argmin()]
            hi = x[(absolute(cdf-(1-(alpha)))).argmin()]
        if return_max:
            xmaxL = x[L.argmax()]
            return xmaxL,lo,hi
        else:
            return (lo,hi)

def gaussian(x,p):
    A,mu,sig = p
    return A*exp(-0.5*(x-mu)**2/sig**2)

def lorentzian(x,p):
    A,mu,gam = p
    return A*gam**2/((x-mu)**2 + gam**2)

def voigt(x,p):
    A,mu,gam,sig = p
    z = ((x-mu)+1j*gam)/(sig*sqrt(2))
    test = cef(z).real[where(abs(x)<2)]
    y = cef(z).real/(sig*sqrt(2*pi))
    ymax = y.max()
    return A*y/ymax

def Nvoigt(x,p):
    """takes 1+5N parameters: offset, then A,mu,sig,gam
    """
    p = atleast_1d(p)
    N = (len(p)-1)/4
    y=0
    for i in arange(N):
        y += voigt(x,p[1+4*i:1+4*i+4])
    return y+p[0]

def fit_Nvoigt(x,y,p0,dy=None,BIC=False):
    def errfn(p,x):
        return Nvoigt(x,p)-y
    pfit,success = leastsq(errfn,p0,args=(x,))
    if not BIC:
        return pfit
    else:
        if dy is None:
            raise ValueError('Must provide uncertainties to calculate BIC')
        ymod = Nvoigt(x,pfit)
        logL = log(1./sqrt(2*pi*dy)*exp(-0.5*((y-ymod)**2/dy**2))).sum()
        bic = -2*logL + len(pfit)*log(len(x))
        return pfit,bic
    

def pvoigt(x,p):
    eta,A,mu,sig,gam = p
    pgau = (1.,mu,sig)
    plor = (1.,mu,gam)
    if eta>1:
        eta = 1
    if eta<0:
        eta = 0
    return A*(eta*lorentzian(x,plor) + (1-eta)*gaussian(x,pgau))

def Npvoigt(x,p):
    """takes 1+5N parameters: offset, then eta,A,mu,sig,gam
    """
    p = atleast_1d(p)
    N = (len(p)-1)/5
    y=0
    for i in arange(N):
        y += pvoigt(x,p[1+5*i:1+5*i+5])
    return y+p[0]

def fit_Npvoigt(x,y,p0,dy=None,BIC=False):
    return fit_fn(Npvoigt(x,y,p0,dy,BIC))

def fit_fn(fn,x,y,p0,dy=None,BIC=False):
    def errfn(p,x):
        return fn(x,p)-y
    pfit,success = leastsq(errfn,p0,args=(x,))
    if not BIC:
        return pfit
    else:
        if dy is None:
            raise ValueError('Must provide uncertainties to calculate BIC')
        ymod = fn(x,pfit)
        logL = log(1./sqrt(2*pi*dy)*exp(-0.5*((y-ymod)**2/dy**2))).sum()
        bic = -2*logL + len(pfit)*log(len(x))
        return pfit,bic
    

def Nlorentzian(x,p):
    p = atleast_1d(p)
    N = (len(p)-1)/3
    y = 0
    for i in arange(N):
        A,mu,sig = p[1+3*i:1+3*i+3]
        y += A*exp(-(x-mu)**2/(2.*sig**2))
    return y + p[0]

def fit_Nlor(x,y,p0,dy=None,BIC=False):
    return fit_fn(Nlorentzian,x,y,p0,dy,BIC)

def Ngauss_1d(x,p):
    p = atleast_1d(p)
    N = (len(p)-1)/3
    y = 0
    for i in arange(N):
        A,mu,sig = p[1+3*i:1+3*i+3]
        y += A*exp(-(x-mu)**2/(2.*sig**2))
    return y + p[0]

def fit_Ngauss_1d(x,y,p0,dy=None,BIC=False):
    return fit_fn(Ngauss_1d,x,y,p0,dy,BIC)
    

def howmany_gauss_1d(x,y,dy,Nmax=3,p0=None):
    bics = zeros(Nmax)
    pfits = []
    if p0 is None:
        base = pctile(y,0.1)
        p0 = (base,(y-base).std(),(y-base).mean(),(y-base).max())
    for i in arange(Nmax):
        p = concatenate(([p0[0]],repeat(p0[1:],i+1)))
        pfit,bic = fit_Ngauss_1d(x,y,p,dy,BIC=True)
        pfits.append(pfit)
        bics[i] = bic
    bics -= bics.max()
    print bics

def erfi(x):
    return -1j*erf(1j*x)

def cef(x):
    return exp(-x**2)*(1+1j*erfi(x))


######### routines developed for Ay117 class at Caltech #######

def conf_interval_old(x,L,conf=0.683,shortest=True,conftol=0.001):
	#x,L = args
	cum = cumsum(L)
	cdf = cum/max(cum)
	if shortest:
		maxind = argmax(L)
		Lval = L[maxind]

		lox = x[0:maxind]
		loL = L[0:maxind]
		locdf = cdf[0:maxind]
		hix = x[maxind:]
		hiL = L[maxind:]
		hicdf = cdf[maxind:]

		dp = 0
		s = -1
		dL = Lval
		switch = False
		last = 0
		while abs(dp-conf) > conftol:
			Lval += s*dL
			loind = argmin(abs(loL - Lval))
			hiind = argmin(abs(hiL - Lval))
			dp = hicdf[hiind]-locdf[loind]
			lo = lox[loind]
			hi = hix[hiind]
			if dp == last:
				break
			last = dp
			cond = dp > conf
			if cond ^ switch:
				dL /= 2.
				s *= -1
				switch = not switch
			
#		while dp < conf:
#			Lval -= dL
#			loind = argmin(abs(loL - Lval))
#			hiind = argmin(abs(hiL - Lval))
#			dp = hicdf[hiind]-locdf[loind]
#			lo = lox[loind]
#			hi = hix[hiind]

	else:
		alpha = (1-conf)/2.
		lo = x[argmin(abs(cdf-alpha))]
		hi = x[argmin(abs(cdf-(1-(alpha))))]
	return (lo,hi)

def conf2d(x,y,L,conf=.683,conftol=0.001):
	"""returns the contour level of L corresponding to a given confidence level.
	
	L is a 2-d likelihood grid that need not be normalized, with x and y representing the two dimensions.
	conftol controls how exact you want your answer to be."""
	
	norm = trapz2d(L,x,y)
	prob = 0
	Lval = max(L.ravel())
	dL = Lval
	s = -1
	switch = False
	last = 0
	while abs(prob-conf) > conftol:
		Lval += s*dL
		Ltemp = L*(L>Lval)
		prob = trapz2d(Ltemp,x,y)/norm
		cond = prob > conf
		if prob == last:
			break
		last = prob
		if cond ^ switch:
			dL /= 2.
			s *= -1
			switch = not switch
		#print Lval,prob
	return Lval

def trapz2d(L,x,y):
	return trapz(trapz(L,y,axis=0),x)

def plot_posterior(x,px,name='x',ax=None,fig=None,conf=0.683,shortest=True,median=False,justline=False,shade=True,label=True,\
		labelpos=(0.05,0.7),horizontal=False,axislabels=True,fmt='%.2f',conflabel=True,evidence=False):
	"""Plots a 1-D posterior pdf described by x and px.  Default is to put a vertical dotted line at the 
	best fit value, to shade the shortest 68% confidence interval, and to annotate the graph with the numerical result.
	
	Inputs:
		x		: vector abcissa values
		px		: probability or likelihood as function of x; must be same size as x, not necessarily normalized
		
	Optional Inputs:
		name	: variable name; for use in labels
		ax		: matplotlib 'axis' object; in case you want to specify the plot to be on a specific axis object
		fig		: the number of the figure to put the plot on; empty creates a new figure if 'ax' is not specified
		conf	: confidence level for shade region
		shortest: make False for symmetric confidence region
		median	: make True to draw vertical line at median value instead of max. likelihood
		justline: make True to plot just the posterior pdf; nothing else
		shade	: make False to turn off shading
		label	: make False to turn off the label w/ the value and error bars
		labelpos: the position to place the label, in axis coordinates
		horizontal: make True to make the plot horizontal (e.g. for a 2d posterior plot)
		axislabels: make False to turn off
		fmt: format string for label
		conflabel: make False to not include the confidence level in the label
		
	Results:
		Makes a nifty plot
	
	Dependencies:
		-numpy,matplotlib
		-conf_interval
	
	"""
	if ax == None:
		plu.setfig(fig)
	
	lo,hi = conf_interval(x,px,conf,shortest=shortest)
	if ax==None:
		ax = plt.gca()

	if not median:
		best = x[argmax(px)]
	else:
		cum = cumsum(px)
		cdf = cum/max(cum)
		best = x[argmin(abs(cdf-0.5))]
		
	loerr = best-lo
	hierr = hi - best
	
	if not horizontal:
		ax.plot(x,px,'k')
		if axislabels:		
			ax.set_xlabel('$ %s $' % name,fontsize=16)
			ax.set_ylabel('$ p(%s) $' % name,fontsize=16)
	else:
		ax.plot(px,x,'k')
		if axislabels:
			ax.set_xlabel('$ p(%s) $' % name,fontsize=16)
			ax.set_ylabel('$ %s $' % name,fontsize=16)

	if justline:
		return
			
	if not horizontal:
		ax.axvline(best,color='k',ls=':')
	else:
		ax.axhline(best,color='k',ls=':')
	w = where((x > lo) & (x < hi))
	
	if shade:
		if not horizontal:
			ix = x[w]
			iy = px[w]
			verts = [(lo,0)] + zip(ix,iy) + [(hi,0)]
		else:
			ix = px[w]
			iy = x[w]
			verts = [(0,lo)] + zip(ix,iy) + [(0,hi)]
		
		poly = plt.Polygon(verts,facecolor='0.8',edgecolor='k')
		ax.add_patch(poly)

	beststr = fmt % best
	hierrstr = fmt % hierr
	loerrstr = fmt % loerr
	if hierrstr == loerrstr:
		resultstr = '$%s=%s \pm %s$' % (name,beststr,hierrstr)
	else:
		resultstr = '$ %s =%s^{+%s}_{-%s}$' % (name,beststr,hierrstr,loerrstr)
	if conflabel:
		#print conf
		resultstr += '\n\n(%i%% confidence)' % int(conf*100) 
	if evidence:
		resultstr += '\n\nevidence = %.2e' % trapz(px,x)
	if label:
		ax.annotate(resultstr,xy=labelpos,xycoords='axes fraction',fontsize=16)
		
def plot_posterior2d(x,y,L,name1='x',name2='y',confs=[0.68,0.95,0.99],conf=0.683,ax=None,fig=None,\
		labelpos1=(0.6,0.5),labelpos2=(0.3,0.8),fmt1='%.2f',fmt2='%.2f',evidence=False,\
		evidencelabelpos=(0.05,0.85),labels=True,shade=True,justline=False,
		     symmetric=False,justcontour=False):
	"""Plots contour plot of 2D posterior surface, with given contour levels, including marginalized 1D 
	posteriors of the two individual parameters.
	
	Inputs:
		x,y		: vectors that represent the two directions of the parameter grid
		L		: 2D grid of likelihood values; not necessarily normalized
	
	Optional Inputs:
		confs	: list of confidence contours to plot
		name1,name2	: names of variables
		ax		: matplotlib 'axis' object, in case you want to specify
		fig		: the number of the figure to put the plot on; creates a new figure if not specified
		labelpos1,labelpos2	: where to put the labels on the 1D posterior plots
		fmt1, fmt2      : format strings for labels
		
	Results:
		Makes a nifty plot
		
	Dependencies:
		--numpy, matplotlib
		--plot_posterior, conf_interval, conf2d
	
	"""
	plu.setfig(fig)

	if ax == None:
		ax = plt.gca()

	if symmetric:
		foo1,foo2 = meshgrid(x,y)
		L[where(foo1-foo2 < 0)] = 0
	px = trapz(L,y,axis=0)
	py = trapz(L,x,axis=1)

	X,Y = meshgrid(x,y)
	
	plt.clf()
		
	if not justcontour:
	
		left, width = 0.1, 0.6
		bottom, height = 0.1, 0.6
		bottom_h = left_h = left+width #+0.05

		nullfmt = matplotlib.ticker.NullFormatter()

		rect_center = [left, bottom, width, height]
		rect_top = [left, bottom_h, width, 0.2]
		rect_right = [left_h, bottom, 0.2, height]
	
		axcenter = plt.axes(rect_center)
		axtop = plt.axes(rect_top)
		axright = plt.axes(rect_right)
	else:
		axcenter = plt.gca()
	
	
	levels = zeros(len(confs))
	i=0
	for c in confs:
		levels[i] = conf2d(x,y,L,c)
		i+=1
	axcenter.contour(X,Y,L,lw=1,levels=levels)
	w = where(L==max(L.ravel()))
	axcenter.plot(x[w[1]],y[w[0]],'k+')
	axcenter.set_xlabel('$%s$' % name1,fontsize=16)
	axcenter.set_ylabel('$%s$' % name2,fontsize=16)
	
	if not justcontour:
		plot_posterior(x,px,name1,conf=conf,ax=axtop,axislabels=False,labelpos=labelpos1,fmt=fmt1,
			       conflabel=False,label=labels,shade=shade,justline=justline,fig=0)
		plot_posterior(y,py,name2,conf=conf,ax=axright,horizontal=True,axislabels=False,labelpos=labelpos2,
			       fmt=fmt2,conflabel=False,label=labels,shade=shade,justline=justline,fig=0)

		axtop.yaxis.set_major_formatter(nullfmt)
		axtop.xaxis.set_major_formatter(nullfmt)
		axright.xaxis.set_major_formatter(nullfmt)
		axright.yaxis.set_major_formatter(nullfmt)

	if evidence:
		axcenter.annotate('evidence = %.2e' % trapz2d(L,x,y),xy=evidencelabelpos,xycoords='axes fraction')

def errorbars(x,L,conf=0.95):
	lo,hi = conf_interval(x,L,conf)
	maxL = x[argmax(L)]
	l = maxL-lo
	h = hi-maxL
	return l,h

def triangle_plot(data,names,marg_orientations='v',fig=None,figsize=(8,8),
             lims=None,ticks=None,plotfn_2d=None,plotfn_1d=None,
             kwargs_2d=None,kwargs_1d=None,small_margs=True,marg_spines=False,
             mark_values=None,mark_markersize=15,
             plot_kwargs=None,hist_kwargs=None,axislabel_kwargs=None):
    """plotfn_2d and plotfn_1d are the 2-d and marginalized plots.
       defaults are scatter plot and histogram.  plotfn_1d must
       take a keyword argument "orientation", which may take
       "vertical" or "horizontal" values

       plotfns must take axis object as argument
    """
    #plu.setfig(fig,figsize=figsize)
    fig = plt.gcf()

    if kwargs_2d is None:
        kwargs_2d = {}
        
    if kwargs_1d is None:
        kwargs_1d = {}

    if plot_kwargs is None:
        plot_kwargs = dict(marker='o',ls='none',ms=1,color='k')
        
    if hist_kwargs is None:
        hist_kwargs = dict(normed=True,histtype='step',color='k')

    if axislabel_kwargs is None:
        axislabel_kwargs = {'fontsize':22}
            
    n = len(names)

    if lims is None:
        lims = []
        for i in range(n):
            lims.append((data[:,i].min(),data[:,i].max()))
    
    if type(marg_orientations) != type([]):
        marg_orientations = [marg_orientations]*n


    outer_grid = gridspec.GridSpec(n, n, wspace=0.0, hspace=0.0)
                                   #width_ratios=[100]*n + [1],
                                   #height_ratios=[1] + [100]*n)

    for i in np.arange(n):
        for j in np.arange(i+1,n):
            #k = (j+1)*(n+1) + (i)
            k = j*n + i
            
            ax = plt.Subplot(fig,outer_grid[k])

            if plotfn_2d is None:
                ax.plot(data[:,i],data[:,j],**plot_kwargs)
                        
            else:
                plotfn_2d(data[:,i],data[:,j],ax=ax,**kwargs_2d)

            if mark_values is not None:
                ax.plot(mark_values[i],mark_values[j],'x',zorder=10,
                        ms=mark_markersize,mew=3,color='r')
                
            if i != 0:
                ticklabels = ax.get_yticklabels()
                plt.setp(ticklabels,visible=False)
            else:
                ax.set_ylabel(names[j],**axislabel_kwargs)
            if j != n-1:
                ticklabels = ax.get_xticklabels()
                plt.setp(ticklabels,visible=False)
            else:
                ax.set_xlabel(names[i],**axislabel_kwargs)

            if ticks is not None:
                if ticks[i] is not None:
                    ax.set_xticks(ticks[i])
                if ticks[j] is not None:
                    ax.set_yticks(ticks[j])
                
            if lims is not None:
                ax.set_xlim(*lims[i])
                ax.set_ylim(*lims[j])
            fig.add_subplot(ax)

        #k = (i+1)*(n+1) + (i)
        k = i*n + i
        if small_margs:
            if marg_orientations[i]=='v':
                orientation = 'vertical'
                h_ratios = [2,1]
                w_ratios = [1]
                inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec=outer_grid[k], wspace=0.0, hspace=0.0,
                    height_ratios=h_ratios, width_ratios=w_ratios)
                ax = plt.Subplot(fig,inner_grid[1])
            elif marg_orientations[i]=='h':
                orientation = 'horizontal'
                h_ratios = [1]
                w_ratios = [1,2]
                inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2,
                    subplot_spec=outer_grid[k], wspace=0.0, hspace=0.0,
                    height_ratios=h_ratios, width_ratios=w_ratios)
                ax = plt.Subplot(fig,inner_grid[0])
        else:
            ax = plt.Subplot(fig,outer_grid[k])
            if marg_orientations[i]=='v':
                orientation='vertical'
            elif marg_orientations[i]=='h':
                orientation='horizontal'

            
        if plotfn_1d is None:
            ax.hist(data[:,i],orientation=orientation,**hist_kwargs)
        else:
            plotfn_1d(data[:,i],ax=ax,orientation=orientation,
                    **kwargs_1d)
            
        if mark_values is not None:
            if marg_orientations[i]=='v':
                ax.axvline(mark_values[i],color='r',lw=2)
            elif marg_orientations[i]=='h':
                ax.axhline(mark_values[i],color='r',lw=2)
            
            
        if marg_orientations[i]=='v':
            ax.set_yticks([])
            ax.xaxis.set_ticks_position('bottom')
            ticklabels = ax.get_xticklabels()
            plt.setp(ticklabels,visible=False)
        elif marg_orientations[i]=='h':
            ax.yaxis.set_ticks_position('left')
            ax.set_xticks([])
            ticklabels = ax.get_yticklabels()
            plt.setp(ticklabels,visible=False)
            
        ax.spines['right'].set_visible(marg_spines)
        if k == n+1:
            ax.spines['left'].set_visible(marg_spines)
        ax.spines['top'].set_visible(marg_spines)
        
        if i == n-1:
            if marg_orientations[i]=='v':
                plt.setp(ticklabels,visible=True)
                plt.xlabel(names[i],fontsize=16)
                if ticks is not None:
                    ax.set_xticks(ticks[i])
            elif marg_orientations[i]=='h':
                ax.spines['bottom'].set_visible(marg_spines)
            
        if lims is not None:
            if marg_orientations[i]=='v':
                ax.set_xlim(*lims[i])
            elif marg_orientations[i]=='h':
                ax.set_ylim(*lims[i])


        fig.add_subplot(ax)
        
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9+1./n,top=0.9+1./n)
