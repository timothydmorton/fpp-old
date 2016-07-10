from numpy import *
import pylab as p
import scipy
import numpy.random as random
from scipy.integrate import *
import matplotlib
import plotutils as pu

def flip(N,H=0.5):
	r = random.random(N)
	return (r < H).astype(int)

def fakeflip(k,N,sensitivities=False):
	data = zeros(N)
	data[0:k] = 1
	return data

def normal(x,mu,sig):
    return 1./(sig*sqrt(2*pi))*exp(-(x-mu)**2/(2*sig**2))

def conf_interval(x,L,conf=0.683,shortest=True,conftol=0.001):
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
		pu.setfig(fig)
	
	lo,hi = conf_interval(x,px,conf,shortest=shortest)
	if ax==None:
		ax = p.gca()

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
		
		poly = p.Polygon(verts,facecolor='0.8',edgecolor='k')
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
	pu.setfig(fig)

	if ax == None:
		ax = p.gca()

	if symmetric:
		foo1,foo2 = meshgrid(x,y)
		L[where(foo1-foo2 < 0)] = 0
	px = trapz(L,y,axis=0)
	py = trapz(L,x,axis=1)

	X,Y = meshgrid(x,y)
	
	p.clf()
		
	if not justcontour:
	
		left, width = 0.1, 0.6
		bottom, height = 0.1, 0.6
		bottom_h = left_h = left+width #+0.05

		nullfmt = matplotlib.ticker.NullFormatter()

		rect_center = [left, bottom, width, height]
		rect_top = [left, bottom_h, width, 0.2]
		rect_right = [left_h, bottom, 0.2, height]
	
		axcenter = p.axes(rect_center)
		axtop = p.axes(rect_top)
		axright = p.axes(rect_right)
	else:
		axcenter = p.gca()
	
	
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

def p_of_f(results,prior=ones(1000),hold=False,noplot=False,plotprior=False,norm=True):
	nheads = sum(results)
	n = len(results)
	ntails = n-nheads
	f = linspace(0,1,len(prior))
	L = f**nheads*(1-f)**ntails
	posterior = L*prior/trapz(L*prior,f)
	if not noplot:
		if not hold:
			p.clf()
		if norm:
			p.plot(f,posterior)
		else:
			p.plot(f,L*prior)
		if plotprior:
			p.plot(f,prior,ls=':')
		p.title('%i trials, %i successes' % (n,nheads))
	if norm:
		return f,posterior
	else:
		return f,L*prior

def meshgridnd(*arrs):
    arrs = tuple(reversed(arrs))  #edit
    lens = map(len, arrs)
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz*=s

    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = array(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)

    return tuple(ans)
