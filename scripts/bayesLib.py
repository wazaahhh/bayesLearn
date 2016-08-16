import sys
import pandas as pd
import csv
import numpy as np
import pylab as pl
import jsonpickle as jp
from scipy.stats import linregress
from scipy import stats  as S
from Entro import Distance

sys.path.append("/Users/maithoma/work/python/")
from tm_python_lib import *
from fitting_tools import *

dir = "/Users/maithoma/github/bayesLearn/scripts/"


def rankorder(x):
	x1 = list(np.sort(x))
	x1.reverse()
	y1 = range(1,len(x1)+1)
	return np.array(x1),np.array(y1)


def findFirstLastValue(df):
    '''Find the first and the last values for
    each participant and returns 2 dictionaries'''

    v0 = []
    vF = []
    for i in range(len(df)):
        i0 = np.argwhere(np.invert(np.isnan(df.iloc[i])))[0]
        v0 = np.append(v0,df.iloc[i][i0])

        iF = np.argwhere(np.invert(np.isnan(df.iloc[i])))[-1]
        vF = np.append(vF,df.iloc[i][iF])
    return {'v0' : v0, 'vF' : vF}



def loadDistances(treatment="simple",remove_duplicates=False):


	data_dic = pd.read_pickle(dir + "Data/%s_models"%treatment)
	dic = {}

	with open(dir + 'Data/%sG'%treatment, 'r') as f:
		G = jp.decode(f.read())


	for key in data_dic:
		dic[key] = np.array([Distance(G["probs"], t) for t in data_dic[key]])
		index = []
		if remove_duplicates:
			index = np.argwhere(np.diff(dic[key])==0) + 1

		dic[key][index] = np.nan
	return pd.DataFrame(dic)



def plotPowerLawFit(loss,xmin=1,continuousFit=True,addnoise=False,confint=.01,plot=False):
    '''General power law plotting method from
    continuous data or discrete data with noise added
    '''


    loss,rank = rankorder(loss)
    y = rank

    if addnoise:
        x = loss + np.random.rand(len(loss)) - 0.5
    else:
        x = loss



    '''Normalized plot of the empirical distribution'''
    rankNorm = rank/float(rank[-1])
    rankMin = rankNorm[loss <= loss][-1]


    '''Plot of the fitted distribution'''
    mu,confidence,nPoints = pwlaw_fit_cont(x,xmin)
    print mu,confidence,nPoints

    xFit = np.logspace(np.log10(xmin),np.log10(max(loss)))
    yFit = (rankMin-0.03)*1/(xFit/float(xmin))**mu

    if plot:
        pl.loglog(loss,rankNorm,'k.' ,alpha=0.5)
        pl.loglog(xFit,yFit,'k-.')


    '''Add confidence intervals'''

    if confint:
        m,L,U,pvalue = bootstrapping(x,xmin,confint=confint,numiter = -1)
        x,y = rankorder(L)
        yLowerNorm = y/float(y[-1])
        #pl.loglog(x,yMin*yLowerNorm,'m')
        x,y = rankorder(U)
        yUpperNorm = y/float(y[-1])


    return {'x':loss,'y':rankNorm,'xFit':xFit,'yFit':yFit}



def bootstrapping(data,xmin,confint=.05,numiter = -1,plot=False,plotconfint=False):
    '''Bootstrapping power law distribution'''
    data = np.array(data) # make sure the input is an array
    sample = data[data >= xmin]
    mu,confidence,nPoints = pwlaw_fit_cont(sample,xmin) #fit original power law

    f = 1/(sample/float(xmin))**mu
    ksInit = kstest(sample,f)
    #print ksInit

    if nPoints==0:
        print "no value larger than %s"%xmin
        return

    if numiter == -1:
        numiter = round(1./4*(confint)**-2)

    m = np.zeros([numiter,nPoints])
    i = 0
    k = 0
    while i < numiter:
        q2 = pwlaw(len(sample),xmin,mu)[0]
        m[i]=np.sort(q2)
        ks = kstest(q2,f)

        if ks > ksInit:
            k += 1

        i+=1

    pvalue = k/float(numiter)
    U=np.percentile(m,100-confint*100,0)
    L=np.percentile(m,confint,0)

    if plot:
        x,y = rankorder(data)
        yNorm = y/float(y[-1])
        yMin = yNorm[x <= xmin][0]

        pl.loglog(x,yNorm,'k.')

        xFit = np.logspace(np.log10(xmin),np.log10(max(sample)))
        yFit = yMin*1/(xFit/float(xmin))**mu

        pl.loglog(xFit,yFit,'r-')

        if plotconfint:
            x,y = rankorder(L)
            yLowerNorm = y/float(y[-1])
            pl.loglog(x,yMin*yLowerNorm,'m')
            x,y = rankorder(U)
            yUpperNorm = y/float(y[-1])
            pl.loglog(x,yMin*yUpperNorm,'b')



    return m,L,U,pvalue

def kstest(sample1,sample2):
    return np.max(np.abs(sample1 - sample2))

def binning(x,y,bins,log_10=False,confinter=5):
    '''makes a simple binning'''

    x = np.array(x);y = np.array(y)

    if isinstance(bins,int) or isinstance(bins,float):
        bins = np.linspace(np.min(x)*0.9,np.max(x)*1.1,bins)
    else:
        bins = np.array(bins)

    if log_10:
        bins = bins[bins>0]
        c = x > 0
        x = x[c]
        y = y[c]
        bins = np.log10(bins)
        x = np.log10(x)
        y = np.log10(y)

    Tbins = []
    Median = []
    Mean = []
    Sigma =[]
    Perc_Up = []
    Perc_Down = []
    Points=[]


    for i,ix in enumerate(bins):
        if i+2>len(bins):
            break

        c1 = x >= ix
        c2 = x < bins[i+1]
        c=c1*c2

        if len(y[c])>0:
            Tbins = np.append(Tbins,np.median(x[c]))
            Median =  np.append(Median,np.median(y[c]))
            Mean = np.append(Mean,np.mean(y[c]))
            Sigma = np.append(Sigma,np.std(y[c]))
            Perc_Down = np.append(Perc_Down,np.percentile(y[c],confinter))
            Perc_Up = np.append(Perc_Up,np.percentile(y[c],100 - confinter))
            Points = np.append(Points,len(y[c]))


    return {'bins' : Tbins,
            'median' : Median,
            'mean' : Mean,
            'stdDev' : Sigma,
            'percDown' :Perc_Down,
            'percUp' :Perc_Up,
            'nPoints' : Points}
