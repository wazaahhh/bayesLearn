from bayesLib import *

def rankorder(x):
	x1 = list(np.sort(x))
	x1.reverse()
	y1 = range(1,len(x1)+1)
	return np.array(x1),np.array(y1)


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
