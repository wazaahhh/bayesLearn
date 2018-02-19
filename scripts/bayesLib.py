import sys
import pandas as pd
import csv
import numpy as np
import pylab as pl
import jsonpickle as jp
import simplejson
from scipy.stats import linregress
from scipy import stats  as S
from Entro import *

sys.path.append("/Users/maithoma/work/python/")
from tm_python_lib import *
from fitting_tools import *
import adaptive_kernel_tom as AK

dir = "/Users/maithoma/github/bayesLearn/scripts/"

fig_width_pt = 420.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width  # *golden_mean      # height in inches
fig_size = [fig_width, fig_height]

params = {'backend': 'ps',
          'axes.labelsize': 25,
          'font.size': 25,
          'legend.fontsize': 18,
          #'title.fontsize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': False,
          'figure.figsize': fig_size}
pl.rcParams.update(params)


def loadDistances(treatment="simple",distanceType="jsd",remove_duplicates=False):
    '''
    Load Score as JSD or Euclidian (sqrt) Distance
    between proposed models and the truth
    '''

    data_dic = pd.read_pickle(dir + "Data/%s_models"%treatment)
    dic = {}

    with open(dir + 'Data/%sG'%treatment, 'r') as f:
        G = jp.decode(f.read())

    for key in data_dic:
        if distanceType == 'jsd':
            dic[key] = np.array([Distance(G["probs"], t) for t in data_dic[key]])
        elif distanceType == 'sqrt':
            dic[key] = np.array([np.sqrt(np.sum((np.array(t) - np.array(G["probs"]))**2)) for t in data_dic[key]])

    index = []
    if remove_duplicates:
        index = np.argwhere(np.diff(dic[key])==0) + 1

    dic[key][index] = np.nan
    return pd.DataFrame(dic)

def makeDicParticipants(div_displace=False,grid_distance=False,overwrite=False,save=True):
    '''makes a dictionary with data for each participant.

       Options :
       - div_displace : adds divergence and diversity metrics
       - grid_distance : distance metrics assuming a n-dimensional grid (solution space partitionning)
       - overwrite : overwrite existing file
       - save : save file
    '''

    if not overwrite:
        try:
            dicParticipants = simplejson.loads(open(dir + "Data/dicParticipants.json",'rb').read())
            return dicParticipants
        except:
            print "json file not found. Rebuilding. Please be patient. It takes some time."
            pass

    dicParticipants = {}

    for treatment in ['simple','complex']:
        print treatment
        with open(dir + 'Data/%sG'%treatment, 'r') as f:
            trueG = jp.decode(f.read())['probs']

        distances = {}
        distances['jsd'] = loadDistances(treatment=treatment,distanceType='jsd',remove_duplicates=False)
        distances['sqrt'] = loadDistances(treatment=treatment,distanceType='sqrt',remove_duplicates=False)
        treatment_dic = pd.read_pickle(dir + "Data/%s_models"%treatment)

        for k,key in enumerate(treatment_dic.keys()): #cleanup data

            index = [] # time index of change occurrences
            V = [[0]*len(trueG)]

            dstTruth = {'sqrt' : [],'jsd' : []}

            for i,v in enumerate(treatment_dic[key]):
                if not isinstance(v, list):continue # skip NaNs

                #try:
                if not list(v)==V[-1]: # skip unchanged states
                    if V[-1] == [0]*len(trueG):
                        V = [v]
                    else:
                        V.append(list(v))
                    index.append(i)
                    '''distance from the truth (i.e., the solution)'''
                    dstTruth['sqrt'].append(distances['jsd'][key][i])
                    dstTruth['jsd'].append(distances['sqrt'][key][i])

            dicParticipants[key] = {'treatment' : treatment,
                                    'index' : index,
                                    'models' : V,
                                    'dstTruth' : dstTruth,
                                    'truth' : trueG
                                   }

    for i,key in enumerate(dicParticipants.keys()):
        if div_displace:
            dicParticipants[key]['dd'] = divergence_displacement(dicParticipants,key)
        if grid_distance:
            dic = gridDistance(dicParticipants,key,digits=1)
            dicParticipants[key]['stepDST'] = dic['stepDST']
            dicParticipants[key]['visitations'] = dic['visitations']

    if save:
        try:
            f = open(dir + "Data/dicParticipants.json",'wb')
            f.write(simplejson.dumps(dicParticipants))
            f.close()
        except:
            print "failed saving json file"
            pass

    return dicParticipants


def makeAllDic(dicParticipants,overwrite=False,save=True):
    '''makes an aggregate of all partipants per treatment (simple or complex)'''

    if not overwrite:
        try:
            allDic = simplejson.loads(open(dir + "Data/allDic.json",'rb').read())
            return allDic
        except:
            print "json file not found. Rebuilding. Please be patient. It may take some time."
            pass

    allDic = {'simple' : {'plot': {'color' : 'blue','marker' : 's'},
                          'values' : {}},
              'complex' :{'plot': {'color' : 'red','marker' : 'o'},
                         'values' :{}}
             }

    for treatment in allDic.keys():

        dR = []
        dT = []
        T = []

        Tnew = []
        St = []

        Divergence = []

        Vr = []

        minScore = []
        meanScore = []
        sMax = []

        maxSt = []

        dScore = []

        for key in dicParticipants.keys():
            if not dicParticipants[key]['treatment'] == treatment : continue

            # Divergence
            divergence = np.array(dicParticipants[key]['dd']['divergence'])
            Divergence = np.append(Divergence,divergence)

            # Displacement
            dr = np.array(dicParticipants[key]['dd']['displacement'])
            dR = np.append(dR,dr)

            # Waiting Time
            dt = np.diff(dicParticipants[key]['index'])
            dT = np.append(dT,dt)
            T = np.concatenate([T,dicParticipants[key]['index'][1:]])
            #Explore / Return
            t,s = np.array(zip(*dicParticipants[key]['visitations']['explore_new']))
            sMax.append(max(s))
            Tnew = np.append(Tnew,t)
            St = np.append(St,s/np.mean(s))
            Vr = np.concatenate([Vr,dicParticipants[key]['visitations']['return_count']])

            maxSt.append(np.max(s))

            # Score
            #print len(dr),len(np.diff(dic[key]['dstTruth']['sqrt']))
            dScore = np.concatenate([dScore,np.diff(dicParticipants[key]['dstTruth']['sqrt'])])
            minS = np.min(dicParticipants[key]['dstTruth']['sqrt'])
            minScore.append(minS)
            meanS = np.mean(dicParticipants[key]['dstTruth']['sqrt'])
            meanScore.append(meanS)

        allDic[treatment]['values'] = {'dR' : list(dR),
                             'dT' : list(dT),
                             'T' : list(T),
                             'Tnew' : list(Tnew),
                             'St' : list(St),
                             'Vr' : list(Vr),
                             'minScore': list(minScore),
                             'meanScore' : list(meanScore),
                             'maxSt' : list(maxSt),
                             'dScore' : list(dScore),
                             'Divergence': list(Divergence)}

    if save:
        try:
            f = open(dir + "Data/allDic.json",'wb')
            f.write(simplejson.dumps(allDic))
            f.close()
        except:
            print "failed saving json file"
            pass

    return allDic


def divergence_displacement(dicParticipants,participantKey):

    divergence = []
    displacement = []
    nPointDiversity = []
    nPointJSD = []

    V = dicParticipants[participantKey]['models']

    for i,v in enumerate(V):
        if i==0:continue
        '''divergence and displacement'''
        M = MixedN(V[:i])
        divergence.append(JensenShannonDivergence(v,M))
        displacement.append(np.sqrt(np.sum((np.array(v) - np.array(V[i - 1]))**2)))
        nPointDiversity.append(Diversity(V[:i]))
        nPointJSD.append(N_point_JSD(V[:i]))

    return {'divergence' : list(divergence),
           'displacement': list(displacement),
           'nPointDiversity' : list(nPointDiversity),
           'nPointJSD' : list(nPointJSD)}


def gridDistance(dicParticipants,participantKey,digits=1):
    '''
    Performs distance measures by assuming that the solution space is a n-dimensional grid.

    Additionally, this function indicates if proposed models are returning
    to previously visited cells, or on the contrary, whether new cells are visited)
    '''

    V = dicParticipants[participantKey]['models']
    trueG = dicParticipants[participantKey]['truth']
    trueG_b = np.floor(np.array(trueG)*10**digits)/10**digits
    trueG_b_str = ",".join(["%s"%np.round(r,digits) for r in trueG_b])

    index = dicParticipants[participantKey]['index']

    stepDST = {}

    visitations = {'return' : {}, 'return_count': [],'explore_new' : []}
    k = 0
    SiteExpl = []

    for i,ix in enumerate(V):
        #if not isinstance(ix, list):continue # skip NaNs
        dic = {'jsd': [],'lsqr' : [],'timeDST' : []}
        for j,jx in enumerate(V[:i]):
            dstJSD = Distance(np.round(ix,10),np.round(jx,10)) # distance
            dstLSQR = np.sqrt(np.sum((np.array(ix) - np.array(jx))**2))

            dic['jsd'].append(dstJSD)
            dic['lsqr'].append(dstLSQR)
            #o = np.argsort(dic[distance]) # index by JSD/LSQR distance (small to large)
            dic['timeDST'].append(index[i]-index[j])

        #dic['order'] = list(S.rankdata(np.array(dic[distance]),method="ordinal") - 1)
        #dic['order'] = list(np.argsort(dic[distance])-1)
        stepDST[index[i]] = dic

        #Unique and Return Count
        b = list(np.floor(np.array(ix)*10**digits)/10**digits)
        b_str = ",".join(["%s"%np.round(r,digits) for r in b])

        if trueG_b_str == b_str:
            print k,i,b_str,trueG_b_str

        try:
            visitations['return'][b_str].append(index[i])
        except:
            visitations['return'][b_str] = [index[i]]
            k += 1
            visitations['explore_new'].append((index[i],k))


    visitations['return_count'] = [len(np.argwhere(np.diff(r)>1))+1 for r in visitations['return'].values()]


    return {'stepDST' :stepDST, 'visitations' : visitations}




'''Analytic'''
def msd(V,n):
    '''Mean Square Distance'''
    return 1/float(n)*np.sum(np.sum((V[1:n]-V[0])**2,0))

def rankorder(x):
	x1 = list(np.sort(x))
	x1.reverse()
	y1 = range(1,len(x1)+1)
	return np.array(x1),np.array(y1)


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


# def findFirstLastValue(df):
#     '''Find the first and the last values for
#     each participant and returns 2 dictionaries'''
#
#     v0 = []
#     vF = []
#     for i in range(len(df)):
#         i0 = np.argwhere(np.invert(np.isnan(df.iloc[i])))[0]
#         v0 = np.append(v0,df.iloc[i][i0])
#
#         iF = np.argwhere(np.invert(np.isnan(df.iloc[i])))[-1]
#         vF = np.append(vF,df.iloc[i][iF])
#     return {'v0' : v0, 'vF' : vF}



''' Handle Data / Build Dictionaries'''
