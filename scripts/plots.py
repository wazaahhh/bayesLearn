        pl.xlim(0,2500)
        pl.legend(loc=0)
        pl.savefig(name+'.eps')

distance_plot(simple_distances, complex_distances)


#pl.show()
import string
#ps = list(zip(string.ascii_lowercase, simpleG['probs']))

from scipy import random


from numpy import zeros

simpleProbs     =simpleG['probs']
complexProbs    =complexG['probs']
#fmereqPrior[-2] = 2

#print(Distance(simpleProbs, freqPrior))



def rational_learning(simpleProbs, complexProbs,time_sample = 2500, sample =1000, sd={}, cd={}):
    for j in range(sample):
        simplePrior = list(zeros(len(simpleProbs)))
        complexPrior =list(zeros(len(complexProbs)))
        simple_distances = list(zeros(time_sample))
        complex_distances= list(zeros(time_sample))
        for t in range(time_sample):
            simple_rand =random.random()
            complex_rand = random.random()
            simple_cum =0
            complex_cum=0
            for i in range(len(simpleProbs)):
                simple_cum+=simpleProbs[i]
                if simple_cum>simple_rand:
                    simplePrior[i]+=1
                    simple_distances[t] =Distance(simplePrior, simpleProbs)
                    break
            for i in range(len(complexProbs)):
                complex_cum+=complexProbs[i]
                if complex_cum>complex_rand:
                    complexPrior[i]+=1
                    complex_distances[t] = Distance(complexPrior, complexProbs)
                    break

        sd[j]=simple_distances
        cd[j]=complex_distances
    return pd.DataFrame(sd), pd.DataFrame(cd)

simple_rational, complex_rational = rational_learning(simpleProbs, complexProbs)
distance_plot(simple_rational, complex_rational, name="rational")
"plots.py" 109L, 3198C                                                                                                                                                                    106,5         Bot
