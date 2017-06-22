import math
import numpy as np

def priorProbabilities(trainIDs, a = 0.1):
    
    #calculate prior probabilities p(j) = 1 / (1 + exp(-1/r))
    priorProbList = []
    
    counter = 0
    for i in trainIDs :
        priorProbList.append(( 1 / ( 1 + math.exp( -a / ( counter + 1 )))))
        counter += 1
        
    priorProbVector = np.array(priorProbList)
    #combine priorProbVector vector with trainIDs vector
    priorProbVector_ID = np.vstack((priorProbVector, trainIDs)).T

    return priorProbVector_ID

#def wordProbabilities(sorted_trainTFIDF, k, T, a = 0.1):
#    
#    oneWord = sorted_trainTFIDF[:,0]
#    wT = np.count_nonzero(oneWord)
#    
#    perImage = sorted_trainTFIDF[:k]
#    print perImage.shape
#    wordProbList = []
#    for i in perImage :
#        print i.shape
#        J = np.count_nonzero(i)
#        print J
#        print a * (float(wT)/T)
#        wordProbList.append(((1 - a) * i/J) + (a * (wT/T)))
#        
#    wordProbVector = np.array(wordProbList)
#    print wordProbVector.shape

    
    

