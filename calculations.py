import math
import numpy as np

def priorProbabilities(k, a = 0.9):
    
    #calculate prior probabilities p(j) = 1 / (1 + exp(-1/r))
    priorProbList = []
    
    for i in range(k) :
        priorProbList.append(( 1 / ( 1 + math.exp( -a / ( i + 1 )))))
        
    priorProbArray = np.array(priorProbList)

    return priorProbArray

def wordProbabilities(sorted_trainTFIDF, word, k, T, a = 0.9):
    
    oneWord = sorted_trainTFIDF[:,word]
    wT = np.count_nonzero(oneWord)
    
    perImage = sorted_trainTFIDF[ : k]
    
    counter=0
    wordProbList = []
    
    for i in perImage :
        J = np.count_nonzero(i)

        if oneWord[counter] != 0 :
            wj = 1
        else:
            wj = 0

        counter += 1
        wordProbList.append(((1 - a) * (float(wj)/J)) + (a * (float(wT)/T)))
     
    wordProbArray = np.array(wordProbList)
   
    return wordProbArray

def visualProbabilities(sorted_trainFeatures, k, T, b = 0.9):
    
    visualAllLists = []
    for j in range(np.shape(sorted_trainFeatures)[1]):
        oneFeature = sorted_trainFeatures[ :, j]
        vT = np.count_nonzero(oneFeature)
        perImage = sorted_trainFeatures[ : k]
        
        counter=0
        visualProbList = []
        
        for i in perImage :
            J = np.count_nonzero(i)
            vj = oneFeature[counter]
            counter += 1
            visualProbList.append(((1 - b) * (float(vj)/J)) + (b * (float(vT)/T)))
            
        visualAllLists.append(visualProbList)
    
    visualProbArray = np.array(visualAllLists)
                                 
    return visualProbArray.T
    
def FJ(prior, word, virtual):
    
    logprior = np.log(prior + (10**-10))
#    print "prior:%f" %logprior
    logword = np.log(word + (10**-10))
#    print "word:%f" %logword
    logvirtual = np.log(virtual + (10**-10))
    sumlogvirtual = np.sum(logvirtual, axis = 1)
#    print "virual:%f" %sumlogvirtual
    
    f = logprior + logword + sumlogvirtual
#    print f
    
    return f