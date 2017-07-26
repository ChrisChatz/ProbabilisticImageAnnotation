import math
import numpy as np
from scipy import spatial

def cosineSimilarity(query, trainF):
    
    result = []
    for i in range(len(trainF)):
        result.append(1 - spatial.distance.cosine(query, trainF[i]))
        
    similarity = np.array(result)
    
    return similarity
    
def sigmoid(x):
    
    #numerically-stable sigmoid function
    return math.exp(-np.logaddexp(0, -x))

def priorProbabilities(k, a = 0.05):
    
    #calculate prior probabilities p(j) = 1 / (1 + exp(-1/r))
    priorProbList = []
    
    for i in range(k) :
        priorProbList.append(sigmoid(a/(i+1)))
        
    priorProbArray = np.array(priorProbList)

    return priorProbArray

def wordProbabilities(sorted_trainTFIDF, word, k, T, a = 0.1):
    
    oneWord = sorted_trainTFIDF[:,word]
    wT = np.count_nonzero(oneWord)
    perImage = sorted_trainTFIDF[ : k]
    
    wordProbList = []
    
    for i in perImage :
        J = np.count_nonzero(i)
        if i[word] != 0 :
            wj = 1
        else:
            wj = 0
        wordProbList.append(((1 - a) * (float(wj)/J)) + (a * (float(wT)/T)))
       
    wordProbArray = np.array(wordProbList)
    
    return wordProbArray

def visualProbabilities(sorted_trainFeatures, k, T, b = 0.9):
    
    visualAllLists = []
    for j in range(np.shape(sorted_trainFeatures)[1]):
        
        oneFeature = sorted_trainFeatures[ :, j]
        vT = np.count_nonzero(oneFeature)
        perImage = sorted_trainFeatures[ : k]
        visualProbList = []
        
        for i in perImage :
            J = np.count_nonzero(i)
            vj = i[j]
            visualProbList.append(((1 - b) * (float(vj)/J)) + (b * (float(vT)/T)))
                
        visualAllLists.append(visualProbList)
    
    visualProbArray = np.array(visualAllLists)                       
    return visualProbArray.T
    
def FJ(prior, word, virtual):
    
    logprior = np.log(prior + (10**-10)) 
#    print logprior.shape
    logword = np.log(word + (10**-10))
#    print logword.shape
    logvirtual = np.log(virtual + (10**-10))
#    print logvirtual.shape
    sumlogvirtual = np.sum(logvirtual, axis = 1) 
#    print sumlogvirtual.shape
    f = logprior + logword + sumlogvirtual
#    print f.shape
#    print "--------"
  
    return f

def precisionRecall(queries,tags):
    
    precisionRecallList = []
    for word in tags:
        tp = 0
        fp = 0
        gs = 0
        for query in queries:
            if word in query["predictions"] and word in query["realTags"]:
                tp += 1
            elif word in query["predictions"] and word not in query["realTags"]:
                fp += 1
                
            if word in query["predictions"]:
                gs += 1
        
        if gs == 0:
            recall = 0
        else:
            recall = float(tp) / gs
                          
        if tp + fp == 0:
            precision = 0
        else:
            precision = float(tp) / (tp+fp)
            
        precisionRecallList.append((recall,precision))

    return precisionRecallList

def f1Calc(preRecList):
    
    f1List = []
    for pre, rec in preRecList:
        if pre == 0 and rec == 0:
            F1 = 0
        else:
            F1 = 2 * (pre * rec) / (pre + rec)
        
        f1List.append(F1)
        
    return f1List