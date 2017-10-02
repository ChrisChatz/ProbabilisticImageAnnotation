import math
import numpy as np
from scipy import spatial

def cosineSimilarity(query, trainF):
    
    result = []
    for i in range(len(trainF)):
        result.append(1 - spatial.distance.cosine(query, trainF[i]))
        
    similarity = np.array(result)
    
    return similarity
    
def priorProbabilities(k, a = 0.3):
    
    #calculate prior probabilities p(j) = 1 / (1 + exp(a*r))
    priorProbList = []
    
    for i in range(k) :
        priorProbList.append(1/(1+math.exp(a*(i+1))))
        
    priorProbArray = np.array(priorProbList)

    return priorProbArray

def wordProbabilities(sorted_trainTFIDF, a = 0.1):
    
    epsilon = 1e-4
    #total sum of similarity_trainTFIDF
    totalSumWords = np.count_nonzero(sorted_trainTFIDF)
    #sum of every column of similarity_trainTFIDF
    columnSumWords = np.count_nonzero(sorted_trainTFIDF, axis = 0)
    #division of every column with total sum
    columnDivTotalWords = np.divide(columnSumWords, float(totalSumWords))
    #sum of every row of image
    rowSumWords = np.count_nonzero(sorted_trainTFIDF, axis = 1)
    #division of every cell of similarity_trainTFIDF with rowSumWords
    wordDivRowSum = np.divide(sorted_trainTFIDF ,(rowSumWords[:, None] + epsilon))
    
    wordProbArray = ((1-a)*wordDivRowSum) + (a*columnDivTotalWords)
    
    return wordProbArray

def visualProbabilities(sorted_trainFeatures, b = 0.9):
    
    #total sum of similarity_trainFeatures
    totalSumFeatures = np.sum(sorted_trainFeatures)
    #sum of every column of similarity_trainFeatures
    columnSumFeatures = sorted_trainFeatures.sum(axis = 0)
    #division of every column with total sum
    columnDivTotalFeatures = np.divide(columnSumFeatures, totalSumFeatures)
    #sum of every row of image
    rowSumFeatures = sorted_trainFeatures.sum(axis = 1)
    #division of every cell of sorted_trainFeatures with rowSumFeatures
    visualDivRowSum = np.divide(sorted_trainFeatures, rowSumFeatures[:, None])
    
    visualProbArray = ((1-b)*visualDivRowSum) + (b*columnDivTotalFeatures)
    
    return visualProbArray
    
def FJ(prior, word, visual, length):
    
    logprior = np.log(prior + (10**-10))
   
    logword = np.log(word + (10**-10))
    
    logvisual = np.log(visual + (10**-10))
    
    sumlogvisual = np.sum(logvisual, axis = 1)
    
    temp = logprior + sumlogvisual
    temp1 = temp [:, None]
    for i in range(length-1):
        temp1 = np.concatenate((temp1,temp[:,None]), axis = 1)
    
    f = temp1 + logword
    
    return f

def evaluationResults(queries, preds, tags):
    
    precisionRecallF1List = []
    N = 0
    for word in tags:
        tp = 0
        fp = 0
        fn = 0
        
        counter = 0
        for query in queries:
            if word in preds[counter] and word in query["realTags"]:
                tp += 1
            elif word in preds[counter] and word not in query["realTags"]:
                fp += 1
                
            if word not in preds[counter] and word in query["realTags"]:
                fn += 1
                
            counter += 1
        
        if tp != 0:
            N += 1
            
        if tp + fn == 0:
            recall = 0
        else:
            recall = float(tp) / (tp+fn)
                          
        if tp + fp == 0:
            precision = 0
        else:
            precision = float(tp) / (tp+fp)
        
            
        precisionRecallF1List.append((recall,precision,f1Calc(recall, precision)))
    
    precisionRecallF1Array = np.array(precisionRecallF1List)

    avgRec = np.mean(precisionRecallF1Array[:, :1])
    avgPre = np.mean(precisionRecallF1Array[:, 1:2])
    avgF1 =  np.mean(precisionRecallF1Array[:, :-1])
    return N, avgPre, avgRec, avgF1

def f1Calc(rec, pre):
    
    if pre == 0 and rec == 0:
        F1 = 0
    else:
        F1 = 2 * (pre * rec) / (pre + rec)
        
    return F1