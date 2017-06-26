from calculations import priorProbabilities, wordProbabilities, visualProbabilities, FJ
from tools import combineNsortR, load_data, combineNsortA
import numpy as np
import math

#Load Files
print "Loading Files...\n"
trainIDs, testIDs, trainFeatures, testQueries, wordsTags, trainTFIDF, testQueriesTextual=load_data()
print "---------------------------------------------------------------------"
k = input("Choose the number of images you want to retrieve (1-4500): ")
print "---------------------------------------------------------------------"
q = input("Choose the number of queries you want to make (1-499): ")
print "---------------------------------------------------------------------"
w = input("Choose the number of top-k words you want to take into account (1-374): ")
print "---------------------------------------------------------------------"
#Probabilistic Image Annotation Algorithm
print "Probabilistic Image Annotation Algorithm starting ..."
print "---------------------------------------------------------------------"
AllF1 = 0
for i in range(q):
    
    print "Test Query %d." %i
    print "---------------------------------------------------------------------"
    #step 1. For an unknown image, g, retrieve the top-k images from the train set
    print "Step 1: Retrieve the top-%s images from the train set...\n" %k
    
    #similarity = g.T * trainFeatures
    similarity = np.dot(testQueries[i], trainFeatures.T)
    #combine similarity vector with trainIDs vector and sort
    similarity_ID = combineNsortR(similarity, trainIDs)
    sorted_trainIDs = similarity_ID[:,1]
      
    print "Retrieval is over."
    print "---------------------------------------------------------------------"
    
    #step 2. Calculate fj , j = 1 ,..., |T|
    print "Step 2: Calculate fj , j = 1 ,..., |T| ...\n"
    
    pProb = priorProbabilities(k)
    print "Prior Probabilities calculated"
    
    #combine similarity vector with trainTFIDF array and sort
    similarity_trainTFIDF = combineNsortR(similarity, trainTFIDF.T)
    #delete first column
    similarity_trainTFIDF = np.delete(similarity_trainTFIDF, (0), axis=1)
    #combine similarity vector with trainFeatures array and sort
    similarity_trainFeatures = combineNsortR(similarity, trainFeatures.T)
    #delete first column
    similarity_trainFeatures = np.delete(similarity_trainFeatures, (0), axis=1)
    T = len(trainIDs)
    
    vProb = visualProbabilities(similarity_trainFeatures, k, T)
    print "Visual Probabilities calculated"
    
    fjList = []
    for j in range(len(wordsTags)):
        wProb = wordProbabilities(similarity_trainTFIDF, j, k, T)
        fj = FJ(pProb, wProb, vProb)
        fjList.append(fj)
    
    fjArray = np.array(fjList)
    
    print "Calculations are over."
    print "---------------------------------------------------------------------"
    #step 3. Calculate m = max(f1, ..., f|T|)
    print "Step 3. Calculate max fjs"
    maxFjsList = []
    for fj in fjList:
        m = np.max(fj)
        maxFjsList.append(m)
    
    maxFjsVector = np.array(maxFjsList)

    print "Maximums calculated"
    print "---------------------------------------------------------------------"
    #step 4. For each w, calculate logp(w,v1,...,vn) = m + log(sum(exp(fj-m)))
    print "Step 4. For each w calculate logp(w,v1,...,vn)."
    
    finalPreds = []
    for l in range(len(wordsTags)):
        m = maxFjsVector[l]
        sumPreds = 0
        for z in range(k):
            sumPreds += math.exp(fjArray[l][z] - m)
        
        finalPreds.append(m + math.log(sumPreds))
    
    print "Calculations are over."
    print "---------------------------------------------------------------------"
    #step5. Take k keywords wi with the largest logp(w, v1, ...,vn)
    print "Step 5. Take k keywords wi with the largest logp(w, v1, ...,vn)"
    
    #Descending sorting 
    finalPredsVector = np.array(finalPreds)
    
    finalPredsArray = combineNsortA(finalPredsVector,wordsTags)
    finalQueryArray = combineNsortR(testQueriesTextual[i],wordsTags)
   
    predsTags = finalPredsArray[:w,1]
    realTags = finalQueryArray[:4,1]
    print predsTags
    print realTags
    tp = 0
    fp = 0
    for i in predsTags:
        if i in realTags:
            tp += 1
        else:
            fp += 1
            
    fn = 0
    for i in realTags:
        if i not in predsTags:
            fn += 1
    
    precision = float(tp) / (tp + fp)
    print "Precision = %f" %precision
    
    recall = float(tp) / (tp + fn)
    print "Recall = %f" %recall
    if precision == 0 and recall == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * recall) / (precision + recall)
    print "F1 = %f" %F1
    
    AllF1 += F1
        
    print "---------------------------------------------------------------------"

avF1 = AllF1 / q
print "Average F1 = %f" %avF1