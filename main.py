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

#Probabilistic Image Annotation Algorithm
print "Probabilistic Image Annotation Algorithm starting ..."
print "---------------------------------------------------------------------"
for i in range(len(testQueries)):
    
    print "Test Query %d." %i
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
    
    pProb = priorProbabilities(sorted_trainIDs[:k])
    
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
    
    fjList = []
    for i in range(len(wordsTags)):
        wProb = wordProbabilities(similarity_trainTFIDF, i, k, T)
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
    for i in range(len(wordsTags)):
        m = maxFjsVector[i]
        sumPreds = 0
        for j in range(k):
            sumPreds += math.exp(fjArray[i][j] - m)
        
        finalPreds.append(m + math.log(sumPreds))
    
    print "Calculations are over."
    print "---------------------------------------------------------------------"
    #step5. Take k keywords wi with the largest logp(w, v1, ...,vn)
    print "Step 5. Take k keywords wi with the largest logp(w, v1, ...,vn)"
    
    #Descending sorting 
    finalPredsVector = np.array(finalPreds)
    n = 5
    finalPredsArray = combineNsortA(finalPredsVector,wordsTags)
     
    for i in range(n):
        print finalPredsArray[i]
        
    print "---------------------------------------------------------------------"