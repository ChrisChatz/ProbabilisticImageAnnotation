from calculations import cosineSimilarity, priorProbabilities, wordProbabilities, visualProbabilities, FJ
from tools import combineNsortR, combineNsortA
import numpy as np
import math

def probabilisticAlg(k, q, w, trainIDs, testIDs, trainFeatures, testQueries, wordsTags, trainTFIDF, testQueriesTextual):
    
    allQueries = []
    #prior probabilities are the same for every query
    pProb = priorProbabilities(k)
    for i in range(q):
        
        print "Test Query %d." %i
        print "---------------------------------------------------------------------"
        #step 1. For an unknown image, g, retrieve the top-k images from the train set
        print "Step 1: Retrieve the top-%s images from the train set...\n" %k
        
        #cosine similarity
        similarity = cosineSimilarity(testQueries[i],trainFeatures)
        
        print "Retrieval is over."
        print "---------------------------------------------------------------------"
        
        #step 2. Calculate fj , j = 1 ,..., |T|
        print "Step 2: Calculate fj , j = 1 ,..., |T| ...\n"
        
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

        query = {"ID": testIDs[i], "predictions": predsTags, "realTags": realTags} 
        allQueries.append(query)
        print "---------------------------------------------------------------------"
    
    return allQueries