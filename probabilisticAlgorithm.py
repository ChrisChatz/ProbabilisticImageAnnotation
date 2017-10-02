from calculations import cosineSimilarity, priorProbabilities, wordProbabilities, visualProbabilities, FJ
from tools import combineNsortR, combineNsortA
import numpy as np
import math

def probabilisticAlg(k, q, w, trainIDs, testIDs, trainFeatures, testQueries, wordsTags, trainTFIDF, testQueriesTextual):
    
    allQueries = []
    #prior probabilities are the same for every query
    pProb = priorProbabilities(k)
        
    for i in range(q):
    
        #step 1. For an unknown image, g, retrieve the top-k images from the train set
        #cosine similarity
        similarity = cosineSimilarity(testQueries[i],trainFeatures)
    
        #step 2. Calculate fj , j = 1 ,..., |T|
        #combine similarity vector with trainTFIDF array and sort
        similarity_trainTFIDF = combineNsortR(similarity, trainTFIDF.T) 
        #delete first column
        similarity_trainTFIDF = np.delete(similarity_trainTFIDF, (0), axis = 1)
        #combine similarity vector with trainFeatures array and sort
        similarity_trainFeatures = combineNsortR(similarity, trainFeatures.T)
        #delete first column
        similarity_trainFeatures = np.delete(similarity_trainFeatures, (0), axis = 1)
        
        #calculate Visual Probabilities
        vProb = visualProbabilities(similarity_trainFeatures[:k])

        #calculate "Word Probabilities
        wProb = wordProbabilities(similarity_trainTFIDF[:k])
        
        #calculate Fjs
        fjArray = FJ(pProb, wProb, vProb, len(wordsTags))
        
        #step 3. Calculate m = max(f1, ..., f|T|)
        maxFjsVector = np.max(fjArray, axis = 0)
        
        #step 4. For each w, calculate logp(w,v1,...,vn) = m + log(sum(exp(fj-m)))
        finalPreds = []
        for l in range(len(wordsTags)):
            m = maxFjsVector[l]
            sumPreds = 0
            for z in range(k):
                sumPreds += math.exp(fjArray[z][l] - m)
            
            finalPreds.append(m + math.log(sumPreds))
            
        #step5. Take k keywords wi with the largest logp(w, v1, ...,vn)

        #Descending sorting 
        finalPredsVector = np.array(finalPreds)
        
        finalPredsArray = combineNsortA(finalPredsVector,wordsTags)
        finalQueryArray = combineNsortR(testQueriesTextual[i],wordsTags)
       
        predsTags = finalPredsArray[:w,1]
        predsScore = finalPredsArray[:w,0]
        realTags = finalQueryArray[:4,1]

        query = {"ID": testIDs[i], "predictions": predsTags, "score": predsScore, "realTags": realTags} 
    
        allQueries.append(query)
    
    return allQueries