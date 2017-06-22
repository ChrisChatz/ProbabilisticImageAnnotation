import readFiles
from calculations import priorProbabilities, wordProbabilities
import numpy as np

#Load Files
print "Loading Files...\n"
trainIDs, testIDs, trainFeatures, testQueries, wordsTags, trainTFIDF, testQueriesTextual=readFiles.load_data()
print "---------------------------------------------------------------------"
k = input("Choose the number of images you want to retrieve (1-4500): ")
print "---------------------------------------------------------------------"

#Probabilistic Image Annotation Algorithm
print "Probabilistic Image Annotation Algorithm starting . . ."
print "---------------------------------------------------------------------"

#step 1. For an unknown image, g, retrieve the top-k images from the train set
print "Step 1: Retrieve the top-%s images from the train set\n" %k

#similarity = g.T * trainFeatures
similarity = np.dot(testQueries[0], trainFeatures.T)

#combine similarity vector with trainIDs vector
similarity_ID = np.vstack((similarity, trainIDs)).T

#sort similarity_ID array in descending order
similarity_ID = np.sort(similarity_ID, axis=0, kind='quicksort')
similarity_ID = similarity_ID[::-1]
sorted_trainIDs = similarity_ID[:,1]

print "The following images are retrieved: "
#print top-k retrieved images
counter = 0
for x, y in similarity_ID :
    print "Similarity: %s | Image ID: %s" %(x, y)
    counter += 1
    if counter == k:
        break
    
print "---------------------------------------------------------------------"

#step 2. Calculate fj , j = 1 ,..., |T|
print "Step 2: Calculate fj , j = 1 ,..., |T|\n"

pProb = priorProbabilities(sorted_trainIDs[:k])

#for x, y in pProb:
#    print "prior: %s | Image ID: %s" %(x, y)

#sort TFIDF
similarity_trainTFIDF = np.vstack((similarity, trainTFIDF.T)).T
similarity_trainTFIDF = np.sort(similarity_trainTFIDF, axis=0, kind='quicksort')
similarity_trainTFIDF = similarity_trainTFIDF[::-1]
sorted_trainTFIDF = np.delete(similarity_trainTFIDF,(0), axis=1)
#sorted_trainTFIDF = sorted_trainTFIDF[:k]
T = len(trainIDs)
#wProb = wordProbabilities(sorted_trainTFIDF, k, T)

print "---------------------------------------------------------------------"

#step 3. Calculate m = max(f1, ..., f|T|)
print "Step 3. Calculate max fj"

    

