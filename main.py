from probabilisticAlgorithm import probabilisticAlg
from tools import load_data, write_data, readCsv
from calculations import evaluationResults
import numpy as np

def my_range(start, end, step):
    
    while start <= end:
        yield start
        start += step

#Load Files
print "Loading Files...\n"
StrainIDs, StestIDs, StrainFeatures, StestQueries, GtrainIDs, GtestIDs, GtrainFeatures, GtestQueries, wordsTags, trainTFIDF, testQueriesTextual=load_data()
print "---------------------------------------------------------------------"
k = input("Choose the number of images you want to retrieve (1-4500): ")
print "---------------------------------------------------------------------"
q = input("Choose the number of queries you want to make (1-499): ")
print "---------------------------------------------------------------------"
w = input("Choose the number of top-k words you want to take into account (1-260). Type 100: ")
print "---------------------------------------------------------------------"
#Probabilistic Image Annotation Algorithm
print "Probabilistic Image Annotation Algorithm starting ..."
print "---------------------------------------------------------------------"
print "Probabilistic Image Annotation Algorithm for DSIFT"
print "---------------------------------------------------------------------"
allQueriesS = probabilisticAlg(k, q, w, StrainIDs, StestIDs, StrainFeatures, StestQueries, wordsTags, trainTFIDF, testQueriesTextual)
print "Write results in files"
write_data(allQueriesS, wordsTags, "DSIFT")

print "Calculate Evaluation Results (Precision, Recall, F1) for DSIFT"
predsTags = []
for query in allQueriesS:
    predsTags.append(query["predictions"][:5])

N, avgPrecision, avgRecall, avgF1 = evaluationResults(allQueriesS, predsTags, wordsTags)
print "Average Precision: %f" %avgPrecision
print "Average Recall: %f" %avgRecall
print "Average F1: %f" %avgF1
print "Word Tags that found: %d" %N
print "---------------------------------------------------------------------"
print "Probabilistic Image Annotation Algorithm for GBOC"
print "---------------------------------------------------------------------"
allQueriesG = probabilisticAlg(k, q, w, GtrainIDs, GtestIDs, GtrainFeatures, GtestQueries, wordsTags, trainTFIDF, testQueriesTextual)

print "Calculate Evaluation Results (Precision, Recall, F1) for GBOC"
predsTags = []
for query in allQueriesG:
    predsTags.append(query["predictions"][:5])
 
N, avgPrecision, avgRecall, avgF1 = evaluationResults(allQueriesG, predsTags, wordsTags)
print "Average Precision: %f" %avgPrecision
print "Average Recall: %f" %avgRecall
print "Average F1: %f" %avgF1
print "Word Tags that found: %d" %N

print "Write results in files"
write_data(allQueriesG, wordsTags, "GBOC")
print "---------------------------------------------------------------------"
print "Probabilistic Image Annotation Algorithm for DIFT and GBOC"
print "---------------------------------------------------------------------"
#combine arrays
SGtrainFeatures = np.vstack((StrainFeatures.T, GtrainFeatures.T)).T
SGtestQueries = np.vstack((StestQueries.T, GtestQueries.T)).T

allQueriesSG = probabilisticAlg(k, q, w, GtrainIDs, GtestIDs, SGtrainFeatures, SGtestQueries, wordsTags, trainTFIDF, testQueriesTextual)

print "Calculate Evaluation Results (Precision, Recall, F1) for DSIFT and GBOC"
predsTags = []
for query in allQueriesSG:
    predsTags.append(query["predictions"][:5])
 
N, avgPrecision, avgRecall, avgF1 = evaluationResults(allQueriesSG, predsTags, wordsTags)
print "Average Precision: %f" %avgPrecision
print "Average Recall: %f" %avgRecall
print "Average F1: %f" %avgF1
print "Word Tags that found: %d" %N

print "Write results in files"
write_data(allQueriesSG, wordsTags, "DSIFTGBOC")
print "---------------------------------------------------------------------"
print "Calculate Evaluation Results (Precision, Recall, F1) for Data Fusion"
DSIFT_Results = readCsv("DSIFT_Results")
GBOC_Results = readCsv("GBOC_Results")

#data fusion
predsTags = []
for i in my_range(1,len(DSIFT_Results)-1,100): #for loop for every query
    scoresList = []
    for word in range(len(wordsTags)):
        score = 0
        flag1 = 0
        flag2 = 0
        for j in range(i,i+100):
            rowSIFT = DSIFT_Results[j]
            rowGBOC = GBOC_Results[j]
            
            if word == int(rowSIFT[2]):
                flag1 = 1
                score += float(rowSIFT[4])
            
            if word == int(rowGBOC[2]):
                flag2 = 1
                score += float(rowGBOC[4])
            
        if flag1 == 1 and flag2 == 1:
            scoresList.append((wordsTags[word],score))
    
    scoresSort = sorted(scoresList, key=lambda x: float(x[1]), reverse = True)

    pred = np.array(scoresSort)
    predsTags.append(pred[:5,0])

N, avgPrecision, avgRecall, avgF1 = evaluationResults(allQueriesS, predsTags, wordsTags)

print "Average Precision: %f" %avgPrecision
print "Average Recall: %f" %avgRecall
print "Average F1: %f" %avgF1
print "Word Tags that found: %d" %N