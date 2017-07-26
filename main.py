from probabilisticAlgorithm import probabilisticAlg
from tools import load_data, write_data
from calculations import precisionRecall, f1Calc
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

print "Calculate Evaluation Results (Precision, Recall, F1) for DSIFT"
precisionRecallListS = precisionRecall(allQueriesS, wordsTags)
F1S = f1Calc(precisionRecallListS)

print "Write results in files"
write_data(allQueriesS, F1S, wordsTags, "DSIFT")
print "---------------------------------------------------------------------"
print "Probabilistic Image Annotation Algorithm for GBOC"
print "---------------------------------------------------------------------"
allQueriesG = probabilisticAlg(k, q, w, GtrainIDs, GtestIDs, GtrainFeatures, GtestQueries, wordsTags, trainTFIDF, testQueriesTextual)
print "Calculate Evaluation Results (Precision, Recall, F1) for GBOC"
precisionRecallListS = precisionRecall(allQueriesS, wordsTags)
F1S = f1Calc(precisionRecallListS)

print "Write results in files"
write_data(allQueriesS, F1S, wordsTags, "GBOC")
print "---------------------------------------------------------------------"