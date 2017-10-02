from numpy import genfromtxt
import numpy as np
import csv

def load_data():
    
    StrainIDs = genfromtxt('corel_data/DSIFT_Corel/trainIDs.csv', delimiter=',')
    StestIDs = genfromtxt('corel_data/DSIFT_Corel/qIDs.csv', delimiter=',')
    StrainFeatures = genfromtxt('corel_data/DSIFT_Corel/Ctrain.csv', delimiter=',')
    StestQueries = genfromtxt('corel_data/DSIFT_Corel/Queries.csv', delimiter=',')
    
    GtrainIDs = genfromtxt('corel_data/GBOC200_Corel/trainIDs.csv', delimiter=',')
    GtestIDs = genfromtxt('corel_data/GBOC200_Corel/qIDs.csv', delimiter=',')
    GtrainFeatures = genfromtxt('corel_data/GBOC200_Corel/Ctrain.csv', delimiter=',')
    GtestQueries = genfromtxt('corel_data/GBOC200_Corel/Queries.csv', delimiter=',')
    
    wordsTags = genfromtxt('corel_data/Textual260_TFIDF/words.csv', dtype='string' ,delimiter=',')
    trainTFIDF = genfromtxt('corel_data/Textual260_TFIDF/Ctrain.csv', delimiter=',')
    testQueriesTextual = genfromtxt('corel_data/Textual260_TFIDF/Queries.csv', delimiter=',')
    
    print "From DSIFT_Corel we have the following arrays:"
    print "trainIDs = %s testIDs = %s trainFeatures = %s TestQueries = %s \n" %(StrainIDs.shape, StestIDs.shape, StrainFeatures.shape, StestQueries.shape)
    
    print "From GBOC200_Corel we have the following arrays:"
    print "trainIDs = %s testIDs = %s trainFeatures = %s TestQueries = %s \n" %(GtrainIDs.shape, GtestIDs.shape, GtrainFeatures.shape, GtestQueries.shape)
    
    print "From Textual260_TFIDF we have the following arrays:"
    print "wordsTags = %s trainTFIDF = %s testQueriesTextual = %s" %(wordsTags.shape, trainTFIDF.shape, testQueriesTextual.shape)
    
    return StrainIDs, StestIDs, StrainFeatures, StestQueries, GtrainIDs, GtestIDs, GtrainFeatures, GtestQueries, wordsTags, trainTFIDF, testQueriesTextual


def write_data(queries, wordsTags, nameOfExp):
        
    with open(nameOfExp+'_Results.csv', 'w') as csvfile:
        fieldnames = ['QIDs', 'Fixed', 'termID', 'Rank', 'Score', 'Name_of_experiment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for query in queries:
            rank = 1
            i = 0
            for word in query['predictions']:
                termID = np.where(wordsTags == word)[0]
                score = query['score'][i]
                i += 1
                writer.writerow({'QIDs': int(query['ID']), 'Fixed': 1, 'termID': int(termID), 'Rank': rank, 'Score': float(score), 'Name_of_experiment': nameOfExp})
                rank += 1

def readCsv(filename):
    
    res = []
    
    with open(filename+'.csv') as csvfile:
        results = csv.reader(csvfile, delimiter=',')
        for row in results:
            res.append(row)
       
    return res

def combineNsortR(a, ida):

    #combine similarity vector with another vector
    a_ida = np.vstack((a, ida)).T
    #array to list to sort it easily
    a_ida_list = a_ida.tolist()
    a_ida_list = sorted(a_ida_list, reverse=True)
    a_ida = np.array(a_ida_list)
    
    return a_ida

def combineNsortA(a, ida):

    #combine similarity vector with another vector
    a_ida = np.vstack((a, ida)).T
    #array to list to sort it easily
    a_ida_list = a_ida.tolist()
    a_ida_list = sorted(a_ida_list)
    a_ida = np.array(a_ida_list)
    
    return a_ida