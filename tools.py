from numpy import genfromtxt
import numpy as np

def load_data():
    
    trainIDs = genfromtxt('corel_data/DSIFT_Corel/trainIDs.csv', delimiter=',')
    testIDs = genfromtxt('corel_data/DSIFT_Corel/qIDs.csv', delimiter=',')
    trainFeatures = genfromtxt('corel_data/DSIFT_Corel/Ctrain.csv', delimiter=',')
    testQueries = genfromtxt('corel_data/DSIFT_Corel/Queries.csv', delimiter=',')
    
    wordsTags = genfromtxt('corel_data/Textual374_TFIDF/words.csv', dtype='string' ,delimiter=',')
    trainTFIDF = genfromtxt('corel_data/Textual374_TFIDF/Ctrain.csv', delimiter=',')
    testQueriesTextual = genfromtxt('corel_data/Textual374_TFIDF/Queries.csv', delimiter=',')
    
    print "From DSIFT_Corel we have the following arrays:"
    print "trainIDs = %s testIDs = %s trainFeatures = %s TestQueries = %s \n" %(trainIDs.shape, testIDs.shape, trainFeatures.shape, testQueries.shape)
    
    print "From Textual374_TFIDF we have the following arrays:"
    print "wordsTags = %s trainTFIDF = %s testQueriesTextual = %s" %(wordsTags.shape, trainTFIDF.shape, testQueriesTextual.shape)
    
    return trainIDs, testIDs, trainFeatures, testQueries, wordsTags, trainTFIDF, testQueriesTextual

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