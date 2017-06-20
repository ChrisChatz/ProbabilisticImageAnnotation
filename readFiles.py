from numpy import genfromtxt

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