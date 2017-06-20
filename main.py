import readFiles

#Load Files
print "Loading Files...\n"
trainIDs, testIDs, trainFeatures, testQueries, wordsTags, trainTFIDF, testQueriesTextual=readFiles.load_data()

print testQueries[0]
#retrieved_images = input("Choose the number of images you want to retrieve (1-4999): ")
#Probabilistic Image Annotation Algorithm
#step1. For an unknown image, g, retrieve the top-k images from the train set


