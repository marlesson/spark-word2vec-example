#
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import Word2Vec
import numpy as np
import re

# $example off$
if __name__ == "__main__":
    sc = SparkContext(appName="Word2Vec")
    sqlContext = SQLContext(sc)

    # load
    trainPath = "data/labeledTrainData.tsv"
    testPath  = "data/testData.tsv"
    path_wordvecs = "data/word2vec"

    # load text
    def skipHeaders(idx, iter): return "" if idx == 0 else iter 

    trainFile = sc.textFile(trainPath)\
                    .mapPartitionsWithIndex(skipHeaders)\
                    .map(lambda row: row.split("\t"))
            
    testFile  = sc.textFile(testPath)\
                    .mapPartitionsWithIndex(skipHeaders)\
                    .map(lambda row: row.split("\t"))

    # To sample
    def toSample(segments):
        if len(segments) == 3:
            return dict(zip(['id', 'sentiment', 'review'], segments))
        else:
            return dict(zip(['id', 'review'], segments))
            

    trainSamples = trainFile.map(toSample)
    testSamples  = testFile.map(toSample)

    # Clean Html
    def cleanHtml(str):
        tag = re.compile(r'<(?!\/?a(?=>|\s.*>))\/?.*?>')
        tag.sub('', str)
        return str

    def cleanSamplesHtml(sample):
        sample['review'] = cleanHtml(sample['review'])
        return sample

    # Words only
    def cleanWord(str):
        return str.split(" ")

    def cleanWord(str):
        return " ".join([x.lower() for x in str.split(" ")])

    def wordOnlySample(sample):
        sample['review'] = cleanWord(sample['review'])
        return sample

    wordOnlyTrainSample = trainSamples.map(cleanSamplesHtml).map(wordOnlySample).cache()
    wordOnlyTestSample  = testSamples.map(cleanSamplesHtml).map(wordOnlySample)

    # Word2Vec
    reviewWordsPairs = wordOnlyTrainSample.map(lambda samples: samples['review'].split(" "))

    # Train Word2Vec
    print("Start Training Word2Vec --->")
    model = Word2Vec().fit(reviewWordsPairs)
    print("Finished Training")

    print(model.transform("london"))

    # Save and Load Vectors
    model.save(sc, path_wordvecs)


    lookup = sqlContext.read.parquet("{}/data".format(path_wordvecs)).alias("lookup")
    lookup_bd = sc.broadcast(lookup.rdd.collectAsMap())

    # Create a feature vectors
    def wordFeatures(words): 
        features = []
        for w in words:
            f = lookup_bd.value.get(w)
            if not None is f:
                features.append(f)
        return np.array(features)

    def avgWordFeatures(words): 
        return np.mean(wordFeatures(words), axis=0)    

    def mapFeaturesPair(sample):
        sample['features'] = avgWordFeatures(sample['review'].split(" "))
        return sample

    featuresPair = trainSamples.map(mapFeaturesPair)\
                      .filter(lambda x: isinstance(x['features'], (np.ndarray)))

    trainingSet  = featuresPair.map(lambda x: LabeledPoint(float(x['sentiment']), x['features']))


    print(trainingSet.take(2))

    # Classification
    print("String Learning and evaluating models")
    x_train, x_test = trainingSet.randomSplit([0.7, 0.3])
    model  = SVMWithSGD.train(x_train, 100)

    result = model.predict(x_test.map(lambda x: x['features']))

    print("10 samples:")
    print(x_test.map(lambda lp: (lp.label, model.predict(lp.features))).take(10))

    # Evaluating the model on training data
    labelsAndPreds = x_test.map(lambda p: (p.label, model.predict(p.features)))
    trainErr       = labelsAndPreds.filter(lambda p: p[0] == p[1]).count() / float(labelsAndPreds.count())
    print("Model Accuracy = " + str(trainErr))

    print("<---- done")