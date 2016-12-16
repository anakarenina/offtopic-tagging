import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import operator
import numpy

import settings

CONTENT_TITLE=settings.CONTENT_TITLE
CONTENT_COLUMN=settings.CONTENT_COLUMN
CONTENT_TAGS= settings.CONTENT_TAGS

N_MOST_FREQUENT=100
LABEL_SIZE=3.5
SAMPLE_SIZE= 3000

stemmer= PorterStemmer()

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)


########### Convert csv input files into dataframes###########
biology_pd = pd.read_csv('preprocessedbiology.csv').sample(n=SAMPLE_SIZE)
cooking_pd = pd.read_csv('preprocessedcooking.csv').sample(n=SAMPLE_SIZE)
cryptology_pd = pd.read_csv('preprocessedcrypto.csv').sample(n=SAMPLE_SIZE)
diy_pd = pd.read_csv('preprocesseddyi.csv').sample(n=SAMPLE_SIZE)
robotics_pd = pd.read_csv('preprocessedrobotics.csv').sample(n=1000)
travel_pd = pd.read_csv('preprocessedtravel.csv').sample(n=SAMPLE_SIZE)
#test_pd = pd.read_csv('test.csv')

topics= ['biology', 'cooking', 'crypto', 'dyi', 'robotics', 'travel']

training_files= []
training_files.append(biology_pd)
training_files.append(cooking_pd)
training_files.append(cryptology_pd)
training_files.append(diy_pd)
training_files.append(robotics_pd)
training_files.append(travel_pd)

##to remove punctuation, we can use instead of nltk.word_tokenize
tokenizer= RegexpTokenizer(r'\w+')


def tag_features(word,title,content):
    features = {}

    features["is_in_title"] = stemmer.stem(word) in title

    #normally, if it occurs many times or has many synonyms it can be a tag?
    features["occurrences"]= content.count(stemmer.stem(word))

    features["is_in_content"] = features["occurrences"] >0

    #another feature is if word appear in most frequent tags or if it appears in tags at all, we can go for SE
    #another feature would be to check if words appear in the tag description
    #we can also do something with ontology to check this
    #shall we account for synonyms in occurrences?
    # we can also do something about relations among tags and how often a synonym goes with that so we should add it
    # synonyms = []
    # for ss in wordnet.synsets(word):
    #     synonyms.append(ss.lemma_names())
    #         #I think every word that is not a "word" is a tag
    # features["is_in_wordnet"] = wordnet.synsets(word)

    #example: food of if there are many FOOD types listed?    features["is_superword"] synset1.lowest_common_hypernyms(synset2) ???
    #chech if it is the main noun in the sentence or verb
    #features["is_main']=
    return features

def isTag(word,tags):
    return stemmer.stem(word) in [stemmer.stem(tag) for tag in tags]

#
# def trainingSet():
#
#     for training_file in training_files:
#         # Get the number of reviews based on the dataframe column size
#         num_reviews = training_file["title"].size
#
#         # Initialize an empty list to hold the clean reviews
#         clean_train_reviews = []
#
#         # Loop over each review; create an index i that goes from 0 to the length
#         # of the movie review list
#         for i in xrange( 0, num_reviews ):
#             clean_train_reviews.append( training_file["title"][i] )
#
#         train_data_features = vectorizer.fit_transform(clean_train_reviews)
#         train_data_features = train_data_features.toarray()
#         vocab = vectorizer.get_feature_names()
#         print(vocab)


def trainingSet():

    training= []

    for training_file in training_files:
        for entry in training_file.itertuples():
            title= nltk.word_tokenize(entry[CONTENT_TITLE].encode('utf-8').decode('utf-8'))
            content = nltk.word_tokenize(entry[CONTENT_COLUMN].encode('utf-8').decode('utf-8'))
            tags= nltk.word_tokenize(entry[CONTENT_TAGS])

            stitle = set([stemmer.stem(w) for w in title])
            scontent = [stemmer.stem(w) for w in content]

            candidates= set (title+content)

            for candidate in candidates:
                tup= (tag_features(candidate,stitle,scontent), isTag(candidate,tags))
                training.append(tup)


        # Loop over each review; create an index i that goes from 0 to the length
        # of the movie review list
        train_set, test_set = training[500:], training[:500]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        print(nltk.classify.accuracy(classifier, test_set))
        classifier.show_most_informative_features(5)


# settings.init()
# dataPreprocessing.init()
trainingSet()


