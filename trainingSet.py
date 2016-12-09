import pandas as pd
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import wordnet
from nltk.probability import FreqDist
from matplotlib import pylab
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.porter import *
from collections import defaultdict
from nltk.corpus import words

#see if we need to use tagging at all, and then if we do and NLTK is not enough check Stanford Tagger
from nltk.tag import pos_tag
from nltk.tag.stanford import StanfordNERTagger


import operator
import numpy


CONTENT_TITLE=2
CONTENT_COLUMN=3
CONTENT_TAGS=4
NGRAM=3

stemmer= PorterStemmer()


tokenizer= RegexpTokenizer(r'\w+')


N_MOST_FREQUENT=200
LABEL_SIZE=3.5
SAMPLE_SIZE= 500

########### Convert csv input files into dataframes###########
biology_pd = pd.read_csv('biology.csv').sample(n=1)
cooking_pd = pd.read_csv('cooking.csv').sample(n=SAMPLE_SIZE)
cryptology_pd = pd.read_csv('crypto.csv').sample(n=SAMPLE_SIZE)
diy_pd = pd.read_csv('diy.csv').sample(n=SAMPLE_SIZE)
robotics_pd = pd.read_csv('robotics.csv').sample(n=SAMPLE_SIZE)
travel_pd = pd.read_csv('travel.csv').sample(n=SAMPLE_SIZE)

topics= ['biology', 'cooking', 'crypto', 'dyi', 'robotics', 'travel']

training_files= []
training_files.append(biology_pd)
training_files.append(cooking_pd)
training_files.append(cryptology_pd)
training_files.append(diy_pd)
training_files.append(robotics_pd)
training_files.append(travel_pd)


def removeStopWords(text):
    return [word for word in text if word not in stopwords.words('english')]

def tag_features(word,title,content):
    features = {}
    features["is_in_title"] = word in title
    features["is_in_content"] = word in content
    #normally, if it occurs many times or has many synonyms it can be a tag?
    features["occurrences"]= title.count(word)+content.count(word)

    #another feature is if word appear in most frequent tags or if it appears in tags at all, we can go for SE
    #another feature would be to check if words appear in the tag description
    #we can also do something with ontology to check this
    #shall we account for synonyms in occurrences?
    # we can also do something about relations among tags and how often a synonym goes with that so we should add it
    synonyms = []
    for ss in wordnet.synsets(word):
        synonyms.append(ss.lemma_names())
            #I think every word that is not a "word" is a tag
    features["is_in_wordnet"] = wordnet.synsets(word)

    #example: food of if there are many FOOD types listed?    features["is_superword"] synset1.lowest_common_hypernyms(synset2) ???
    #chech if it is the main noun in the sentence or verb
    #features["is_main']=
    return features





def isTag(word,tags):
    return stemmer.stem(word) in [stemmer.stem(tag) for tag in tags]

def trainingSet(df):
    labeled_words= preprocessWords(df)
    featuresets = [(tag_features(word), is_tag) for (word, is_tag, title, content) in labeled_words]






##to remove punctuation, use instead of nltk.word_tokenize
tokenizer= RegexpTokenizer(r'\w+')


for df in training_files:
   preprocessWords(df)





