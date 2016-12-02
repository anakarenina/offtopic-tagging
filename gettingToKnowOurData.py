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

import operator
import numpy


CONTENT_TITLE=2
CONTENT_COLUMN=3
CONTENT_TAGS=4
NGRAM=3


########### Convert csv input files into dataframes###########
biology_pd = pd.read_csv('biology.csv')
cooking_pd = pd.read_csv('cooking.csv')
cryptology_pd = pd.read_csv('crypto.csv')
diy_pd = pd.read_csv('diy.csv')
robotics_pd = pd.read_csv('robotics.csv')
travel_pd = pd.read_csv('travel.csv')
#test_pd = pd.read_csv('test.csv')


def removeStopWords(text):
    return [word for word in text if word not in stopwords.words('english')]

training_files= []
training_files.append(biology_pd)
training_files.append(cooking_pd)
training_files.append(cryptology_pd)
training_files.append(diy_pd)
training_files.append(robotics_pd)
training_files.append(travel_pd)

##to remove punctuation, use instead of nltk.word_tokenize
tokenizer= RegexpTokenizer(r'\w+')

contents=[]

tags = []
titles=[]

for training_file in training_files:
    ctags = []
    wtags=[]
    titles = []
    ctagintitle=0
    ctagnotintitle=0

    for entry in training_file.itertuples():
        #htmlcontent= BeautifulSoup(entry['content'])
        #content= entry[CONTENT_COLUMN]
        #htmlcontent= BeautifulSoup(content,"html.parser")
        #urls=htmlcontent.findAll('a',href=True)

        #contains u from unicode. Check if there is any problem with this
        #tcontent=removeStopWords(tokenizer.tokenize(htmlcontent.get_text().encode('utf-8')))
        ttitle= removeStopWords(tokenizer.tokenize(entry[CONTENT_TITLE]))

        ## if we use nltk.word_tokenize instead we keep the compund words because it does not remove punctuation characters
        completetags= nltk.word_tokenize(entry[CONTENT_TAGS])
        wordintags= tokenizer.tokenize(entry[CONTENT_TAGS])

        #parsed_sentence_dict = pyparseface.parse_sentence(ttitle)
        #print("OrderedDict: %s\n" % parsed_sentence_dict)
        ctags=ctags+ completetags
        wtags=wtags+ wordintags

        # one way to see if tag is in title. Go through each word in tag and check if it is included in title. Check option 2
        # titles.append(ttitle)
        # #contents.append(contents)
        # for word in wordintags:
        #     if (word in ttitle):
        #         ctagintitle+=1
        #     else:
        #         ctagnotintitle+=1

    # print(ctagintitle/ctagnotintitle)

    #whole_data = contents.append(titles)


    ##Getting the distribution of the first
    fdist1 = FreqDist(tags)

    print(fdist1.N())
    # fdist1.plot(50, cumulative=True)
    # print(fdist1.most_common(50))




