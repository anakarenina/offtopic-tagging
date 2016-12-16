import pandas as pd
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
from nltk.tag import pos_tag
import csv
import re
import string
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import settings
import gettingToKnowOurData



CONTENT_TITLE=settings.CONTENT_TITLE
CONTENT_COLUMN=settings.CONTENT_COLUMN
CONTENT_TAGS= settings.CONTENT_TAGS

tokenizer= RegexpTokenizer(r'\w+')
stops = set(stopwords.words("english"))

def cleanText (text): 
	ccontent = filter(lambda word: word not in string.punctuation+'``', text)
def removeStopWords(text):
    return [word for word in text if word not in stops]

def cleanText(text):
    #ccontent = removeStopWords(text) in this case we want to leave in stop words 
    	# we want to take them out, however when we are looking at stop words 
    ccontent = filter(lambda word: word not in string.punctuation+'``', text)
    ccontent= filter(lambda word: None == re.match(r'\'', word), ccontent)
    pos_content = nltk.pos_tag(ccontent) #part of speech labeling 

    ##further removing little lexical content words
    ## CD: 
    ## LS:
    ## PDT: pre-determiner 'all, both, half, quite, this, such, sure' 
    ## PRP: pronoun, personal
    ## PRP$: posessive pronoun 
    ## TO: to
    ## UH: interjection
    posTaggedContent = filter(lambda (word, tag): tag not in ('CD', 'LS', 'PDT', 'PRP', 'PRP$', 'TO', 'UH'),
                              pos_content)
    return [x for (x, y) in posTaggedContent]


#preprocess file, removing stop words and more irrelevant information for the training set
def preprocessFile(tf,topic):

    df = pd.DataFrame()

    for entry in tf.itertuples():
        i=0
        incorporate=False

        # COMPLETE TAGS
        tags = nltk.word_tokenize(entry[CONTENT_TAGS])

        # WORDS WITHIN TAGS
        wordsintags = entry[CONTENT_TAGS]
        cwordsintags = set(cleanText(tokenizer.tokenize(wordsintags)))

        #get main words
        topTags= gettingToKnowOurData.topTags[topic]
        #should lemmatize or stem
        for word in cwordsintags:

            if word in topTags:
                incorporate=True

        #only considering rows which include EXACT TOP KEYWORDS
        # we could try lemmatizing here
        # or we could include non-top keywords
        if (incorporate):
            #CONTENT
            content = entry[CONTENT_COLUMN]
            htmlcontent = BeautifulSoup(content, "html.parser")

            # urls=htmlcontent.findAll('a',href=True)
            #remove urls

            for url in htmlcontent.findAll('a'):
                del htmlcontent['href']

            ccontent= cleanText(nltk.word_tokenize(htmlcontent.get_text().lower().encode('utf-8').decode('utf-8')))
            #TITLE
            title = entry[settings.CONTENT_TITLE].lower().encode('utf-8').decode('utf-8')
            ctitle = cleanText(nltk.word_tokenize(title))

            df = df.append({'content':  ' '.join(ccontent), 'title': ' '.join(ctitle), 'complete_tags': ' '.join(tags),'words_in_tags': ' '.join(cwordsintags)}, ignore_index=True)

        i+=1

    df = df[["title","content","complete_tags","words_in_tags"]]
    df.to_csv('n-gram'+topic+'.csv')
    
    #data pre-process file 
def preprocessFiles():
    i=0
    for trainingfile in settings.training_files:
        preprocessFile(trainingfile, settings.topics[i])
        i+=1

settings.init()
preprocessFiles()