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

CONTENT_TITLE=settings.CONTENT_TITLE
CONTENT_COLUMN=settings.CONTENT_COLUMN
CONTENT_TAGS= settings.CONTENT_TAGS

tokenizer= RegexpTokenizer(r'\w+')

def removeStopWords(text):
    return [word for word in text if word not in stopwords.words('english')]


def cleanText(text):
    ccontent = removeStopWords(text)
    ccontent = filter(lambda word: word not in string.punctuation+'``', ccontent)
    ccontent= filter(lambda word: None == re.match(r'\'', word), ccontent)
    pos_content = nltk.pos_tag(ccontent)

    ##further removing little lexical content words
    posTaggedContent = filter(lambda (word, tag): tag not in (
        'CC', 'DT', 'EX', 'LS', 'MD', 'PDT', 'PRP', 'PRP', 'RB', 'RBS', 'RP', 'UH', 'WDT', 'WP', 'WRB', 'TO', 'IN','CD'),
                              pos_content)
    return [x for (x, y) in posTaggedContent]


#preprocess file, removing stop words and more irrelevant information for the training set
def preprocessFile(tf,topic):

    df = pd.DataFrame()

    for entry in tf.itertuples():
        i=0

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

        #COMPLETE TAGS
        tags = nltk.word_tokenize(entry[CONTENT_TAGS])

        #WORDS WITHIN TAGS
        wordsintags = entry[CONTENT_TAGS]
        cwordsintags= set(cleanText(tokenizer.tokenize(wordsintags)))

        df = df.append({'content':  ' '.join(ccontent), 'title': ' '.join(ctitle), 'complete_tags': ' '.join(tags),'words_in_tags': ' '.join(cwordsintags)}, ignore_index=True)

        i+=1

    df = df[["title","content","complete_tags","words_in_tags"]]
    df.to_csv('preprocessed'+topic+'.csv')


def preprocessFiles():
    i=0
    for trainingfile in settings.training_files:
        preprocessFile(trainingfile, settings.topics[i])
        i+=1

settings.init()
preprocessFiles()