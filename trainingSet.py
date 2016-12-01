import pandas as pd
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import wordnet


#see if we need to use tagging at all, and then if we do and NLTK is not enough check Stanford Tagger
from nltk.tag import pos_tag
from nltk.tag.stanford import StanfordNERTagger


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

def tag_features(word,title,content):
    features = {}
    features["is_in_title"] = word in title
    features["is_in_content"] = word in content
    #normally, if it occurs many times or has many synonyms it can be a tag?
    features["occurrences"]= title.count(word)+content.count(word)

    #shall we account for synonyms in occurrences?
    synonyms = []
    for ss in wordnet.synsets(word):
        synonyms.append(ss.lemma_names())
            #I think every word that is not a "word" is a tag
    features["is_in_wordnet"] = wordnet.synsets(word)

    #example: food of if there are many FOOD types listed?    features["is_superword"] synset1.lowest_common_hypernyms(synset2) ???
    #chech if it is the main noun in the sentence or verb
    #features["is_main']=
    return features


##to remove punctuation, use instead of nltk.word_tokenize
tokenizer= RegexpTokenizer(r'\w+')

total_tags=0
total_tags_in_title=0
tag_ngrams = []
ngrams={}

for entry in biology_pd.itertuples():
   #htmlcontent= BeautifulSoup(entry['content'])
    content= entry[CONTENT_COLUMN]
    htmlcontent= BeautifulSoup(content,"html.parser")
    urls=htmlcontent.findAll('a',href=True)

    #contains u from unicode. Check if there is any problem with this
    tcontent=removeStopWords(tokenizer.tokenize(htmlcontent.get_text().encode('utf-8')))
    ttitle= removeStopWords(tokenizer.tokenize(entry[CONTENT_TITLE]))
    ttags= tokenizer.tokenize(entry[CONTENT_TAGS])
    total_tags += len(ttags)


    #parsed_sentence_dict = pyparseface.parse_sentence(ttitle)
    #print("OrderedDict: %s\n" % parsed_sentence_dict)

    for tag in ttags:
        tag_features(tag, ttitle,tcontent)

    i=0
    for word in ttitle:
        i += 1
        if word in ttags:
            if(i-NGRAM>=0):
                t= tuple(ttitle[i-NGRAM:i])

                if not ngrams.has_key(t):
                    ngrams.update({t:1})
                else:
                    ngram_occurrences = ngrams[t]
                    ngrams.update({t:ngram_occurrences+1})




print(total_tags_in_title/total_tags*100)





