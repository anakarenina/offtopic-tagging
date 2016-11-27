import pandas as pd
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

CONTENT_TITLE=2
CONTENT_COLUMN=3
CONTENT_TAGS=4

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

##to remove punctuation, use instead of nltk.word_tokenize
tokenizer= RegexpTokenizer(r'\w+')


for entry in biology_pd.itertuples():
   #htmlcontent= BeautifulSoup(entry['content'])
    content= entry[CONTENT_COLUMN]
    htmlcontent= BeautifulSoup(content,"html.parser")
    urls=htmlcontent.findAll('a',href=True)

    #contains u from unicode. Check if there is any problem with this
    tcontent=removeStopWords(tokenizer.tokenize(htmlcontent.get_text()))
    ttitle= removeStopWords(tokenizer.tokenize(entry[CONTENT_TITLE]))
    ttags= tokenizer.tokenize(entry[CONTENT_TAGS])






