import pandas as pd
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

##do we need beautiful soup?


from nltk.corpus import stopwords

# Convert csv input files into dataframes
biology_pd = pd.read_csv('biology.csv')
cooking_pd = pd.read_csv('cooking.csv')
cryptology_pd = pd.read_csv('crypto.csv')
diy_pd = pd.read_csv('diy.csv')
robotics_pd = pd.read_csv('robotics.csv')
travel_pd = pd.read_csv('travel.csv')
#test_pd = pd.read_csv('test.csv')

def removeStopWords(text):
    return [word for word in text if word not in stopwords.words('english')]

content_text = BeautifulSoup(biology_pd['content'][0],"html.parser")
content_text.find('p')

title= nltk.word_tokenize(str(biology_pd['title'][1]))
body= nltk.word_tokenize(str(biology_pd['content'][1]))
print(body)
body=removeStopWords(body)







print(body)

