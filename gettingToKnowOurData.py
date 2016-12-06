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

'Test'

import operator
import numpy


CONTENT_TITLE=2
CONTENT_COLUMN=3
CONTENT_TAGS=4
NGRAM=3

N_MOST_FREQUENT=200
LABEL_SIZE=3.5

########### Convert csv input files into dataframes###########
biology_pd = pd.read_csv('biology.csv').sample(n=500)
cooking_pd = pd.read_csv('cooking.csv').sample(n=500)
cryptology_pd = pd.read_csv('crypto.csv').sample(n=500)
diy_pd = pd.read_csv('diy.csv').sample(n=500)
robotics_pd = pd.read_csv('robotics.csv').sample(n=500)
travel_pd = pd.read_csv('travel.csv').sample(n=500)
#test_pd = pd.read_csv('test.csv')

topics= ['biology', 'cooking', 'crypto', 'dyi', 'robotics', 'travel']

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
#tokenizer = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
contents=[]

tags = []
titles=[]

def getTopNTagsFreqDist(tags, topic):
    fdist = FreqDist(tags)
    freq = fdist.most_common(N_MOST_FREQUENT)
    plt.gcf().subplots_adjust(bottom=0.4)
    plt.title('Top ' + str(N_MOST_FREQUENT) + " most frequent tags in "+topic)
    fdist.plot(N_MOST_FREQUENT, cumulative=True)
    plt.rc('xtick', labelsize=LABEL_SIZE)
    print(fdist.most_common(N_MOST_FREQUENT))


def getCumulativePercentage(tags, topic):
    fdist1 = FreqDist(tags)
    freq = fdist1.most_common(N_MOST_FREQUENT)
    freqwords = [seq[0] for seq in freq]

    frequencies = [seq[1] for seq in freq]
    total = fdist1.N()
    x = list(range(N_MOST_FREQUENT))
    percentages = [freq / float(total) for freq in frequencies]

    cs = np.cumsum(percentages)
    plt.rc('xtick', labelsize=LABEL_SIZE)
    plt.xticks(x, freqwords)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    plt.gcf().subplots_adjust(bottom=0.4)
    plt.plot(x, percentages)
    plt.title('Accumulative percentage of tags covered by the most ' + str(N_MOST_FREQUENT) + " frequent tags in "+topic)
    plt.plot(x, cs, 'r--')
    plt.show()



def getTopNTags(tags):
    fdist1 = FreqDist(tags)
    freq = fdist1.most_common(N_MOST_FREQUENT)
    return [seq[0] for seq in freq]

#should call it draw bars or something like that
def getPercentageOfTags(tagInTitledict, title):
    X = np.arange(len(tagInTitledict))
    plt.bar(X, tagInTitledict.values(), align='center', width=0.5)
    plt.xticks(X, tagInTitledict.keys())
    ymax = max(tagInTitledict.values()) + 0.3
    plt.ylim(0, ymax)
    plt.title('Percentage of tags that are in '+title)
    plt.show()

stemmer= PorterStemmer()

i=0
all_tags=[]

perTaginTitle = dict()
perTaginContent = dict()
perFreqTaginTitle=dict()
perFreqTaginContent=dict()

for training_file in training_files:
    #complete-tags
    ctags = []
    #words in tags
    wtags=[]

    titles = []
    sumpertagsintitle=0
    sumpertagsincontent=0
    sumperfreqtagsintitle=0
    sumperfreqtagsincontent=0
    nsamples=0



    for entry in training_file.itertuples():

        stemscontent=[]
        stemstitle=[]
        ctagsintitle = 0
        ctagsnotintitle = 0
        ctagsincontent = 0
        ctagsnotincontent = 0
        nsamples+=1

        ftagsTitle = 0
        ftagsContent = 0

        content = entry[CONTENT_COLUMN]
        htmlcontent = BeautifulSoup(content, "html.parser")
        #urls=htmlcontent.findAll('a',href=True)

        #contains u from unicode. Check if there is any problem with this
        #tcontent=removeStopWords(tokenizer.tokenize(htmlcontent.get_text().encode('utf-8').decode('utf-8')))
        tcontent= removeStopWords(nltk.word_tokenize(htmlcontent.get_text().lower().encode('utf-8').decode('utf-8')))
        tcontent= filter(lambda word: word not in ",-?.", tcontent)
        pos_content=nltk.pos_tag(tcontent)

        ##further removing little lexical content words
        posTaggedContent= filter(lambda (word, tag): tag not in ('CC','DT', 'EX', 'LS' , 'MD', 'PDT', 'PRP', 'PRP', 'RB', 'RBS', 'RP', 'UH', 'WDT', 'WP', 'WRB','TO','IN'), pos_content)
        #check if we need to remove CD, cardinal
        tcontent= [x for (x,y) in posTaggedContent]

        title= entry[CONTENT_TITLE].lower().decode('utf-8')
        ttitle= removeStopWords(nltk.word_tokenize(title))
        ttitle= filter(lambda word: word not in ",-?.", ttitle)
        pos_title= nltk.pos_tag(ttitle)

        posTaggedTitle = filter(lambda (word, tag): tag not in ('CC', 'DT', 'EX', 'LS', 'MD', 'PDT', 'PRP', 'PRP', 'RB', 'RBS', 'RP', 'UH', 'WDT', 'WP', 'WRB', 'TO', 'IN'),
                                  pos_title)
        # check if we need to remove CD, cardinal
        ttitle = [x for (x, y) in posTaggedTitle]


        ## if we use nltk.word_tokenize instead we keep the compund words because it does not remove punctuation characters
        completetags= nltk.word_tokenize(entry[CONTENT_TAGS])
        wordintags= tokenizer.tokenize(entry[CONTENT_TAGS])

        #parsed_sentence_dict = pyparseface.parse_sentence(ttitle)
        #print("OrderedDict: %s\n" % parsed_sentence_dict)
        ctags=ctags+ completetags
        wtags=wtags+ wordintags

        if (nsamples == 3126):
            print('hola')

        stemwordintags= [stemmer.stem(w) for w in wordintags]

        # tagFreqTitle = dict.fromkeys(stemwordintags,0)
        # tagFreqContent = dict.fromkeys(stemwordintags,0)

        #maybe should do the same with complete words?

        for word in ttitle:
            s = stemmer.stem(word)
            if s in stemwordintags:
                ftagsTitle+=1  # we want to count frequency
                if (s not in stemstitle): #we just care about presence, so if repeated it should not count
                    stemstitle.append(stemmer.stem(word))
                    ctagsintitle += 1
            else:
                ctagsnotintitle+=1

        # for word in tcontent:
        #     s= stemmer.stem(word)
        #     if s in stemwordintags:
        #         ftagsContent+=1 # we want to count frequency
        #         if (s not in stemscontent): #we just care about presence, so if repeated it should not count
        #             stemscontent.append(stemmer.stem(word))
        #             ctagsincontent+=1
        #     else:
        #         ctagsnotincontent+=1

        # percentagetagsintitle = float(ctagsintitle) / len(completetags)
        # sumpertagsintitle += percentagetagsintitle
        #
        # percentagetagsincontent = float(ctagsincontent) / len(completetags)
        # sumpertagsincontent += percentagetagsincontent
        #

        if ctagsintitle!=0:
            avgfreqtagsintitle = float(ftagsTitle) / ctagsintitle
            sumperfreqtagsintitle+= avgfreqtagsintitle

        # if ctagsincontent!=0:
        #     avgfreqtagsincontent = float(ftagsContent) / ctagsincontent
        #     sumperfreqtagsincontent+= avgfreqtagsincontent








    #whole_data = contents.append(titles)


    #percentage in title

    ##Getting some plots
    # topNctags= getTopNTags(ctags)
    # topNwtags= getTopNTags(wtags)
    # notincommon = set([obj for obj in topNctags if obj not in wtags])
    # all_tags.append(topNctags)
    #
    # #print(len(notincommon)/float(N_MOST_FREQUENT)*100)
    # getTopNTagsFreqDist(ctags,topics[i])
    # getCumulativePercentage(ctags,topics[i])
    # perTaginTitle[topics[i]]=sumpertagsintitle/nsamples
    # perTaginContent[topics[i]]=sumpertagsincontent/nsamples

    perFreqTaginTitle[topics[i]] = sumperfreqtagsintitle / nsamples
    # perFreqTaginContent[topics[i]] = sumperfreqtagsincontent / nsamples
    i+=1

    #print("Intersection: " + set.intersection(*map(set,all_tags)) )


# getPercentageOfTags(perTaginTitle,'titles')
# getPercentageOfTags(perTaginContent,'content')

getPercentageOfTags(perFreqTaginTitle,'titles')
# getPercentageOfTags(perFreqTaginContent,'content')




