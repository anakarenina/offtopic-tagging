import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.porter import *

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import operator
import numpy

import settings

CONTENT_TITLE=settings.CONTENT_TITLE
CONTENT_COLUMN=settings.CONTENT_COLUMN
CONTENT_TAGS= settings.CONTENT_TAGS
PLOT = False

N_MOST_FREQUENT=100
LABEL_SIZE=3.5

global topTags
topTags = dict()

##to remove punctuation, we can use instead of nltk.word_tokenize
tokenizer= RegexpTokenizer(r'\w+')

def getTopNTagsFreqDist(tags, topic, plot):
    fdist = FreqDist(tags)
    freq = fdist.most_common(N_MOST_FREQUENT)
    topTags[topic]= [seq[0] for seq in freq]

    if plot:
        plt.gcf().subplots_adjust(bottom=0.4)
        plt.title('Top ' + str(N_MOST_FREQUENT) + " most frequent tags in "+topic)
        fdist.plot(N_MOST_FREQUENT, cumulative=True)
        plt.rc('xtick', labelsize=LABEL_SIZE)
        print(fdist.most_common(N_MOST_FREQUENT))


def getCumulativePercentage(tags, topic, plot):
    fdist1 = FreqDist(tags)
    freq = fdist1.most_common(N_MOST_FREQUENT)
    freqwords = [seq[0] for seq in freq]

    frequencies = [seq[1] for seq in freq]
    total = fdist1.N()
    x = list(range(N_MOST_FREQUENT))
    percentages = [freq / float(total) for freq in frequencies]

    cs = np.cumsum(percentages)

    if plot:
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

#gets a vector of values and the title to plot as a Bar plot
def plotBarGraph(tagInTitledict, title):
    X = np.arange(len(tagInTitledict))
    bar= plt.bar(X, tagInTitledict.values(), align='center', width=0.5)
    bar[0].set_color('#E26D6D')
    bar[1].set_color('#E26DBF')
    bar[2].set_color('#6DA0E2')
    bar[3].set_color('#6DE2B7')
    bar[4].set_color('#ACE26D')
    bar[5].set_color('#93CFDE')
    plt.xticks(X, tagInTitledict.keys())
    ymax = max(tagInTitledict.values()) + 0.3
    plt.ylim(0, ymax)
    plt.title(title)
    plt.show()

stemmer= PorterStemmer()

i=0
all_tags=[]

perTaginTitle = dict()
perTaginContent = dict()
perFreqTaginTitle=dict()
perFreqTaginContent=dict()
perFreqNonTagWordsinContent= dict()

for training_file in settings.training_files:
    #complete-tags
    ctags = []
    #words in tags
    wtags=[]

    titles = []
    sumpertagsintitle=0
    sumpertagsincontent=0
    sumperfreqtagsintitle=0
    sumperfreqtagsincontent=0
    sumperfreqnontagwordsincontent=0
    nsamples=0
    nsampleswithtagintext = 0
    nsampleswithtagincontext=0


    for entry in training_file.itertuples():

        stemscontent=[]
        nontagwordsincontent=[]
        stemstitle=[]
        ctagsintitle = 0
        ctagsnotintitle = 0
        ctagsincontent = 0
        cnontagwordsincontent = 0
        nsamples+=1

        ftagsTitle = 0
        ftagsContent = 0
        fnontagwordsincontent=0

        content = nltk.word_tokenize(entry[CONTENT_COLUMN].encode('utf-8').decode('utf-8'))
        title= nltk.word_tokenize(entry[CONTENT_TITLE].encode('utf-8').decode('utf-8'))

        completetags= nltk.word_tokenize(entry[CONTENT_TAGS])
        wordintags= tokenizer.tokenize(entry[CONTENT_TAGS])

        ctags=ctags+ completetags
        wtags=wtags+ wordintags

        stemwordintags= [stemmer.stem(w) for w in wordintags]

        tagFreqTitle = dict.fromkeys(stemwordintags,0)
        tagFreqContent = dict.fromkeys(stemwordintags,0)


        for word in title:
            s = stemmer.stem(word)
            if s in stemwordintags:
                ftagsTitle += 1  # we want to count frequency
                if (s not in stemstitle):  # we just care about presence, so if repeated it should not count
                    stemstitle.append(stemmer.stem(word))
                    ctagsintitle += 1
            else:
                ctagsnotintitle += 1 #this is wrong


        for word in content:
            s = stemmer.stem(word)
            if s in stemwordintags:
                ftagsContent += 1  # we want to count frequency
                if (s not in stemscontent):  # we just care about presence, so if repeated it should not count
                    stemscontent.append(stemmer.stem(word))
                    ctagsincontent += 1
            else:
                fnontagwordsincontent += 1
                if (s not in nontagwordsincontent):  # we just care about presence, so if repeated it should not count
                    nontagwordsincontent.append(stemmer.stem(word))
                    cnontagwordsincontent += 1


        percentagetagsintitle = float(ctagsintitle) / len(completetags)
        sumpertagsintitle += percentagetagsintitle

        percentagetagsincontent = float(ctagsincontent) / len(completetags)
        sumpertagsincontent += percentagetagsincontent


        if ctagsintitle!=0:
            avgfreqtagsintitle = float(ftagsTitle) / ctagsintitle
            sumperfreqtagsintitle+= avgfreqtagsintitle
            nsampleswithtagintext += 1

        if ctagsincontent!=0:
            avgfreqtagsincontent = float(ftagsContent) / ctagsincontent
            sumperfreqtagsincontent+= avgfreqtagsincontent
            nsampleswithtagincontext+=1

        avgfreqnontagwordsincontent= float(fnontagwordsincontent) / cnontagwordsincontent
        sumperfreqnontagwordsincontent += avgfreqnontagwordsincontent

    #Gathering information for plotting and analyzing data
    topNctags= getTopNTags(ctags)
    topNwtags= getTopNTags(wtags)
    notincommon = set([obj for obj in topNctags if obj not in wtags])
    all_tags.append(topNctags)

    #print(len(notincommon)/float(N_MOST_FREQUENT)*100)
    getTopNTagsFreqDist(ctags,settings.topics[i],PLOT)
    getCumulativePercentage(ctags,settings.topics[i],PLOT)

    perTaginTitle[settings.topics[i]]= sumpertagsintitle/nsamples
    perTaginContent[settings.topics[i]]=sumpertagsincontent/nsamples

    perFreqTaginTitle[settings.topics[i]] = sumperfreqtagsintitle / nsampleswithtagintext
    perFreqTaginContent[settings.topics[i]] = sumperfreqtagsincontent / nsampleswithtagincontext
    perFreqNonTagWordsinContent[settings.topics[i]] = sumperfreqnontagwordsincontent / nsamples
    i+=1

    #print("Intersection: " + set.intersection(*map(set,all_tags)) ) #NO INTERSECTION!


plotBarGraph(perTaginTitle,'Percentage of tags that are in titles')
plotBarGraph(perTaginContent,'Percentage of tags that are in content')

plotBarGraph(perFreqTaginTitle,'Frequency of tags in titles')
plotBarGraph(perFreqTaginContent,'Frequency of tags in content')
plotBarGraph(perFreqNonTagWordsinContent,'Frequency of non tag words in content')







