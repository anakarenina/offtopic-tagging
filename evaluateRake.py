from __future__ import absolute_import
from __future__ import print_function
from six.moves import range
__author__ = 'a_medelyan'
import RAKE2
import sys
from bs4 import BeautifulSoup
import itertools


def getSingleTags(tags):
    x= list(itertools.chain.from_iterable([tag.split('-') for tag in tags]))
    return x


def test():

    test_set = RAKE2.training_files

    # evaluating
    rake_object = RAKE2.Rake("SmartStoplist.txt", 5, 3, 2)

    j = 0

    for file in RAKE2.training_files:

        total_precision = 0
        total_recall = 0

        for entry in file.itertuples():

            tags = entry[RAKE2.CONTENT_TAGS].split()

            content = entry[RAKE2.CONTENT_COLUMN]
            htmlcontent = BeautifulSoup(content, "html.parser")

            for url in htmlcontent.findAll('a'):
                del htmlcontent['href']

            print('file ', RAKE2.topics[j])
            print(len(tags), 'manual keywords: ', tags)

            keywords = rake_object.run(text = entry[RAKE2.CONTENT_TITLE].encode('utf-8').decode('utf-8')+htmlcontent.get_text().encode('utf-8').decode('utf-8'))
            print('RAKE keywords:', keywords)

            num_manual_keywords = len(tags)

            correct = 0
            for i in range(0,len(keywords)):
                words_in_tags=getSingleTags(tags)
                if keywords[i] in set(words_in_tags):
                    correct += 1
            if len(keywords)>0:
                total_precision += correct/float(len(keywords))
                total_recall += correct/float(len(tags))
                print('correct:', correct, 'out of', num_manual_keywords)
            else:
                total_precision+=0
                total_recall+=0
                print('no tags predicted')

        j+=1

        # avg_precision = round(total_precision*100/float(file.shape[0]), 2)
        # avg_recall = round(total_recall*100/float(file.shape[0]), 2)
        avg_precision = round(total_precision/float(file.shape[0]), 2)
        avg_recall = round(total_recall/float(file.shape[0]), 2)

        avg_fmeasure = round(2*avg_precision*avg_recall/(avg_precision + avg_recall), 2)

        print("Precision", avg_precision, "Recall", avg_recall, "F-Measure", avg_fmeasure)

test()