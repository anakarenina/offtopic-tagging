# settings.py
import pandas as pd

CONTENT_TITLE = 2
CONTENT_COLUMN = 3
CONTENT_TAGS = 4
NGRAM = 3
SAMPLE_SIZE = 500

def init():

    ########### Convert csv input files into dataframes###########
    biology_pd = pd.read_csv('biology.csv').sample(n=SAMPLE_SIZE)
    cooking_pd = pd.read_csv('cooking.csv').sample(n=SAMPLE_SIZE)
    cryptology_pd = pd.read_csv('crypto.csv').sample(n=SAMPLE_SIZE)
    diy_pd = pd.read_csv('diy.csv').sample(n=SAMPLE_SIZE)
    robotics_pd = pd.read_csv('robotics.csv').sample(n=SAMPLE_SIZE)
    travel_pd = pd.read_csv('travel.csv').sample(n=SAMPLE_SIZE)
    # test_pd = pd.read_csv('test.csv')


    global topics
    topics= ['biology', 'cooking', 'crypto', 'dyi', 'robotics', 'travel']

    global training_files
    training_files= []
    training_files.append(biology_pd)
    training_files.append(cooking_pd)
    training_files.append(cryptology_pd)
    training_files.append(diy_pd)
    training_files.append(robotics_pd)
    training_files.append(travel_pd)


init()