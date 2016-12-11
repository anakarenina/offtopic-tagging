# settings.py
import pandas as pd

CONTENT_TITLE = 2
CONTENT_COLUMN = 3
CONTENT_TAGS = 4
NGRAM = 3

def init():

    ########### Convert csv input files into dataframes###########
    biology_pd = pd.read_csv('biology.csv')
    cooking_pd = pd.read_csv('cooking.csv')
    cryptology_pd = pd.read_csv('crypto.csv')
    diy_pd = pd.read_csv('diy.csv')
    robotics_pd = pd.read_csv('robotics.csv')
    travel_pd = pd.read_csv('travel.csv')
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