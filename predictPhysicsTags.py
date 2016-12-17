import pandas as pd
from bs4 import BeautifulSoup
import RAKE2

IDS=1
CONTENT_TITLE = 2
CONTENT_COLUMN = 3

def init():

    ########### Convert csv input files into dataframes###########
    physics_pd = pd.read_csv('test.csv')
    rake_object = RAKE2.Rake("SmartStoplist.txt", 5, 3, 2)
    predicted_tags = pd.DataFrame()

    number_lines = physics_pd.shape[0]

    i=1
    chunksize = 100

    for entry in physics_pd.itertuples():

        id=entry[IDS]
        content = entry[CONTENT_COLUMN]
        htmlcontent = BeautifulSoup(content, "html.parser")

        for url in htmlcontent.findAll('a'):
            del htmlcontent['href']


        keywords = set(rake_object.run(text=entry[CONTENT_TITLE].encode('utf-8').decode('utf-8') + htmlcontent.get_text().encode('utf-8').decode('utf-8')))
        predicted_tags = predicted_tags.append({'id': str(id), 'tags': ' '.join(keywords)}, ignore_index=True)

        if (i % chunksize == 0):
            predicted_tags.to_csv('submission.csv',mode='a',chunksize=chunksize,header=False)
            predicted_tags= pd.DataFrame()

        i+=1

    predicted_tags.to_csv('submission.csv', mode='a', chunksize=chunksize, header=False)


    predicted_tags = predicted_tags[["id", "tags"]]


init()
