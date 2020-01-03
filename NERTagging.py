from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

st = StanfordNERTagger('/home/muditdhawan/Downloads/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
					   '/home/muditdhawan/Downloads/stanford-ner-2018-10-16/stanford-ner.jar',
					   encoding='utf-8')

import pandas as pd
import string
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk import word_tokenize
from collections import Counter
import glob

text_list=[]
named_entity=[]
final_counter = Counter({})
i=0
for file in glob.glob("/home/muditdhawan/Downloads/captions/*.txt"):

    # print(file)
    # print(i)  

    i= i+1
    print(i)
    papers = pd.read_csv(file, delimiter="\t",names=["data"], skipinitialspace=True, header=None)

    # dropping the line "caption of image"
    papers=papers[~(papers.isin(papers.iloc[::4]))].dropna()
    # print(papers.shape)

    # make a list of the words 
    papers["data"]= papers["data"].str.split(" ")

    # remove the numbering
    papers=papers["data"].map(lambda x: x[1:])

    # removing punctuations
    papers=papers.map(lambda x: [''.join(c for c in s if c not in string.punctuation) for s in x])

    papers=papers.map(lambda x: ' '.join(x).split())

    # removing the probability term in each caption
    papers=papers.map(lambda x: " ".join(x[:-1]))

    # removing StopWords
    papers=papers.map(lambda x: [s for s in x.split() if s not in gensim.parsing.preprocessing.STOPWORDS ])

    # converting list into string
    papers = papers.map(lambda x: " ".join(x))

    # tokeninizing
    # word_counts = Counter(word_tokenize('\n'.join(papers)))

    # print(word_counts)
    # print(' '.join(papers))
    classified_text = st.tag((' '.join(papers)).split(" "))
# using the classified data to analyse which words don't come under the non-named entities 
    for j in range(len(classified_text)):
        if(classified_text[j][1]!='O'):
            named_entity.append(classified_text[j][0])
            print(classified_text[j])
    text_list.append((' '.join(papers)).split(" "))
    # print(text_list)
    # final_counter= final_counter+word_counts

# will upload the file as and when the code completes running and the file is written
with open('NER.txt', 'w') as f:
    for item in named_entity:
        f.write("%s\n" % item)
