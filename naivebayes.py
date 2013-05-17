import random
import math
import copy
import nltk
from nltk.tag.simplify import simplify_wsj_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import bigrams

#vraca feature u dokumentu
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['%s' % word] = (word in document_words)
    return features

def izbaci_stop_words(all_words):
    stopwords = nltk.corpus.stopwords.words('english')
    all_words = [w for w in all_words if w not in stopwords]
    return all_words
    
#razdvaja rijeci po apostrofu i uzima samo dio rijeci prije apostrofa
def stem(word):
    #for suffix in ['ed', 'ies', 'y', 'es', 's']:
    # if word.endswith(suffix):
     # word=word[:-len(suffix)]
    l=word.split("'")
    return l[0]

#provjerava je li vrsta rijeci u dopustenom skupu rijeci
def ok(word):
    if word == '.' or word == ',' or word == ':' or word == 'CNJ' or word == 'DET' or word == 'EX' or word == 'NUM' or word == 'P' or word == 'TO' or word == 'WH':
        return 0
    return 1

#za svaku rijec u all_words provjerava da li u all_words postoji njen sinonim
#ako postoji dodaje sinonim ne tu rijec
def sinonimi(all_words):
    lista = []
    counter = 0
    for word in all_words:
        if len(wn.synsets(word)) != 0:
            x = wn.synsets(word)[0].lemma_names
            if x.__contains__(word):
                index = x.index(word)
                x.pop(index)
            r = [i for i in x if i in all_words]
            if len(r):
                lista.append(r[0])
                counter = counter + 1
            else:
                lista.append(word)
        else:
            lista.append(word)
    print counter
    return lista

#vracamo rijeci sortirane po omjeru pojavljivanja u subjektivnim i objektivnim recenicama
#preferiramo rijeci s najmanjim omjerom
def omjer_pojavljivanja(documents):
    mapobj={}
    mapsubj={}
    for i in range(7000):
        if (documents[i][1]=='objective'):
            for j in range(len(documents[i][0])):
                if documents[i][0][j] not in mapobj.keys():
                    mapobj[documents[i][0][j]]=0
                mapobj[documents[i][0][j]]=mapobj[documents[i][0][j]]+1
        if (documents[i][1]=='subjective'):
            for j in range(len(documents[i][0])):
                if documents[i][0][j] not in mapsubj.keys():
                    mapsubj[documents[i][0][j]]=0
                mapsubj[documents[i][0][j]]=mapsubj[documents[i][0][j]]+1
    omjer={}
    feat=[]
    for key in mapsubj.keys():
        if key in mapobj.keys():
            s=mapsubj[key]
            o=mapobj[key]
            if (s<o): omjer[key]=s/o
            else: omjer[key]=o/s
    omjer2=sorted(omjer, key=omjer.get)
    for i in range(len(omjer2)):
        feat.append(omjer2[i])
    return feat

#za feature odabiremo rijeci s najvecom frekvencijom
def najveca_frekvencija(obj,subj):
    all=obj[:][:] + subj[:][:]
    all_words = []
    for i in range(7000):
        for j in range(len(all[i])):
            all_words.append(all[i][j])
    all_words = nltk.FreqDist(all_words)
    return all_words.keys()

#otklanjanje nepozeljnih rijeci
def otkloni_nepozeljne(all_words):
    tagged_sent = nltk.pos_tag(all_words[:10000])
    simplified = [(word, simplify_wsj_tag(tag)) for word, tag in tagged_sent]
    useful_words = [t[0] for t in simplified if t[0] != '"' and (ok(t[1]) or t[0] == '--')]
    return useful_words[:2000]

#ucitavanje datoteka
def ucitavanje(obj, subj):
    f=open("obj.txt")
    for i in range(5000):
        line=f.readline()
        vector=line.split()
        obj.append(vector)
    f.close()

    f=open("subj.txt")
    for i in range(5000):
        line=f.readline()
        vector=line.split()
        subj.append(vector)
    f.close()
    documents=[]
    for i in range(5000):
        documents.append([obj[i][:], 'objective'])
        documents.append([subj[i][:], 'subjective'])
        
    random.shuffle(documents)
    return documents
    

#POCETAK KODA
obj=[]
subj=[]
documents = ucitavanje(obj,subj)
all_words = najveca_frekvencija(obj,subj)
#print all_words[:20]
#all_words = izbaci_stop_words(all_words)
#print all_words[:20]
#all_words = [stem(word) for word in all_words]
print bigrams(documents[10][0])
word_features = otkloni_nepozeljne(all_words)
#word_features = all_words.keys()[:2000]

featuresets = [(document_features(d), c) for (d,c) in documents]
train_set = featuresets[:7000]
test_set = featuresets[7000:]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print nltk.classify.accuracy(classifier, test_set)
print nltk.classify.accuracy(classifier, train_set)
##for i in range(10):
##    train_set = featuresets[:i*1000] + featuresets[(i + 1)*1000:]
##    test_set = featuresets[i*1000:(i+1)*1000]
##    classifier = nltk.NaiveBayesClassifier.train(train_set)
##    #classifier.show_most_informative_features(20)
##    print nltk.classify.accuracy(classifier, test_set)
