import random
import math
import copy
import nltk
from nltk.tag.simplify import simplify_wsj_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import bigrams
import svm
from svm import *
import svmutil
from svmutil import *

#izdvajamo rijeci koje se pojavljuju u dokumentu u svrhu odabira featura
def words_for_feature_selection(document):
    features = []
    bigram = []
    bigram.append([w for w in bigrams(document)])
    document_bigram=[]
    [document_bigram.extend(w) for w in bigram]
    document.extend(document_bigram)
    for word in word_features1:
        features.append(document.count(word))
    return features

#vraca feature u dokumentu
def document_features(document):
    bigram = []
    bigram.append([w for w in bigrams(document)])
    document_bigram=[]
    [document_bigram.extend(w) for w in bigram]
    document.extend(document_bigram)
    features = {}
    i=0
    for word in word_features:
        features[i] = document.count(word)
        i=i+1
    return features

def izbaci_stop_words(all_words):
    stopwords = nltk.corpus.stopwords.words('english')
    all_words = [w for w in all_words if w not in stopwords]
    return all_words
    
#razdvaja rijeci po apostrofu i uzima samo dio rijeci prije apostrofa
def stem(word):
    l=word.split("'")
    return l[0]

#provjerava je li vrsta rijeci u dopustenom skupu rijeci
def ok(word):
    if word == '.' or word == ',' or word == ':' or word == 'CNJ' or word == 'DET' or word == 'EX' or word == 'NUM' or word == 'P' or word == 'TO' or word == 'WH':
        return 0
    return 1

#provjerava da li je rijec interpunkcijski znak
def nije_interpunkcija(word):
    if word == '.' or word == ',' or word == ':' or word == '"' or word == ')' or word == '(' or word == '-':
        return 0
    return 1

#ako se neka rijec u test setu nije pojavila u nasim featurima,a njen sinonim jest,zamjenjujemo je sa sinonimom
def sinonimi(documents,word_features,dg,gg):
    for i in range(dg,gg):
        ex = documents[i]
        index = 0
        for word in ex[0]:
            if(word not in word_features):
                if(len(wn.synsets(word)) != 0):
                    x = wn.synsets(word)[0].lemma_names
                    r = [i for i in x if i in word_features]
                    if len(r):
                        ex[0][index] = r[0]
            index = index + 1
    return test_set

#vracamo rijeci sortirane po omjeru pojavljivanja u subjektivnim i objektivnim recenicama
#preferiramo rijeci s najmanjim omjerom
def omjer_pojavljivanja(documents,dg1,gg1,dg2,gg2):
    mapobj={}
    mapsubj={}
    for i in range(dg1,gg1):
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
    for i in range(dg2,gg2):
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
        elif(mapsubj[key] > 10):
            omjer[key] = 0
    for key in mapobj.keys():
        if key not in mapsubj.keys() and mapobj[key] > 10:
            omjer[key] = 0
    omjer2=sorted(omjer, key=omjer.get)
    for i in range(len(omjer2)):
        feat.append(omjer2[i])
    return feat

#za feature odabiremo rijeci s najvecom frekvencijom
def najveca_frekvencija(obj,subj,dg1,gg1,dg2,gg2):
    all=obj[:][:] + subj[:][:]
    all_words = []
    for i in range(dg1,gg1):
        for j in range(len(all[i])):
            all_words.append(all[i][j])
    for i in range(dg2,gg2):
        for j in range(len(all[i])):
            all_words.append(all[i][j])
    all_words = nltk.FreqDist(all_words)
    return all_words.keys()

#otklanjanje nepozeljnih rijeci, argument koliko znaci koliko rijeci zelimo da funkcija vrati
def otkloni_nepozeljne(all_words,koliko):
    tagged_sent = nltk.pos_tag(all_words[:10000])
    simplified = [(word, simplify_wsj_tag(tag)) for word, tag in tagged_sent]
    useful_words = [t[0] for t in simplified if t[0] != '"' and (ok(t[1]) or t[0] == '--')]
    return useful_words[:koliko]

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

#funkcija vraca bigrame koji se pojavljuju u recenicama
def bigrami(documents):
    bigram = []
    stopwords = nltk.corpus.stopwords.words('english')
    for i in range(7000):
        bigram.append([w for w in bigrams(documents[i][0])])
    result = []
    [result.extend(w) for w in bigram]
    result = [w for w in result if w[0] not in stopwords and w[1] not in stopwords and w[0] and nije_interpunkcija(w[0]) and nije_interpunkcija(w[1])]
    result = nltk.FreqDist(result)
    return result.keys()

#parametri:
    #koliko - koliko najinformativnijih featura zelimo izdvojiti
    #funkcija na osnovu chi2 testa odabire najinformativnije feature
def feature_selection(documents,koliko):
    selector = SelectKBest(chi2, koliko)
    y = [c for (d,c) in documents][:7000]
    X = []
    for (d,c) in documents[:7000]:
        X.append(words_for_feature_selection(d))
    selector.fit(X,y)
    m = selector.get_support()
    return [i[0] for i in izip_longest(word_features1, m[:len(X)], fillvalue=True) if i[1]]

#POCINJE KOD
obj=[]
subj=[]
documents=ucitavanje(obj,subj)
all_words = najveca_frekvencija(obj,subj,0,7000,0,0)
all_bigrams = bigrami(documents)
#all_words = [stem(word) for word in all_words]
word_features = otkloni_nepozeljne(all_words,2000)
#word_features = all_words.keys()[:2000]
featuresets = [(document_features(d,1)) for (d,c) in documents]
classes=[c for (d,c) in documents]
train_set = featuresets[:7000]
train_classes=classes[:7000]
test_classes = classes[7000:]
test_set = featuresets[7000:]
px = svm_problem(train_classes,train_set)
param = svm_parameter()
param.kernel_type = LINEAR
param.C = 8
param.gama = 0.015625
m = svm_train(px, param)
svm_predict(test_classes[:],test_set[:],m)

