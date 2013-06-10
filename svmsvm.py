# -*- coding: cp1250 -*-
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
from grid import *
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
import itertools

#-----PARAMETRI KOJE JE MOGUCE PODESITI-----#

#zelite li koristiti i bigrame
KORISTENJE_BIGRAMA = 0
#odabirom najveca frekvencija = 0, koristi se omjer_pojavljivanja
NAJVECA_FREKVENCIJA = 1
#koristenje otkloni nepozeljne
OTKLONI_NEPOZELJNE = 0
#koristenje stem-a
STEM = 0
#koristenje izbaci stop words
STOP = 0
#koristenje k-struke cross validacije
K = 4
#koristenje feature selectiona
FEATURE_SELECTION = 0
#koristenje sinonima kod pripreme test seta
SINONIMI = 0
#broj faetura
BROJ_FEATURA = 2000

#--------------------------------------------#


#izdvajamo rijeci koje se pojavljuju u dokumentu u svrhu odabira featura
def words_for_feature_selection(document):
    features = []
    if(KORISTENJE_BIGRAMA == 1):
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
    if(KORISTENJE_BIGRAMA == 1):
        bigram = []
        bigram.append([w for w in bigrams(document)])
        document_bigram=[]
        [document_bigram.extend(w) for w in bigram]
        document.extend(document_bigram)
    features = {}
    i=0
    for word in word_features:
        if word in document:
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
            if(word not in word_features and type(word) != tuple):
                if(len(wn.synsets(word)) != 0):
                    x = wn.synsets(word)[0].lemma_names
                    r = [i for i in x if i in word_features]
                    if len(r):
                        ex[0][index] = r[0]
            index = index + 1
    return documents

#vracamo rijeci sortirane po omjeru pojavljivanja u subjektivnim i objektivnim recenicama
#preferiramo rijeci s najmanjim omjerom
#ako se neka rijec javlja samo u jednoj klasi,definiramo da je njen omjer 0
#i smatramo ju korisnom
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
        documents.append([obj[i][:], -1])
        documents.append([subj[i][:], 1])
        
    random.shuffle(documents)
    return documents

#funkcija vraca bigrame koji se pojavljuju u recenicama
def bigrami(documents,dg1,gg1,dg2,gg2):
    bigram = []
    stopwords = nltk.corpus.stopwords.words('english')
    for i in range(dg1,gg1):
        bigram.append([w for w in bigrams(documents[i][0])])
    for i in range(dg2,gg2):
        bigram.append([w for w in bigrams(documents[i][0])])
    result = []
    [result.extend(w) for w in bigram]
    result = [w for w in result if w[0] not in stopwords and w[1] not in stopwords and w[0] and nije_interpunkcija(w[0]) and nije_interpunkcija(w[1])]
    result = nltk.FreqDist(result)
    return result.keys()

#parametri:
    #koliko - koliko najinformativnijih featura zelimo izdvojiti
    #funkcija na osnovu chi2 testa odabire najinformativnije feature
def feature_selection(documents,koliko,dg1,gg1,dg2,gg2):
    selector = SelectKBest(chi2, koliko)
    y = []
    X = []
    for i in range (dg1,gg1):
        y.append(documents[i][1])
        X.append(words_for_feature_selection(documents[i][0]))
    for i in range (dg2,gg2):
        y.append(documents[i][1])
        X.append(words_for_feature_selection(documents[i][0]))
    X = np.array(X)
    y = np.array(y)
    selector.fit(X,y)
    m = selector.get_support()
    return list(itertools.compress(word_features1,m))

#prima podatke za treniranje, podatke za validaciju i cross validation parametar (cv)
#ako je cv!=0, provodi cross validaciju sa cv podskupova, inace koristi skup za validaciju
#testira algoritam za sve parametre c i g izmedu pocetnic i zavrsnic, odnosno pocetnig i zavrsnig, s korakom korakc, odnosno korakg
#varijabla tip oznacava tip jezgre
def nas_grid(tr_classes,tr_set,val_classes,val_set,cv,pocetnic,zavrsnic,korakc,pocetnig,zavrsnig,korakg,tip):
    dat=open('nas_grid_sinonimi.txt','a+b')
    maks=0    
    px=svm_problem(tr_classes,tr_set)
    i=pocetnic
    while i<=zavrsnic:
        j=pocetnig
        while j<=zavrsnig:
            c=math.pow(2,i)
            g=math.pow(2,j)
            param = svm_parameter()
            param.kernel_type=tip
            param.C = c
            param.gamma = g
            if (cv):
                param.nr_fold=cv
                param.cross_validation=1
                temp=svm_train(px, param)
                print >>dat,str(i) + " "+ str(temp)+ " "
            else:
                m=svm_train(px, param)
                temp=svm_predict(val_classes,val_set,m)
            if(maks<temp):
                maks=temp
                cmaks=c
                gmaks=g
            j=j+korakg
        i=i+korakc
    print >>dat," "
    dat.close()
    return [cmaks,gmaks]

#ispunjava datoteku podacima u formatu priladođenom za funkciju find_parameters
def ispuni_datoteku(ime_datoteke):
    sve=[]
    for i in range(7000):
        sve.append([train_classes[i],train_set[i]])
    f = open(ime_datoteke, 'wb')
    pom=""
    for d in sve:
        pom=pom+str(d[0])+" "
        arr=[]
        for k in d[1]:
            arr.append(k)
        arr.sort()
        for el in arr:
            pom=pom+str(el)+":"+str(d[1][el])+" "       
        print >>f, pom
        pom=""
    f.close()

#-----main funkcija-----#
obj=[]
subj=[]
documents=ucitavanje(obj,subj)
korak = 10000/K
if(len(sys.argv) != 10):
    print "Koristim parametre podesene u kodu"
    print "Za promjenu ponovno pokrenuti program"
    print "Unesite svih 9 argumenata komandne linije s vrijednostima 0 ili 1,osim K i BROJ_FEATURA sljedecim redosljedom"
    print "KORISTENJE_BIGRAMA NAJVECA_FREKVENCIJA OTKLONI_NEPOZELJNE STEM STOP K FEATURE_SELECTION SINONIMI BROJ_FEATURA"
else:
   KORISTENJE_BIGRAMA = int(sys.argv[1])
   NAJVECA_FREKVENCIJA = int(sys.argv[2])
   OTKLONI_NEPOZELJNE = int(sys.argv[3])
   STEM = int(sys.argv[4])
   STOP = int(sys.argv[5])
   K = int(sys.argv[6])
   FEATURE_SELECTION = int(sys.argv[7])
   SINONIMI = int(sys.argv[8])
   BROJ_FEATURA = int(sys.argv[9])

#-----K-struka cross validacija-----#
for cv in range(K):
    word_features1 = []
    print "izdvajam..."
    if(NAJVECA_FREKVENCIJA):
        word_features1 = najveca_frekvencija(obj,subj,0,cv*korak,(cv+1)*korak,10000)
    else:
        word_features1 = omjer_pojavljivanja(documents,0,cv*korak,(cv+1)*korak,10000)
    if(STEM):
        word_features1 = [stem(word) for word in word_features1]
    if(STOP):
        word_features1 = izbaci_stop_words(word_features1)
    if(OTKLONI_NEPOZELJNE):
        word_features1 = otkloni_nepozeljne(word_features1,BROJ_FEATURA)
    if(KORISTENJE_BIGRAMA):
        all_bigrams = bigrami(documents,0,cv*korak,(cv+1)*korak,10000)
        if(FEATURE_SELECTION):
            word_features1.extend(all_bigrams[:5000])
        else:
            word_features1.extend(all_bigrams[:2000])
    if(FEATURE_SELECTION):
        print "feature selection..."
        word_features = feature_selection(documents,BROJ_FEATURA,0,cv*korak,(cv+1)*korak,10000)
    else:
        word_features = word_features1[:BROJ_FEATURA]
    if(SINONIMI):
        print "sinonimi..."
        documents = sinonimi(documents,word_features,cv*korak,(cv+1)*korak)
    print "odabir trening i testing seta..."
    featuresets = [(document_features(d)) for (d,c) in documents]
    classes=[c for (d,c) in documents]
    train_set = featuresets[:cv*korak] + featuresets[(cv+1)*korak:]
    test_set = featuresets[cv*korak:(cv+1)*korak]
    train_classes=classes[:cv*korak] + classes[(cv + 1)*korak:]
    test_classes = classes[cv*korak:(cv+1)*korak] 

    print "trazim parametre za RBF jezgru..."
    ispuni_datoteku('podaci')

    #RBF kernel i implementirani grid
    #ukoliko ne zelite traziti parametre, ovu liniju koda zakomentirajte i dolje direktno upisite zeljene vrijednosti
    ratetest, paramtest = find_parameters('PUT_DO_DIREKTORIJA\podaci', '-log2c 0,4,2 -log2g -8,-4,2')
    px=svm_problem(train_classes,train_set)
    param = svm_parameter()
    param.kernel_type = RBF
    param.C = paramtest['c'] #ovdje mozete direktno upisati zeljene vrijednosti (npr param.C=1)
    param.gamma= paramtest['g'] #ovdje mozete direktno upisati zeljene vrijednosti
    print "treniram..."
    m = svm_train(px, param)
    
    print "testiram..."
    tocnost=svm_predict(test_classes,test_set,m)
    print "tocnost...", tocnost[1][0]

    #Linear kernel i nas grid
    print "trazim parametre za linearnu jezgru..."
    #ukoliko ne zelite traziti parametre, ovu liniju koda zakomentirajte i dolje direktno upisite zeljene vrijednosti
    [a,b]=nas_grid(train_classes,train_set,[],[],4,-6,4,2,0,0,2,0)
    px=svm_problem(train_classes,train_set)
    param = svm_parameter('-t 0')
    param.C = a #ovdje mozete direktno upisati zeljene vrijednosti (npr param.C=1)
    param.gamma=b #primjer za opći slučaj, kod linearne jezgre nepotrebno
    print "treniram..."
    m = svm_train(px, param)
    print "testiram..."
    tocnost = svm_predict(test_classes,test_set,m)
    print "tocnost...", tocnost[1][0]
    
