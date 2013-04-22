import random
import math
import copy
import nltk

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

f=open("obj.txt")
obj=[]
for i in range(5000):
    line=f.readline()
    vector=line.split()
    obj.append(vector)
f.close()

f=open("subj.txt")
subj=[]
for i in range(5000):
    line=f.readline()
    vector=line.split()
    subj.append(vector)
f.close()
documents=[]
all_words=[]
for i in range(5000):
    documents.append([obj[i][:], 'objective'])
    documents.append([subj[i][:], 'subjective'])
    
random.shuffle(documents)
all=obj[:][:] + subj[:][:]
for i in range(10000):
    k=all[i][:]
    m=len(k)
    for j in range(m):
        all_words.append(all[i][j])  
all_words = nltk.FreqDist(all_words)
word_features = all_words.keys()[:2000]

featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[3000:], featuresets[:3000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, test_set)
classifier.show_most_informative_features(100)

print word_features[0:50]

