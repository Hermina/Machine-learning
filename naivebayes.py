import random
import math
import copy
import nltk

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['%s' % word] = (word in document_words)
    return features

def stem(word):
    #for suffix in ['ed', 'ies', 'y', 'es', 's']:
    #    if word.endswith(suffix):
     #       word=word[:-len(suffix)]
    l=word.split("'")
    return l[0]
    
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
    for j in range(len(all[i])):
        all_words.append(all[i][j])
all_words = [word for word in all_words if (word!='.' and word!=',' and word!=')' and word!='(' and word!='a' and word!='is' and word!='the' and word!='and')]
#all_words = [stem(word) for word in all_words]
all_words = nltk.FreqDist(all_words)
word_features = all_words.keys()[:2000]

featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, devtest_set, test_set = featuresets[:4000], featuresets[4000:7000], featuresets[7000:]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, devtest_set)

errors = []
##for (df, tag) in devtest_set:
##    guess = classifier.classify(df)
##    if guess != tag:
##        errors.append( (tag, guess, df) )
##for (tag, guess, df) in sorted(errors): # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
##    lista=[key for key in df.keys() if df[key]==True]
##    print 'correct=%-8s guess=%-8s features=%-30s' %(tag, guess, lista)
classifier.show_most_informative_features(20)


