"""
The file preprocesses the files/train.txt and files/test.txt files.

I requires the dependency based embeddings by Levy et al.. Download them from his website and change 
the embeddingsPath variable in the script to point to the unzipped deps.words file.
"""
from __future__ import print_function
import numpy as np
import gzip
import os
import sys


if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

import codecs

mode = 'ent_candidate'

print("mode: " + mode)
pkl_path = 'pkl/bioc_rel_%s.pkl.gz' % mode
root_dir = 'data/org_ent'
fns = ['training.txt', 'development.txt', 'test.txt']
files = [os.path.join(root_dir, fn) for fn in fns]

print(files)
#We download English word embeddings from here https://www.cs.york.ac.uk/nlp/extvec/
# embeddingsPath = 'embeddings/wiki_extvec.gz'
embeddingsPath = os.path.join(os.environ['WE_PATH'], 'glove.6B.300d.txt')
# embeddingsPath = os.path.join(os.environ['WE_PATH'], 'pmc.w2v.vector.wf=7.itr=1.layer=100')



#Mapping of the labels to integers
rel2id = {'NA': 0,
            'CPR:3': 1,
            'CPR:4': 2,
            'CPR:5': 3,
            'CPR:6': 4,
            'CPR:9': 5,
          }


words = {}
maxSentenceLen = [0] * len(files)


distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
minDistance = -30
maxDistance = 30
for dis in range(minDistance, maxDistance+1):
    distanceMapping[dis] = len(distanceMapping)


def createMatrices(file, word2Idx, maxSentenceLen=100):
    """Creates matrices for the events and sentence for the given file"""
    labels = []
    positionMatrix1 = []
    positionMatrix2 = []
    tokenMatrix = []
    
    for line in open(file):

        splits = line.strip().split('\t')
        
        label = splits[0]

        pos1 = splits[3]
        pos2 = splits[4]
        sentence = splits[5]
        tokens = sentence.split(" ")
        

        tokenIds = np.zeros(maxSentenceLen)
        positionValues1 = np.zeros(maxSentenceLen)
        positionValues2 = np.zeros(maxSentenceLen)
        
        for idx in range(0, min(maxSentenceLen, len(tokens))):
            tokenIds[idx] = getWordIdx(tokens[idx], word2Idx)
            
            distance1 = idx - int(pos1)
            distance2 = idx - int(pos2)
            
            if distance1 in distanceMapping:
                positionValues1[idx] = distanceMapping[distance1]
            elif distance1 <= minDistance:
                positionValues1[idx] = distanceMapping['LowerMin']
            else:
                positionValues1[idx] = distanceMapping['GreaterMax']
                
            if distance2 in distanceMapping:
                positionValues2[idx] = distanceMapping[distance2]
            elif distance2 <= minDistance:
                positionValues2[idx] = distanceMapping['LowerMin']
            else:
                positionValues2[idx] = distanceMapping['GreaterMax']

        # if rel2id[label] > 5:
        #     continue

        tokenMatrix.append(tokenIds)
        positionMatrix1.append(positionValues1)
        positionMatrix2.append(positionValues2)

        # print(rel2id[label])
        labels.append(rel2id[label])

    return np.array(labels, dtype='int32'), np.array(tokenMatrix, dtype='int32'),\
           np.array(positionMatrix1, dtype='int32'), np.array(positionMatrix2, dtype='int32'),
        
        
        
 
def getWordIdx(token, word2Idx): 
    """Returns from the word2Idex table the word index for a given token"""       
    if token in word2Idx:
        return word2Idx[token]
    elif token.lower() in word2Idx:
        return word2Idx[token.lower()]
    
    return word2Idx["UNKNOWN_TOKEN"]


for fileIdx in range(len(files)):
    file = files[fileIdx]
    print(file)
    for line in open(file):
        splits = line.strip().split('\t')
        
        label = splits[0]
        sentence = splits[5]
        tokens = sentence.split(" ")
        maxSentenceLen[fileIdx] = max(maxSentenceLen[fileIdx], len(tokens))
        for token in tokens:
            words[token.lower()] = True
            

print("Max Sentence Lengths: ", maxSentenceLen)
        
# :: Read in word embeddings ::
# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []

fEmbeddings = codecs.open(embeddingsPath)

print("Load pre-trained embeddings file")
for line in fEmbeddings:
    split = line.decode('utf-8').strip().split(" ")
    word = split[0]
    
    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)
        
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        wordEmbeddings.append(vector)

    if word.lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[word] = len(word2Idx)
       
        
wordEmbeddings = np.array(wordEmbeddings)

print("Embeddings shape: ", wordEmbeddings.shape)
print("Len words: ", len(words))


# :: Create token matrix ::
# train_set = createMatrices(files[0], word2Idx, max(maxSentenceLen))
# dev_set = createMatrices(files[1], word2Idx, max(maxSentenceLen))
# test_set = createMatrices(files[2], word2Idx, max(maxSentenceLen))

sent_length = 170
train_set = createMatrices(files[0], word2Idx, sent_length)
dev_set = createMatrices(files[1], word2Idx, sent_length)
test_set = createMatrices(files[2], word2Idx, sent_length)

print("training token matrix shape: " + str(train_set[1].shape))
print("dev token matrix shape: " + str(dev_set[1].shape))
print("testing token matrix shape: " + str(test_set[1].shape))


data = {'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx, 
        'train_set': train_set, 'dev_set': dev_set, 'test_set': test_set}

f = gzip.open(pkl_path, 'wb')
pkl.dump(data, f)
f.close()

print("Data stored in pkl folder")