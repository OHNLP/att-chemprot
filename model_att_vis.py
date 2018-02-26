
# coding: utf-8

# In[1]:
import matplotlib

matplotlib.use('Agg')

import keras
import matplotlib.pyplot as plt



# In[2]:

from cnn import load_model, predict_classes, data
import gzip
import numpy as np
# from attention_utils import get_activations

from attention_lstm import AttentionWithContext


import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

mode = 'ent_candidate'


# In[3]:

model_dir = 'model/cnn'
model = load_model(model_dir)


# ## Plot DNN model diagram

# In[4]:

from keras.utils import plot_model
plot_model(model, to_file='output/plots/att-lstm-model.png')


# In[5]:

import keras.backend as K
import numpy as np


# In[6]:

import os 
print("mode: " + mode)

embeddings = data['wordEmbeddings']

yTrain, sentenceTrain, positionTrain1, positionTrain2 = data['train_set']
yTest, sentenceTest, positionTest1, positionTest2  = data['dev_set']

max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1

n_out = max(yTrain)+1
#train_y_cat = np_utils.to_categorical(yTrain, n_out)
max_sentence_len = sentenceTrain.shape[1]

print("sentenceTrain: ", sentenceTrain.shape)
print("positionTrain1: ", positionTrain1.shape)
print("yTrain: ", yTrain.shape)

print("sentenceTest: ", sentenceTest.shape)
print("positionTest1: ", positionTest1.shape)
print("yTest: ", yTest.shape)

print("Embedding: ", embeddings.shape)

###
time_steps = sentenceTest.shape[1]


# In[7]:

testing_inputs_1 = [sentenceTest, positionTest1, positionTest2]
# print testing_inputs_1
# td = np.transpose(testing_inputs_1, (1, 0, 2))
# print td.shape


# In[8]:

model.predict(testing_inputs_1)

inputs = testing_inputs_1


# In[9]:

def get_activations_2(model, model_inputs, print_shape_only=False, layer_name=None):
#     print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(1.)
    else:
        list_inputs = [model_inputs, 1.]

    # Learning phase. 1 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 1.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
#         if print_shape_only:
#             print(layer_activations.shape)
#         else:
#             print(layer_activations)
    return activations


# In[ ]:

sent_splits = []
with open('data/org_ent/development.txt') as f:
    for line in f:
        splits = line.strip().split('\t')
        # sentence = splits[5]
        # label = splits[0]
        sent_splits.append(splits)


# In[ ]:

activation = get_activations_2(model, testing_inputs_1, print_shape_only=True)  # with just one sample.


# In[ ]:

idx = 1

layer_idx = -2
activation[layer_idx][idx].shape


# In[ ]:




# In[ ]:

"""
from keras.models import Model

layer_name = 'attention_with_context_1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(testing_inputs_1)
"""


# In[ ]:
layer_name = 'attention_with_context_1'
W, b, u = model.get_layer(layer_name).get_weights()


# In[ ]:

import numpy as np

def get_att_weights(i):
    """
    Run attention layer using LSTM outputs
    """

    x = activation[-3][i]

    uit = np.dot(x, W)

    uit += b

    uit = np.tanh(uit)
    mul_a = uit * u
    ait = np.sum(mul_a, axis=1)

    a = np.exp(ait)     # attention weights

    a /= np.sum(a)

    return a

to_plot = True
if to_plot:
    for sent_idx in range(1000):

        rel_instance = sent_splits[sent_idx]

        tokens = rel_instance[5].split()

        sent_len = len(tokens)
        if rel_instance[0] == 'NA':
            continue

        fig = plt.figure(figsize=(len(tokens), 2))

        chem_ent_tk_id, gene_ent_tk_id = int(rel_instance[3]), int(rel_instance[4])

        values = get_att_weights(sent_idx)[:sent_len]

        fig = plt.figure(figsize=(sent_len, 2))
        # make 170 x 3 matrix for image plot.
        x_matrix = np.array([values for _ in range(3)])

        ax = plt.imshow(x_matrix, aspect='auto', interpolation='nearest', cmap='Blues', origin='upper').get_axes()

        # plt.xticks(range(len(label)), label, size='medium')
        # plt.tight_layout(pad=2, w_pad=2, h_pad=2)

        for i, token in enumerate(tokens):
            color = 'black'
            if values[i] > 0.9 * values.max():
                color = 'white'

            tk2plot = token.decode("utf-8", "ignore")
            if i == chem_ent_tk_id or i == gene_ent_tk_id:
                tk2plot = "[%s]" % tk2plot

            ax.text(i, 0, tk2plot, horizontalalignment='center', color=color)

        ax.set_ylim([-1, 0.8])
        plt.axis('off')

        plt.tight_layout()

        plt.title("%s id:%d" % (mode, sent_idx))

        plt.savefig('output/att_vis/dev_%d.png' % sent_idx)
        plt.close()


# ## Get top words

# In[ ]:

from collections import Counter

fo = open('output/kw/att_keywords_top3.txt', 'w')

top_word_list = []

from nltk.stem.porter import PorterStemmer
st = PorterStemmer()

for sent_idx, sent_split in enumerate(sent_splits):
    values = get_att_weights(sent_idx)
    tokens = sent_split[5].split()
    values = values[:len(tokens)]

    label = sent_split[0]

    top_n = 3
    # select top n for counts
    values = np.array(values)
    tokens = np.array(tokens)
    top_index = np.argsort(values)[::-1][:top_n]
    for top_token in tokens[top_index]:
        fo.write("%s\t%s\n" % (label, top_token))

fo.close()
print Counter([st.stem(token) for _, token in top_word_list])

