"""
Attention based Recurrent Neural Network for biomedical relation extraction within a sentence.

The architecture is based on:

Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou and Jun Zhao, 2014, Relation Classification via Convolutional Deep Neural Network

Code was tested with:
- Python 2.7
- TensorFlow 1.2.1
- Keras 2.0.5
"""

from __future__ import print_function
import numpy as np
import gzip
import os

import sys

import keras
from keras.models import Model, model_from_yaml, Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate, TimeDistributed
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import LSTM, GRU, Bidirectional, SimpleRNN
from keras.layers.merge import Concatenate
from keras.regularizers import Regularizer
from keras.preprocessing import sequence
from keras.callbacks import CSVLogger


from attention_lstm import AttentionWithContext
from keras.initializers import RandomNormal, RandomUniform


from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, confusion_matrix

from annot_util.config import get_stage_config, get_target_labels

if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

np.random.seed(42)  # for reproducibility

config = get_stage_config('config/main_config.ini')

# CNN
batch_size = 64
nb_filter = 200
filter_length = 3

# Common
nb_epoch = 5
position_dims = 50

dropout_rate = 0.2

lstm_units = 128

weights = 1
class_weights = {0:1., 1:weights, 2:weights, 3:weights, 4:weights, 5:weights}

# mode = 'original_candidate'

mode = 'ent_candidate'

# mode = 'ent_candidate_40'


model_name = 'cnn'
# model_name = 'att_gru'
# model_name = 'gru'
# model_name = 'att_lstm'


model_dir = 'model'

pkl_path = 'pkl/bioc_rel_%s.pkl.gz' % mode
root_dir = 'data/org_ent'
fns = ['training.txt', 'development.txt', 'test.txt']
files = [os.path.join(root_dir, fn) for fn in fns]


# mode = 'split'
# pkl_path = 'pkl/i2b2_rel_within_sent_%s.pkl.gz' % mode

# pkl_path = 'pkl/timer_rel_contain_%s.pkl.gz' % mode

print("mode: " + mode)


gs_dev_txt = files[1]
gs_test_txt = files[2]

print("Loading dataset")
# f = gzip.open('pkl/sem-relations.pkl.gz', 'rb')
f = gzip.open(pkl_path, 'rb')

data = pkl.load(f)
f.close()

embeddings = data['wordEmbeddings']
y_train, sentence_train, position_train1, position_train2 = data['train_set']
y_dev, sentence_dev, position_dev1, position_dev2 = data['dev_set']
y_test, sentence_test, position_test1, position_test2 = data['test_set']

max_position = max(np.max(position_train1), np.max(position_train2)) + 1

n_out = max(y_train) + 1

#train_y_cat = np_utils.to_categorical(yTrain, n_out)
max_sentence_len = sentence_train.shape[1]

print("sentenceTrain: ", sentence_train.shape)
print("positionTrain1: ", position_train1.shape)
print("yTrain: ", y_train.shape)

print("sentenceDev: ", sentence_dev.shape)
print("positionDev1: ", position_dev1.shape)
print("yDev: ", y_dev.shape)


## stack training with dev
y_train = np.hstack((y_train, y_dev))
sentence_train = np.vstack((sentence_train, sentence_dev))
position_train1 = np.vstack((position_train1, position_dev1))
position_train2 = np.vstack((position_train2, position_dev2))


target_names = get_target_labels('config/main_config.ini')

max_sentence_len = max(sentence_train.shape[1], sentence_dev.shape[1])

print("class weights:")
print(class_weights)


def predict_classes(prediction, pred_tag=''):
    # save probabilities for dev set
    if pred_tag != '':
        np.savetxt('output/pred_prob_%s_%s.txt' % (pred_tag, model_name), prediction, fmt="%.5f")
    return prediction.argmax(axis=-1)


def save_model(model_dir, model):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(os.path.join(model_dir, model_name + ".yaml"), "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights( os.path.join(model_dir, model_name + ".h5"))

    print("Model saved to disk: " + model_dir)


def load_model(model_dir):
    # load YAML and create model
    yaml_file = open(os.path.join(model_dir, model_name + '.yaml'), 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml, custom_objects={'AttentionWithContext': AttentionWithContext})

    # load weights into new model
    loaded_model.load_weights(os.path.join(model_dir, model_name + '.h5'))
    print("Loaded model %s from disk: %s" % (model_name, model_dir))

    return loaded_model


def init_cnn_model():
    print("Embeddings: ", embeddings.shape)

    words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')
    words = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)(words_input)

    distance1_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance1_input')
    distance1 = Embedding(max_position, position_dims)(distance1_input)

    distance2_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance2_input')
    distance2 = Embedding(max_position, position_dims)(distance2_input)

    output = concatenate([words, distance1, distance2])

    output = Convolution1D(filters=nb_filter,
                           kernel_size=filter_length,
                           padding='same',
                           activation='tanh',
                           strides=1)(output)

    # we use standard max over time pooling
    output = GlobalMaxPooling1D()(output)

    output = Dropout(dropout_rate)(output)
    output = Dense(n_out, activation='softmax')(output)

    model = Model(inputs=[words_input, distance1_input, distance2_input], outputs=[output])

    return model


def init_att_lstm_model():
    words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')
    words = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)(words_input)
    distance1_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance1_input')
    distance1 = Embedding(max_position, position_dims)(distance1_input)

    distance2_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance2_input')
    distance2 = Embedding(max_position, position_dims)(distance2_input)

    output = concatenate([words, distance1, distance2])

    output = SimpleRNN(lstm_units, return_sequences=True, dropout=dropout_rate)(output)
    output = AttentionWithContext()(output)
    output = Dense(n_out, activation='sigmoid')(output)
    model = Model(inputs=[words_input, distance1_input, distance2_input], outputs=output)

    return model


def init_att_gru_model():
    words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')
    words = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)(words_input)
    distance1_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance1_input')
    distance1 = Embedding(max_position, position_dims)(distance1_input)

    distance2_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance2_input')
    distance2 = Embedding(max_position, position_dims)(distance2_input)

    output = concatenate([words, distance1, distance2])

    output = GRU(lstm_units, return_sequences=True, dropout=dropout_rate)(output)
    output = AttentionWithContext()(output)
    output = Dense(n_out, activation='sigmoid')(output)
    model = Model(inputs=[words_input, distance1_input, distance2_input], outputs=output)

    return model

def init_rnn_model():
    words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')
    words = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)(words_input)
    distance1_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance1_input')
    distance1 = Embedding(max_position, position_dims)(distance1_input)

    distance2_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance2_input')
    distance2 = Embedding(max_position, position_dims)(distance2_input)

    output = concatenate([words, distance1, distance2])

    # initializer = RandomUniform(minval=-0.54, maxval=0.5, seed=None)

    output = GRU(lstm_units, return_sequences=False, dropout=dropout_rate)(output)

    output = Dense(n_out, activation='sigmoid')(output)
    model = Model(inputs=[words_input, distance1_input, distance2_input], outputs=output)

    return model


def do_training():

    init_func = {
        'cnn': init_cnn_model,
        'att_gru': init_att_gru_model,
        'att_lstm': init_att_lstm_model,
        'gru': init_rnn_model,
    }

    model = init_func[model_name]()

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    csv_logger = CSVLogger('run/training_%s.log' % model_name)

    model.fit([sentence_train, position_train1, position_train2], y_train, batch_size=batch_size,
              callbacks=[csv_logger],
              verbose=2, epochs=nb_epoch,
              class_weight=class_weights,
              validation_data=([sentence_test, position_test1, position_test2], y_test)
              )

    # pred_train = predict_classes(model.predict([sentenceTrain, positionTrain1, positionTrain2], verbose=False), pred_tag='train')

    save_model(model_dir, model)

    print("Training done. Model saved at: ")


def do_test(stage='test'):


    print("##" * 40)

    print('Stage: %s. starting evaluating using %s set: ' % (stage, stage))

    model = load_model(model_dir)

    if stage == 'dev':
        y_gs = y_dev
        pred = predict_classes(model.predict([sentence_dev, position_dev1, position_dev2], verbose=False))
        gs_txt = gs_dev_txt
    elif stage == 'test':
        y_gs = y_test
        pred = predict_classes(model.predict([sentence_test, position_test1, position_test2], verbose=False))
        gs_txt = gs_test_txt
    else:
        raise ValueError("Unsupported stage. Requires either \"dev\" or \"test\".")

    output_tsv = config.get(stage, 'output_tsv')
    gs_tsv = config.get(stage, 'gs_tsv')

    # official eval has different working directory (./eval)

    write_results(os.path.join('eval', output_tsv), gs_txt, pred)
    official_eval(output_tsv, gs_tsv)

    print(y_gs)
    print(pred)

    print(confusion_matrix(y_gs, pred))

    print(classification_report(y_gs, pred, labels=range(1, 6),
                                target_names=target_names[1:],
                                digits=3))

    return pred


def write_results(output_tsv, gs_path, pred):
    """
    Write list of output in official format
    :param output_tsv:
    :param pred:
    :return:
    """
    ft = open(gs_path)
    lines = ft.readlines()
    assert len(lines) == len(pred),  'line inputs does not match: input vs. pred : %d / %d' % (len(lines), len(pred))

    with open(output_tsv, 'w') as fo:
        for pred_idx, line in zip(pred, lines):
            splits = line.strip().split('\t')
            if target_names[pred_idx] == "NA":
                continue

            fo.write("%s\t%s\tArg1:%s\tArg2:%s\n" %
                     (splits[-1], target_names[pred_idx],
                     splits[-3], splits[-2],
                     ))
            # fo.write("%s\t%s" % (target_names[pred_idx], line))
    print("results written: " + output_tsv)
    ft.close()


def official_eval(output_tsv, gs_tsv):
    """
    Run official evaluation
    :param output_tsv:
    :param gs_tsv:
    :return:
    """
    os.chdir('eval')
    os.system("./eval.sh %s %s" % (output_tsv, gs_tsv))
    os.chdir('..')
    print()

if __name__ == '__main__':
    do_training()
    do_test()
