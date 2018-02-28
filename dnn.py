"""
This is a CNN for relation classification within a sentence. The architecture is based on:

Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou and Jun Zhao, 2014, Relation Classification via Convolutional Deep Neural Network

Performance (without hyperparameter optimization):
Accuracy: 0.7943
Macro-Averaged F1 (without Other relation):  0.7612

Performance Zeng et al.
Macro-Averaged F1 (without Other relation): 0.789


Code was tested with:
- Python 2.7 & Python 3.6
- Theano 0.9.0 & TensorFlow 1.2.1
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
nb_epoch = 20
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
model_name = 'att_lstm'


model_dir = 'model/cnn'
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
yTrain, sentenceTrain, positionTrain1, positionTrain2 = data['train_set']
yDev, sentenceDev, positionDev1, positionDev2 = data['dev_set']
yTest, sentenceTest, positionTest1, positionTest2 = data['test_set']

max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1

n_out = max(yTrain) + 1

#train_y_cat = np_utils.to_categorical(yTrain, n_out)
max_sentence_len = sentenceTrain.shape[1]

print("sentenceTrain: ", sentenceTrain.shape)
print("positionTrain1: ", positionTrain1.shape)
print("yTrain: ", yTrain.shape)

print("sentenceDev: ", sentenceDev.shape)
print("positionDev1: ", positionDev1.shape)
print("yDev: ", yDev.shape)


## stack training with dev
# yTrain = np.hstack((yTrain, yDev))
# sentenceTrain = np.vstack((sentenceTrain, sentenceDev))
# positionTrain1 = np.vstack((positionTrain1, positionDev1))
# positionTrain2 = np.vstack((positionTrain2, positionDev2))


target_names = get_target_labels('config/main_config.ini')

max_sentence_len = max(sentenceTrain.shape[1], sentenceDev.shape[1])

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
    with open( os.path.join(model_dir, model_name + ".yaml"), "w") as yaml_file:
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


def init_att_cnn():
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
    # output = GlobalMaxPooling1D()(output)

    output = AttentionWithContext()(output)
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


def init_seq_att_model():
    words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')
    words = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)(words_input)
    distance1_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance1_input')
    distance1 = Embedding(max_position, position_dims)(distance1_input)

    distance2_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance2_input')
    distance2 = Embedding(max_position, position_dims)(distance2_input)


    wm = Sequential()
    wm.add(Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False))

    d1m = Sequential()
    d1m.add(Embedding(max_position, position_dims))

    d2m = Sequential()
    d2m.add(Embedding(max_position, position_dims))

    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(AttentionWithContext())
    model.add(Dense(n_out, activation='sigmoid'))
    # model = Model(input=[words_input, distance1_input, distance2_input], output=output)

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

    model.fit([sentenceTrain, positionTrain1, positionTrain2], yTrain, batch_size=batch_size,
                callbacks=[csv_logger],
                verbose=2, epochs=nb_epoch,
                class_weight=class_weights,
                validation_data=([sentenceTest, positionTest1, positionTest2], yTest)
                )

    # pred_train = predict_classes(model.predict([sentenceTrain, positionTrain1, positionTrain2], verbose=False), pred_tag='train')

    save_model(model_dir, model)

    print("Training done. ")


def try_class_weights():
    model = init_cnn_model()
    # model = init_rnn_model()
    # model = init_att_rnn_model()
    # model = init_cnn_2_model()
    # model = init_att_context_model()

    # model = init_att_cnn()

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    for weight in [.33 * i for i in range(1, 16)]:

        class_weights = {0: 1., 1: weight, 2: weight, 3: weight, 4: weight, 5: weight}
        model.reset_states()
        model.fit([sentenceTrain, positionTrain1, positionTrain2], yTrain, batch_size=batch_size,
                      verbose=0, epochs=nb_epoch,
                      class_weight=class_weights,
                      )

        pred_dev = predict_classes(model.predict([sentenceDev, positionDev1, positionDev2], verbose=False))
        p, r, f1, support = precision_recall_fscore_support(yDev, pred_dev, average='weighted', labels=range(1, 6))
        print("%.3f, %.3f, %.3f, %.3f" % (weight, p, r, f1))

    # pred_train = predict_classes(model.predict([sentenceTrain, positionTrain1, positionTrain2], verbose=False), pred_tag='train')

    # save_model(model_dir, model)

    print("Training done. ")


def do_test(stage='test'):

    model_dir = config.get('main', 'model_dir')

    print("##" * 40)

    print('Stage: %s. starting evaluating using %s set: ' % (stage, stage))

    model = load_model(model_dir)

    if stage == 'dev':
        y_gs = yDev
        pred = predict_classes(model.predict([sentenceDev, positionDev1, positionDev2], verbose=False))
        gs_txt = gs_dev_txt
    elif stage == 'test':
        y_gs = yTest
        pred = predict_classes(model.predict([sentenceTest, positionTest1, positionTest2], verbose=False))
        gs_txt = gs_test_txt
    else:
        raise ValueError("Unsupported stage. Requires either \"dev\" or \"test\".")

    output_tsv = config.get(stage, 'output_tsv')
    gs_tsv = config.get(stage, 'gs_tsv')

    # official eval has different working directory (./eval)

    write_results(os.path.join('eval', output_tsv), gs_txt, pred)
    official_eval(output_tsv, gs_tsv)

    print(classification_report(y_gs, pred, labels=range(1, 6),
                                target_names=target_names[1:],
                                digits=3))

    print(confusion_matrix(y_gs, pred))

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


def submission_run(model_name):
    pred_dev, pred_test = do_training()

    output_dev_tsv = 'data/pred_dev_%s.tsv' % model_name
    output_test_tsv = 'data/pred_test_%s.tsv' % model_name

    # pred_dev = do_dev()

    gs_dev_tsv = '/infodev1/non-phi-data/share_tasks/biocreative/biocreative2017/chemprot/chemprot_development/chemprot_development_gold_standard.tsv'

    print(gs_dev_txt)
    # official eval has different working directory (./eval)
    write_results(os.path.join('eval', output_dev_tsv), gs_dev_txt, pred_dev)
    official_eval(output_dev_tsv, gs_dev_tsv)

    # write_results(os.path.join('eval', output_test_tsv), gs_test_txt, pred_test)


if __name__ == '__main__':
    # submission_run(model_name)
    # load_and_run('model/cnn')
    do_training()
    do_test()
    # do_dev('config/test_config.ini')
