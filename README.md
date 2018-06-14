# Attention based Neural Networks for Chemical Protein Relation

Task description: http://www.biocreative.org/tasks/biocreative-vi/track-5/
Data (may require login): http://www.biocreative.org/accounts/login/?next=/resources/corpora/chemprot-corpus-biocreative-vi/


## Prerequisites

* Python 3.6 (should work for 2.7 too)
* TensorFlow 1.2.1
* Keras 2.0.5
* NLTK
* Scikit-learn

## Usage

### Update configuration file
 Go through the config file `config/main_config.ini` to modify the paths
 accordingly.
    * `corpus_dir`: unzipped corpus directory
    * `out_dir`: the output directory of preprocessed file

### Preprocessing
```
python extract_sentences.py
```

By default, you will see the relation instances of train, dev and test
sets under `out_dir`: `training.txt`, `development.txt` and `test.txt`.

### Encode Sentences

Load word embeddings and generate word index (word2id) for the corpus by
 running:
```
python preprocess.py
```

A subset of word embeddings, vocabulary, and sentencse are stored under
the compressed pkl file
 `pkl/bioc_rel_ent_candidate.pkl.gz`

### Train Models

Load the encoded sentences, initalize model parameters, compile
Tensorflow and Keras models, and run training and testing on
the dataset by:

```
python dnn.py
```

The output file is in TSV format, the same as the gold standard files.
The offical envaluation script (locates under `./eval`) will be
automatically called.

The official results will be like:

```
Total annotations: 3458
Total predictions: 2358
TP: 1492
FN: 1966
FP: 866
Precision: 0.6327396098388465
Recall: 0.43146327356853675
F-score: 0.5130674002751032
```

The confusion matrix and classification report will also be printed:

```
Confusion Matrix:
[[9226  185  357   79   30  108]
 [ 309  257   31    0    0    1]
 [ 599   40  860    3    4    6]
 [  63    3    1   93    0    1]
 [ 151    0    2    6  111    0]
 [ 389    0   14    0    0  166]]

Classification Report:
             precision    recall  f1-score   support

      CPR:3      0.530     0.430     0.475       598
      CPR:4      0.680     0.569     0.619      1512
      CPR:5      0.514     0.578     0.544       161
      CPR:6      0.766     0.411     0.535       270
      CPR:9      0.589     0.292     0.390       569

avg / total      0.633     0.478     0.538      3110
```

### Visualization of Attention

Please follow the Jupyter Notebook `model_att_vis.ipynb` for details.
It can also be done via:
```
python model_att_vis.py
```


## Reference

> Sijia Liu, Feichen Shen, Y Wang, M Rastegar-Mojarad, RK Elayavilli, V Chaudhary, H Liu. *Attention-based Neural Networks for Chemical Protein Relation Extraction*. BioCreative VI Workshop Proceedings. 2017. [Slides](http://www.acsu.buffalo.edu/~sijialiu/uploads/slides_bioc_17.pdf)
