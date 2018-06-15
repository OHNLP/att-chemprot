# Attention based Neural Networks for Chemical Protein Relation

* Task description: http://www.biocreative.org/tasks/biocreative-vi/track-5/
* Data (may require login): http://www.biocreative.org/accounts/login/?next=/resources/corpora/chemprot-corpus-biocreative-vi/


## Prerequisites

* Python 2.7
* TensorFlow 1.2.1
* Keras 2.0.5
* NLTK
* Scikit-learn
* ConfigParser

## Usage

### Update configuration file
 Go through the config file `config/main_config.ini` to modify the
 following paths
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

The output file is in [Brat standoff format](http://brat.nlplab.org/standoff.html),
the same as the gold standard files. Each epoch of ATT-GRU will take 83
seconds on a NVIDIA Tesla P40 GPU to complete.


The offical envaluation script can be downloaded at the offical site:
 [ChemProt evaluation kit](http://www.biocreative.org/media/store/files/2017/evaluation-kit.zip)
A copy is also provided under `./eval` for the convenience of usage, and it will be called
automatically after the output file on the test set is generented.

The official result is:

```
Total annotations: 3458
Total predictions: 4184
TP: 2003
FN: 1455
FP: 2181
Precision: 0.47872848948374763
Recall: 0.5792365529207635
F-score: 0.5242083224286836
```

The confusion matrix and classification report will also be printed.

Confusion Matrix:
```
[[8028  382  918  120  236  301]
 [ 189  308   92    0    7    2]
 [ 315   33 1142    7    8    7]
 [  47    4    3   95   11    1]
 [  67    0    8    8  187    0]
 [ 265   11   27    0    0  266]]
```
Classification Report:
```
             precision    recall  f1-score   support

      CPR:3      0.417     0.515     0.461       598
      CPR:4      0.521     0.755     0.617      1512
      CPR:5      0.413     0.590     0.486       161
      CPR:6      0.416     0.693     0.520       270
      CPR:9      0.461     0.467     0.464       569

avg / total      0.476     0.642     0.544      3110
```

### Visualization of Attention

Please follow the Jupyter Notebook `model_att_vis.ipynb` for details.
It can also be done via:
```
python model_att_vis.py
```


## Reference

> Sijia Liu, Feichen Shen, Y Wang, M Rastegar-Mojarad, RK Elayavilli, V Chaudhary, H Liu. *Attention-based Neural Networks for Chemical Protein Relation Extraction*. BioCreative VI Workshop Proceedings. 2017. [Slides](http://www.acsu.buffalo.edu/~sijialiu/uploads/slides_bioc_17.pdf)
