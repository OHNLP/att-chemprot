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
Total predictions: 2939
TP: 1687
FN: 1771
FP: 1252
Precision: 0.5740047635250085
Recall: 0.48785425101214575
F-score: 0.5274347350320463
```

The confusion matrix and classification report will also be printed.

Confusion Matrix:
```
[[8859  138  573   70   73  272]
 [ 344  212   37    1    1    3]
 [ 485   33  969    3   11   11]
 [  69    4    1   86    0    1]
 [ 125    0    5    4  136    0]
 [ 274    3   12    0    0  280]]
```
Classification Report:
```
             precision    recall  f1-score   support

      CPR:3      0.544     0.355     0.429       598
      CPR:4      0.607     0.641     0.623      1512
      CPR:5      0.524     0.534     0.529       161
      CPR:6      0.615     0.504     0.554       270
      CPR:9      0.494     0.492     0.493       569

avg / total      0.570     0.541     0.551      3110

```

### Visualization of Attention

Please follow the Jupyter Notebook `model_att_vis.ipynb` for details.

Examples:

![attention-vis-6](png_examples/6.png?raw=true "Examples-6")

![attention-vis-96](png_examples/96.png?raw=true "Examples-96")



## Acknowledgment

The relation classification architechture is based on the
 [Relation CNN implementation](https://github.com/UKPLab/deeplearning4nlp-tutorial/tree/master/2017-07_Seminar/Session%203%20-%20Relation%20CNN)
from [UKPLab](https://github.com/UKPLab).

The attention RNN is inspired by the code snippet by cbaziotis: https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2.

## Reference

> Sijia Liu, Feichen Shen, Y Wang, M Rastegar-Mojarad, RK Elayavilli, V Chaudhary, H Liu. *Attention-based Neural Networks for Chemical Protein Relation Extraction*. BioCreative VI Workshop Proceedings. 2017. [[PDF](http://www.biocreative.org/media/store/files/2018/BC6_track5_4.pdf)] [[Slides](http://www.acsu.buffalo.edu/~sijialiu/uploads/slides_bioc_17.pdf)]
