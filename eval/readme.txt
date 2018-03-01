


Contents
========
1. Introduction
2. Installation and Execution
3. Examples Execution Sequences
4. Output
5. Contact





1. Introduction
===============

This evaluation kit contains: 

- bc6chemprot_eval.jar: a JAR file that aggregates many Java class files for evaluating the performance of BioCreative VI CHEMPROT task. 

- readme.txt: this file. 

- eval.sh: script to make evaluation easier. It allows 4 optional parameters and 1 mandatory. 

- data: this folder contains the gold-standard file. The user is free to include in it her/his prediction file or/and her/his gold-standard file. 

- out: it will store the evaluation results. 


This file contains information on how to install, configure and run the evaluation script (see bellow).

2. Installation and Execution
=============================

The source code is written in Java, so it should be platform independent. 
Users need to install Java in their environment. 
Scripts are provided for the UNIX command line. 

2.1 Unpacking
-------------

A. Copy bc6chemprot.evaluation.tar.gz to the directory in which the 
   distribution should be unpacked. From now on I will refer to 
   this directory as <root>.
B. Use 'tar xvfz bc6chemprot.evaluation.tar.gz' to unpack the distribution.
   Inside the <root> folder will be the five items shown in the 'introduction' section. 

2.2 Execution
-------------

The execution is very straightforward. To evaluate your prediction file, simply run: 

sh eval.sh [options] <input-prediction-file> [input-gold-file]

[options]:
  -TP <boolean>: Create an output file with 'True Positive' := Predictions in common with Gold. FALSE (default) or TRUE.
  -FP <boolean>: Create an output file with 'False Positive' := Predictions - TP. FALSE (default) or TRUE.
  -FN <boolean>: Create an output file with 'False Negative' := Gold - TP. FALSE (default) or TRUE.

<input-prediction-file>: TSV Prediction file. This parameter is required.

[input-gold-file]: TSV Gold file. If it is not provided will be the following: ./data/chemprot_sample_gold_standard.tsv

3. Examples Execution Sequences
===============================

Assume that $WORKDIR is your working directory and that you have the prediction file at $WORKDIR/prediction.tsv. 
The following execution sequences can be used to evaluate your prediction file against the gold-standard file. 

For example, the execution sequence to evaluate the prediction file at <root>/data/sample_chemprot_predictions.tsv is the following:

sh eval.sh <root>/data/sample_chemprot_predictions.tsv 

The execution sequence to evaluate your prediction file without optional parameters is the following: 

sh eval.sh $WORKDIR/prediction.tsv

The following execution sequence can be used to evaluate your prediction file and also obtain a file with the TPs (True Positive):  

sh eval.sh -TP TRUE $WORKDIR/prediction.tsv

Similar execution sequences can be used to obtain FPs (False Positive) and FNs (False Negative) files. 
For instance, the execution sequence that obtains the three tests is the following: 

sh eval.sh -TP TRUE -FP TRUE -FN TRUE $WORKDIR/prediction.tsv

Finally, we shown the execution sequence that you should run in the case that you want to use your own gold-standard file. 
We assume that your gold-standard file is at $WORKDIR/gold.tsv, and you do not want to use optional parameters: 

sh eval.sh $WORKDIR/prediction.tsv $WORKDIR/gold.tsv

4. Output
=========

The evaluation output is stored at the 'out' folder. 
Four plain text files are possible store at that folder:

- eval.txt: contains the result of the evaluation itself: total annotations, total predictions, TP, FN, FP, precision, recall, and F-score.
            An example of this file could be the following: 

Total annotations: 239
Total predictions: 142
TP: 121
FN: 118
FP: 21
Precision: 0.852112676056338
Recall: 0.5062761506276151
F-score: 0.6351706036745408

- tp.txt: this file is optional (see the section 3 to obtain it). Shows the TPs obtained by your prediction file.
          An example of this file could be the following: 

12453616	CPR:3	Arg1:T6	Arg2:T17
12453616	CPR:3	Arg1:T6	Arg2:T18
12453616	CPR:3	Arg1:T1	Arg2:T10

- fn.txt: this file is optional (see the section 3 to obtain it). Shows the FNs obtained by your prediction file.
          An example of this file could be the following: 

23538162	CPR:4	Arg1:T5	Arg2:T19
23538162	CPR:4	Arg1:T5	Arg2:T20
23538162	CPR:6	Arg1:T7	Arg2:T21

- fp.txt: this file is optional (see the section 3 to obtain it). Shows the FPs obtained by your prediction file.
          An example of this file could be the following: 

10403635	CPR:4	Arg1:T35	Arg2:T40
10403635	CPR:4	Arg1:T35	Arg2:T42
10403635	CPR:9	Arg1:T16	Arg2:T40

5. Contact
==========

If you have any questions, remarks, bug reports, bug fixes or extensions, we will be happy to hear from you.

Martin Krallinger
krallinger.martin@gmail.com

Jesús Santamaría
jesus.sant@telefonica.net











