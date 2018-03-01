# eval.sh
# script to evaluate predictions against the gold-standard 
# the main evaluation metric for this task will be the micro-averaged F1-score
# See the readme file for details
#
PARAM1=$1
PARAM2=$2
PARAM3=$3
PARAM4=$4
PARAM5=$5
PARAM6=$6
PARAM7=$7
PARAM8=$8
java -cp bc6chemprot_eval.jar org.biocreative.tasks.chemprot.main.Main $PARAM1 $PARAM2 $PARAM3 $PARAM4 $PARAM5 $PARAM6 $PARAM7 $PARAM8

cat out/eval.txt
