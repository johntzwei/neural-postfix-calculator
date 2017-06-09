# experiment notes
# ...
# ...

ROOT=/home/johntzwei/Documents/uw2017/neural-postfix-calculator
NAME=`basename $0 | cut -d . -f 1`
EXP_DIR=$ROOT/experiments/$NAME

mkdir $EXP_DIR

#train/test generation
training_ex=`python $ROOT/trees.py data/$NAME-train generateAllTrees --p1 0 --p2 10 --p3 0 --p4 0`
testing_ex=`python $ROOT/trees.py data/$NAME-train generateAllTrees --p1 0 --p2 10 --p3 0 --p4 0`

#train/test lstm with all architectures
python $ROOT/lstm.py $training_ex $testing_ex $EXP_DIR/
