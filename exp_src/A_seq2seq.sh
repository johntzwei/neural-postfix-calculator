# experiment notes
# tests whether 2 layer lstm with 10 hidden dim can predict
# 0 depth expressions of arbitrary length, e.g. [133] = 133

ROOT=/home/jwei/neural-postfix-calculator
NAME=`basename $0 | cut -d . -f 1`
EXP_DIR=$ROOT/experiments/$NAME

SERIES='preliminary_seq2seq'

EPOCHS=1
BATCH_SIZE=32
TRAIN_TREE_TYPE="generateAllTrees"
TRAIN_P1=0
TRAIN_P2=3
TRAIN_P3=01
TRAIN_P4=2
TRAIN_P5=0

TEST_SPLIT_TRAIN=true
TEST_PER=20         #percentage out of 100

TEST_TREE_TYPE=""
TEST_P1=0
TEST_P2=0
TEST_P3=0
TEST_P4=0
TEST_P5=0

mkdir -p $EXP_DIR

#train/test generation
train_ex="$ROOT/data/$NAME-train_${TRAIN_TREE_TYPE}_${TRAIN_P1}_${TRAIN_P2}_${TRAIN_P3}_${TRAIN_P4}_${TRAIN_P5}"

if [ ! $TEST_SPLIT_TRAIN ]; then
    test_ex="$ROOT/data/$NAME-test_${TEST_TREE_TYPE}_${TEST_P1}_${TEST_P2}_${TEST_P3}_${TEST_P4}_${TEST_P5}"
else
    test_ex="$ROOT/data/$NAME-test_SPLIT_${TEST_PER}"
fi
echo "training data will be saved to $train_ex"
echo "testing data will be saved to $test_ex"

if [ ! -f $train_ex ]; then
    echo "generating train data"
    python $ROOT/trees.py $TRAIN_TREE_TYPE --p1 $TRAIN_P1 --p2 $TRAIN_P2 --p3 $TRAIN_P3 --p4 $TRAIN_P4 --p5 $TRAIN_P5 --infix > $train_ex

    echo "shuffling train data"
    shuf $train_ex > ${train_ex}.shuf
    mv ${train_ex}.shuf $train_ex
fi

if [ ! -f $test_ex ] && [ ! $TEST_SPLIT_TRAIN ]; then
    echo "generating test data"
    python $ROOT/trees.py $TEST_TREE_TYPE --p1 $TEST_P1 --p2 $TEST_P2 --p3 $TEST_P3 --p4 $TEST_P4 --p5 $TEST_P5 --infix > $test_ex

    echo "deleting duplicate lines in test data"
    sort -u $test_ex | uniq $test_ex > ${test_ex}.dedup
    mv ${test_ex}.dedup $test_ex

    echo "removing all lines in the test data from the training data"
    awk '{if (f==1) { r[$0] } else if (! ($0 in r)) { print $0 } } ' f=1 $test_ex f=2 $train_ex > ${train_ex}.dedup
    mv ${train_ex}.dedup $train_ex
fi

if [ $TEST_SPLIT_TRAIN ]; then
    echo "spliting train file with the test file"
    num_ex=`wc -l $train_ex | cut -f 1 -d ' '`
    head -n $(( ($num_ex * $TEST_PER) / 100 )) $train_ex > $test_ex
    tail -n +$(( ($num_ex * $TEST_PER) / 100 )) $train_ex > ${train_ex}.split
    mv ${train_ex}.split ${train_ex}
fi

python $ROOT/main.py $train_ex $test_ex $SERIES $EXP_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE
