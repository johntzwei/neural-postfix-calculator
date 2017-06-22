# experiment notes
# tests whether 2 layer lstm with 10 hidden dim can predict
# 0 depth expressions of arbitrary length, e.g. [133] = 133

ROOT=/home/jwei/neural-postfix-calculator
NAME=`basename $0 | cut -d . -f 1`
EXP_DIR=$ROOT/experiments/$NAME

SERIES='preliminary_seq2seq'

EPOCHS=5
BATCH_SIZE=1
TRAIN_TREE_TYPE="generateAllTrees"
TRAIN_P1=0
TRAIN_P2=0
TRAIN_P3=0
TRAIN_P4=3
TRAIN_P5=0

#percentage out of 100
TEST_PER=50

TEST_TREE_TYPE="generateRandomTrees"
TEST_P1=0
TEST_P2=0
TEST_P3=0
TEST_P4=1
TEST_P5=0

mkdir -p $EXP_DIR

#train/test generation
train_ex="$ROOT/data/$NAME-train_${TRAIN_TREE_TYPE}_${TRAIN_P1}_${TRAIN_P2}_${TRAIN_P3}_${TRAIN_P4}_${TRAIN_P5}"
test_ex="$ROOT/data/$NAME-test_SPLIT_${TEST_PER}"

if [ ! -f $train_ex ]; then
    python $ROOT/trees.py $TRAIN_TREE_TYPE --p1 $TRAIN_P1 --p2 $TRAIN_P2 --p3 $TRAIN_P3 --p4 $TRAIN_P4 --p5 $TRAIN_P5 --infix > $train_ex

    shuf $train_ex > ${train_ex}.shuf
    mv ${train_ex}.shuf $train_ex

    num_ex=`wc -l $train_ex | cut -f 1 -d ' '`
    head -n $(( ($num_ex * $TEST_PER) / 100 )) $train_ex > $test_ex
    tail -n +$(( ($num_ex * $TEST_PER) / 100 )) $train_ex > ${train_ex}.split
    mv ${train_ex}.split ${train_ex}
fi

sort -u $test_ex | uniq $test_ex > ${test_ex}.dedup
mv ${test_ex}.dedup $test_ex

awk '{if (f==1) { r[$0] } else if (! ($0 in r)) { print $0 } } ' f=1 $test_ex f=2 $train_ex > ${train_ex}.dedup
mv ${train_ex}.dedup $train_ex

#train
python $ROOT/main.py $train_ex $test_ex $SERIES $EXP_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE
