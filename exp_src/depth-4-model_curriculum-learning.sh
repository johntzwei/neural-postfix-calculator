# experiment notes
# tests whether 2 layer lstm with 10 hidden dim can predict
# 0 depth expressions of arbitrary length, e.g. [133] = 133

ROOT=/home/jwei/neural-postfix-calculator
NAME=`basename $0 | cut -d . -f 1`
EXP_DIR=$ROOT/experiments/$NAME

SERIES='depth_4_model'

EPOCHS=20000
BATCH_SIZE=1
TRAIN_TREE_TYPE="generateAllTrees"
TRAIN_P1=0
TRAIN_P2=0
TRAIN_P3=0
TRAIN_P4=4
TRAIN_P5=0

TEST_PER=10
#percentage out of 100

TEST_TREE_TYPE="generateRandomTrees"
TEST_P1=0
TEST_P2=0
TEST_P3=0
TEST_P4=0
TEST_P5=0

shuffle () {
    shuf $1 > ${1}.shuf
    mv ${1}.shuf $1
}

remove_lines_from () {
    awk '{if (f==1) { r[$0] } else if (! ($0 in r)) { print $0 } } ' f=1 $1 f=2 $2 > ${2}.dedup
    mv ${2}.dedup $2
}

mkdir -p $EXP_DIR

#train/test generation
train_ex="$ROOT/data/$NAME-train_${TRAIN_TREE_TYPE}_${TRAIN_P1}_${TRAIN_P2}_${TRAIN_P3}_${TRAIN_P4}_${TRAIN_P5}"

if [[ 0 -eq $TEST_PER ]]; then
    test_ex="$ROOT/data/$NAME-test_${TEST_TREE_TYPE}_${TEST_P1}_${TEST_P2}_${TEST_P3}_${TEST_P4}_${TEST_P5}"
else
    test_ex="$ROOT/data/$NAME-test_SPLIT_${TEST_PER}"
fi

if [ ! -f $train_ex ]; then
    echo "training data will be saved to $train_ex"
    echo "generating train data"
    python $ROOT/trees.py $TRAIN_TREE_TYPE --p1 $TRAIN_P1 --p2 $TRAIN_P2 --p3 $TRAIN_P3 --p4 $TRAIN_P4 --p5 $TRAIN_P5 --infix > $train_ex

    echo "shuffling train data"
    shuf $train_ex > ${train_ex}.shuf
    mv ${train_ex}.shuf $train_ex
fi

if [ ! -f $test_ex ]; then
    echo "testing data will be saved to $test_ex"

    if [[ 0 -eq $TEST_PER ]]; then
        echo "generating test data"
        python $ROOT/trees.py $TEST_TREE_TYPE --p1 $TEST_P1 --p2 $TEST_P2 --p3 $TEST_P3 --p4 $TEST_P4 --p5 $TEST_P5 --infix > $test_ex
    else
        echo "spliting train file to the test file"
        num_ex=`wc -l $train_ex | cut -f 1 -d ' '`
        head -n $(( ($num_ex * $TEST_PER) / 100 )) $train_ex > $test_ex
        tail -n +$(( ($num_ex * $TEST_PER) / 100 )) $train_ex > ${train_ex}.split
        mv ${train_ex}.split ${train_ex}
    fi

    echo "deleting duplicate lines in test data"
    sort -u $test_ex | uniq $test_ex > ${test_ex}.dedup
    mv ${test_ex}.dedup $test_ex

    echo "removing all lines in the test data from the training data"
    awk '{if (f==1) { r[$0] } else if (! ($0 in r)) { print $0 } } ' f=1 $test_ex f=2 $train_ex > ${train_ex}.dedup
    mv ${train_ex}.dedup $train_ex
fi

#curriculum depth 2, 3, 4
train_ex_2="$ROOT/data/$NAME-train_${TRAIN_TREE_TYPE}_${TRAIN_P1}_${TRAIN_P2}_${TRAIN_P3}_2_${TRAIN_P5}"
python $ROOT/trees.py generateAllTrees --p1 $TRAIN_P1 --p2 $TRAIN_P2 --p3 $TRAIN_P3 --p4 2 --p5 $TRAIN_P5 --infix > $train_ex_2
train_ex_3="$ROOT/data/$NAME-train_${TRAIN_TREE_TYPE}_${TRAIN_P1}_${TRAIN_P2}_${TRAIN_P3}_3_${TRAIN_P5}"
python $ROOT/trees.py generateAllTrees --p1 $TRAIN_P1 --p2 $TRAIN_P2 --p3 $TRAIN_P3 --p4 3 --p5 $TRAIN_P5 --infix > $train_ex_3
train_ex_4="$ROOT/data/$NAME-train_${TRAIN_TREE_TYPE}_${TRAIN_P1}_${TRAIN_P2}_${TRAIN_P3}_4_${TRAIN_P5}"
python $ROOT/trees.py generateAllTrees --p1 $TRAIN_P1 --p2 $TRAIN_P2 --p3 $TRAIN_P3 --p4 4 --p5 $TRAIN_P5 --infix > $train_ex_4

remove_lines_from $test_ex $train_ex_2
remove_lines_from $test_ex $train_ex_3
remove_lines_from $test_ex $train_ex_4

python $ROOT/main.py $train_ex_2 $test_ex $SERIES $EXP_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE
python $ROOT/main.py $train_ex_3 $test_ex $SERIES $EXP_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE --model $ROOT/experiments/depth-4-model_curriculum-learning/seq2seq-1x30
python $ROOT/main.py $train_ex_4 $test_ex $SERIES $EXP_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE --model $ROOT/experiments/depth-4-model_curriculum-learning/seq2seq-1x30
