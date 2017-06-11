# experiment notes
# ...
# ...

ROOT=/home/johntzwei/Documents/uw2017/neural-postfix-calculator
NAME=`basename $0 | cut -d . -f 1`
EXP_DIR=$ROOT/experiments/$NAME

SERIES='preliminaries'

EPOCHS=1000
BATCH_SIZE=32
TRAIN_TREE_TYPE="generateRandomTreesFixedNodes"
TRAIN_P1=100000
TRAIN_P2=999999
TRAIN_P3=0
TRAIN_P4=1
TRAIN_P5=300000

TEST_TREE_TYPE="generateRandomTreesFixedNodes"
TEST_P1=100000
TEST_P2=999999
TEST_P3=0
TEST_P4=1
TEST_P5=3000

mkdir -p $EXP_DIR

#train/test generation
train_ex="$ROOT/data/$NAME-train_${TRAIN_TREE_TYPE}_${TRAIN_P1}_${TRAIN_P2}_${TRAIN_P3}_${TRAIN_P4}_${TRAIN_P5}"
test_ex="$ROOT/data/$NAME-test_${TEST_TREE_TYPE}_${TEST_P1}_${TEST_P2}_${TEST_P3}_${TEST_P4}_${TEST_P5}"

#remove duplicates
awk '{if (f==1) { r[$0] } else if (! ($0 in r)) { print $0 } } ' f=1 $test_ex f=2 $train_ex > ${test_ex}.dedup
mv ${test_ex}.dedup $test_ex

python $ROOT/trees.py $TRAIN_TREE_TYPE --p1 $TRAIN_P1 --p2 $TRAIN_P2 --p3 $TRAIN_P3 --p4 $TRAIN_P4 --p5 $TRAIN_P5 > $train_ex
python $ROOT/trees.py $TEST_TREE_TYPE --p1 $TEST_P1 --p2 $TEST_P2 --p3 $TEST_P3 --p4 $TEST_P4 --p5 $TEST_P5 > $test_ex

python $ROOT/main.py $train_ex $test_ex $SERIES $EXP_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE
