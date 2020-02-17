#/bin/sh

TEST_DIR="test"

for t in $(ls $TEST_DIR)
do
    echo -e "\n\t#### Executing tests in $t ####\n"
    julia "$TEST_DIR/$t"
done
