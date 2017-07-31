#! /usr/bin/env bash

export FAILED=0
echo "Running all tests and placing output into: test.out"

# Note the good parenthesis before the read and after the end so that failed is in one subshell
find build/bin -name "test_*" -perm /u+x | ( while read test; do
    #echo "Now running $test..." && ./$test && echo ""; 
    #echo "Now running $test..."
    ./$test > test.out 2>&1
    
    if [ $? != 0 ]; then
        echo "Failure in test: $test"
        let "FAILED++"
    fi
done
echo "Failed ${FAILED} test."; exit ${FAILED} )
