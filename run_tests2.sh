#! /usr/bin/env bash

# Call this script with an optional build directory name (defaults to ./build)

export FAILED=0
echo "Running all tests found in ${1-build}/bin and placing output into: test.out"
rm test.out

# Note the good parenthesis before the read and after the end so that failed is in one subshell
find ${1-build}/bin -name "test_*" -perm /u+x | ( while read test; do
    #echo "Now running $test..." && ./$test && echo ""; 
    echo "Now running $test..." >> test.out
    ./$test >> test.out 2>&1
    
    if [ $? != 0 ]; then
        echo "Failure in test: $test"
        let "FAILED++"
    fi
done
echo "Failed ${FAILED} test."; exit ${FAILED} )
