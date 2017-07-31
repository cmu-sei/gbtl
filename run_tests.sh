find build/bin -name "test_*" -perm /u+x | while read test; do echo "Now running $test..." && ./$test && echo ""; done
