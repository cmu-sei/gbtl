/bin/ls | awk '$0 !~ /(\..|^M)/ {print $0 }' - | while read demo; do echo "Now running $demo..." && ./$demo && echo ""; done
