#!/bin/sh
autoreconf -f -Wall -v -i
# It bugs me it creates these useless files, so remove them...
rm -f config.h.in~
rm -f stamp-h1

# Also getting rid of INSTALL for now
rm -f INSTALL
