SUBDIRS = demo test
# TODO I need to update this once we have a documentation system set up.
dist_doc_DATA = README

if HAVE_DOXYGEN
directory = $(top_srcdir)/docs/man/man3/

doxyfile.stamp:
	$(DOXYGEN) Doxyfile
	echo Timestamp > doxyfile.stamp

CLEANFILES = doxyfile.stamp

doxygen: doxyfile.stamp
all-local: doxygen

clean-local:
	rm -rf $(top_srcdir)/docs
endif
