SUFFIXES = .cu
.cu.$(OBJEXT):
	$(NVCC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(NVCC_CXXFLAGS) \
		$(AM_CPPFLAGS) $(CPPFLAGS) $(BACKEND_CPPFLAGS) $(AM_CXXFLAGS) \
		$(CXXFLAGS) -c $< -o $@

# TODO nvcc forwards c/c++ sources to system CC/CXX.  We need to stop that.
.cpp.$(OBJEXT):
	$(NVCC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(NVCC_CXXFLAGS) \
		$(AM_CPPFLAGS) $(CPPFLAGS) $(BACKEND_CPPFLAGS) $(AM_CXXFLAGS) \
		$(CXXFLAGS) -c $< -o $@
