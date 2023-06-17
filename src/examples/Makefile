CXX=g++-13

SOURCES += $(wildcard *.cpp)
TARGETS := $(patsubst %.cpp, %, $(SOURCES))

INCLUDE_FLAGS = -I../ -I../graphblas/platforms/sequential -I../interfaces/spec

CXXFLAGS = -std=c++2b -O3 $(INCLUDE_FLAGS)

all: $(TARGETS)

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(CXXFLAGS)

clean:
	rm -fv $(TARGETS)
