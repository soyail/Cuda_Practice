# Compiler and flags
CXX = nvcc
CXXFLAGS = --generate-line-info -arch=sm_89 -O3

# Automatically find all .cu, .h, and .cc files
CU_SRCS := $(wildcard *.cu)
CC_SRCS := $(wildcard *.cc)
DEPS := $(wildcard *.h)
OBJS := $(CU_SRCS:.cu=.o) $(CC_SRCS:.cc=.o)

# Rule to compile .cu and .cc files to .o
%.o: %.cu $(DEPS)
	$(CXX) -c $(CXXFLAGS) $< -o $@

%.o: %.cc $(DEPS)
	$(CXX) -c $(CXXFLAGS) $< -o $@

# Link final executable
test: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Clean rule
.PHONY: clean
clean:
	rm -f *.o test
