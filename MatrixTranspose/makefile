CXX = nvcc
CXXFlAGS = --generate-line-info -arch=sm_89 -O3

DEPS = naive_matrix_transpose.h 1_kernel_matrix_transpose.h 2_kernel_matrix_transpose.h 3_kernel_matrix_transpose.h
OBJS = naive_matrix_transpose.o 1_kernel_matrix_transpose.o 2_kernel_matrix_transpose.o 3_kernel_matrix_transpose.o

%.o: %.cc $(DEPS)
	$(CXX) -c $(CXXFLAGS) $< -o $@

%.o: %.cu $(DEPS)
	$(CXX) -c $(CXXFlAGS) $< -o $@

profile_kernel: $(OBJS) profile_kernel.o
	$(CXX) $(CXXFLAGS) $^ -o profile_kernel

test: $(OBJS) test.o
	$(CXX) $(CXXFLAGS) $^ -o test

clean:
	rm -f *.o profile_kernel test