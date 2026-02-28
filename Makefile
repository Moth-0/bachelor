MAKEFLAGS += -r
# Variables
CXXFLAGS = -Wall -Werror -O -std=c++23 -fopenmp
CXX = c++
LDFLAGS += -fopenmp
LDLIBS = -lstdc++ -lm
RM = rm -f

all : main

HEADERS = qm/matrix.h qm/gaussian.h qm/hamiltonian.h qm/jacobian.h qm/eigen.h

% : %.o 
	$(CXX) $(LDFLAGS) $(LDLIBS) -o $@ $^
	./$@ 1> $@.txt 2> $@.log

%.o : %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean :
	$(RM) *.o *.txt *.log