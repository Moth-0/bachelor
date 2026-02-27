# Variables
CXXFLAGS = -Wall -Werror -O -std=c++23 -fopenmp
CXX = c++
LDFLAGS += -fopenmp
LDLIBS = -lstdc++ -lm
RM = rm -f

all : main

QM : 

% : %.o
	$(CXX) $(LDFLAGS) $(LDLIBS) -o $@ $^
	./$@ 2> $@.log

%.o : %.cc matrix.h gaussian.h hamiltonian.h jacobian.h eigen.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean :
	$(RM) *.o *.txt *.log