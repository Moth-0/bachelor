# Variables
CXXFLAGS = -Wall -Werror -O -std=c++23
CXX = c++
LDLIBS = -lstdc++ -lm
RM = rm -f

all : Out.txt

Out.txt : main Makefile
	./main 2> out.log

main : main.o
	$(CXX) $(LDFLAGS) $(LDLIBS) -o $@ $^

main.o : main.cc matrix.h gaussian.h hamiltonian.h jacobian.h eigen.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

hyd : hyd.o
	$(CXX) $(LDFLAGS) $(LDLIBS) -o $@ $^
	./$@ 2> $@.log

hyd.o : hyd.cc matrix.h gaussian.h hamiltonian.h jacobian.h eigen.h
	$(CXX) $(CXXFLAGS) -c $< -o $@



clean :
	$(RM) *.o *.txt *.log