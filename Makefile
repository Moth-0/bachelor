# Variables
CXXFLAGS = -Wall -Werror -O -std=c++23
CXX = c++
LDLIBS = -lstdc++ -lm
RM = rm -f

all : Out.txt

Out.txt : main Makefile
	./main > Out.txt
	cat Out.txt

main : main.o
	$(CXX) $(LDFLAGS) $(LDLIBS) -o $@ $^

main.o : main.cc matrix.h gaussian.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean :
	$(RM) *.o *.txt