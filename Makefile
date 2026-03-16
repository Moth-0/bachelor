MAKEFLAGS += -r
# Variables
CXXFLAGS = -Wall -Werror -O3 -std=c++23 -fopenmp
CXX = c++
LDFLAGS += -fopenmp
LDLIBS = -lstdc++ -lm
RM = rm -f

all : main

HEADERS = qm/matrix.h qm/gaussian.h qm/hamiltonian.h qm/particle.h qm/jacobi.h qm/solver.h

heatmap.png : heatmap.dat Makefile
	echo '\
	set terminal pngcairo size 800,600 enhanced font "Arial,12";\
	set output "heatmap.png";\
	set title "Deuteron Binding Energy Heatmap";\
	set xlabel "Form Factor Width b (fm)";\
	set ylabel "Coupling Strength S (MeV)";\
	set cblabel "Binding Energy (MeV)";\
	set pm3d map;\
	set palette rgbformulae 22,13,-31;\
	set zrange [-5:0];\
	set cbrange [-5:0];\
	splot "heatmap.dat" using 1:2:3 title "";\
	' | tee log.fig1.gpi | gnuplot

% : %.o 
	$(CXX) $(LDFLAGS) $(LDLIBS) -o $@ $^

%.o : %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean :
	$(RM) *.o *.log *.dat *.gpi