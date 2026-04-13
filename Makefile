MAKEFLAGS += -r
# Variables
CXXFLAGS = -Wall -Werror -O3 -std=c++23 -fopenmp
CXX = c++
LDFLAGS += -fopenmp
LDLIBS = -lstdc++ -lm
RM = rm -f

all : deu 

HEADERS = qm/matrix.h qm/gaussian.h qm/operators.h qm/jacobi.h qm/solver.h deuterium.h proton.h

% : %.o 
	$(CXX) $(LDFLAGS) $(LDLIBS) -o $@ $^

%.o : %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

convergence.png: convergence_master.data Makefile
	echo '\
	set terminal pngcairo size 800,600 enhanced font "Arial,11" ;\
	set output "convergence.png" ;\
	set title "Convergence of SVM under Different Kinematics" ;\
	set xlabel "Optimization Iterations" ;\
	set ylabel "Ground State Energy (MeV)" ;\
	set grid ;\
	set key top right box ;\
	plot for [i=0:*] "convergence_master.data" index i using 1:2 with lines lw 2 title columnheader(2), \
	     -2.224 with lines dt 2 lw 2 lc rgb "purple" title "Target (-2.224 MeV)" \
	' | tee plot.gpi | gnuplot

clean :
	$(RM) *.o *.log *.dat *.gpi