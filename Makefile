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

convergence.png : conv_cla.data conv_rel.data Makefile
	echo '\
	set terminal pngcairo ;\
	set output "convergence.png" ;\
	set title "Convergence of SVM" ;\
	set xlabel "Points" ;\
	set ylabel "Energy" ;\
	set grid ;\
	plot "conv_cla.data" using 1:2 skip 1 with lines lw 2 lc rgb "black" title "Classic", \
		 "conv_rel.data" using 1:2 skip 1 with lines lw 2 lc rgb "red" title "Relativistic", \
		 -2.221 with lines dt 2 lc rgb "blue" title "Target" \
	' | tee plot.gpi | gnuplot

clean :
	$(RM) *.o *.log *.dat *.gpi