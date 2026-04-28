MAKEFLAGS += -r
# Variables
CXXFLAGS = -Wall -Werror -O3 -std=c++23 -fopenmp
CXX = c++
LDFLAGS += -fopenmp
LDLIBS = -lstdc++ -lm
RM = rm -f

all : deu

HEADERS = qm/matrix.h qm/gaussian.h qm/operators.h qm/jacobi.h qm/solver.h qm/serialization.h deuterium.h SVM.h

% : %.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o : %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: deu params
	@echo "======================================================================"
	@echo "Step 1: Computing binding energy with b_range=$(b_range) b_form=$(b_form) S=$(S)"
	@echo "======================================================================"
	./deu -b_range $(b_range) -b_form $(b_form) -S $(S) 
	@echo ""
	@echo "======================================================================"
	@echo "Step 2: Binary search for S with  b_range=$(b_range) b_form=$(b_form)"
	@echo "======================================================================"
	./params $(b_range) $(b_form)

convergence.png: convergence.data Makefile
	echo '\
	set terminal pngcairo size 800,600 enhanced font "Arial,11" ;\
	set output "convergence.png" ;\
	set title "Convergence of SVM under Different Kinematics" ;\
	set xlabel "SVM Iterations" ;\
	set ylabel "Ground State Energy (MeV)" ;\
	set grid ;\
	set key top right box ;\
	plot for [i=0:*] "convergence.data" index i using 1:2 with lines lw 2 title columnheader(2), \
	     -2.224 with lines dt 2 lw 2 lc rgb "purple" title "Target (-2.224 MeV)" \
	' | tee plot.gpi | gnuplot

clean :
	$(RM) *.o *.log *.dat *.gpi