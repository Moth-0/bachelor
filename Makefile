MAKEFLAGS += -r
# Variables
CXXFLAGS = -Wall -Werror -O3 -std=c++23 -fopenmp
CXX = c++
LDFLAGS += -fopenmp
LDLIBS = -lstdc++ -lm
RM = rm -f

# Default parameters for full run
b_range ?= 100
b_form ?= 1.4
S ?= 1.0

all : deu nuc

HEADERS = qm/matrix.h qm/gaussian.h qm/operators.h qm/jacobi.h qm/solver.h SVM.h deuterium.h nucleus.h oscillator.h

% : %.o
	$(CXX) $(LDFLAGS) $(LDLIBS) -o $@ $^

%.o : %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Full automated run: nucleon -> extract E_self -> deuteron
run: nuc deu
	@echo "======================================================================"
	@echo "Step 1: Computing nucleon self-energy with b_range=$(b_range) b_form=$(b_form) S=$(S)"
	@echo "======================================================================"
	./nuc -n p -b_range $(b_range) -b_form $(b_form) -S $(S) > /tmp/nuc_output.txt 2>&1
	@E_self=$$(grep "Final E_self" /tmp/nuc_output.txt | awk '{print $$4}'); \
	echo ""; \
	echo "======================================================================";\
	echo "Step 2: Computing deuteron with E_self=$$E_self";\
	echo "======================================================================";\
	./deu -b_range $(b_range) -b_form $(b_form) -S $(S) -E_self $$E_self

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
	$(RM) *.o *.log *.dat *.gpi /tmp/nuc_output.txt

.PHONY: all clean run